# hi hello
from __future__ import annotations;

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import serial
#michael was here :)å
logger = logging.getLogger(__name__)


#model for data coming from the reactor
@dataclass
class ReactorReading:
    raw_adc: int = 0
    electrode_a: int = 0
    electrode_b: int = 0
    time_ms: int = 0
    flag: str = "auto"
    received_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

class SerialBridge:

    """ 
    This class is to maintain a consistent serial connection to the arduino :P
    
    """
    
    BAUD = 115_200
    READ_TIMEOUT = 2.0

    # initializes the class attributes

    def __init__(self, port: str, baud: int = BAUD):
        self._port = port
        self._baud = baud
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._latest_reading = ReactorReading()
        self._connected = False
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # starts a connection and opens the serial port and starts the background reader thread.
    def connect(self) -> None:
        self._ser = serial.Serial(self._port, self._baud, timeout=self.READ_TIMEOUT)
        
        time.sleep(3.0)
        self._connected = True
        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True , name="serial-reader")
        self._reader_thread.start()
        logger.info("SerialBridge connected on port %s at baud %d", self._port, self._baud)
    
    #stops the reader thread and closes serial port
    def disconnect(self) -> None:
        self._stop_event.set()
        if self._reader_thread:
            self._reader_thread.join(timeout=3.0)
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._connected = False
        logger.info("SerialBridge Disconnected.")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *_):
        self.disconnect()
    

    # this function continously reads lines from the Arduino and updates the latest reading
    def _reader_loop(self) -> None:

        assert self._ser is not None
        while not self._stop_event.is_set():
            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                self._handle_incoming(line)
            except serial.SerialException as exc:
                logger.error("Serial read error: %s", exc)
                break
            except Exception as exc:
                logger.warning("Reader loop execution: %s", exc)
    
    #handles incoming data and retrieves values
    def _handle_incoming(self, line:str) -> None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Non-JSON from Arduino: %s", line)
            return
        
        msg_type = data.get("type")
        if msg_type == "reading":
            reading = ReactorReading(
                raw_adc = data.get("raw_adc", 0),
                electrode_a=data.get("electrode_a", 0),
                electrode_b=data.get("electrode_b", 0),
                time_ms = data.get("time_ms", 0),
                flag = data.get("flag", "auto"),
                received_at = time.time(),
            )

            with self._lock:
                self._latest_reading = reading
            logger.debug("Reading: %s", reading)
        
        elif msg_type == "error":
            logger.error("Arduino error: %s", data.get("msg", "unknown"))
        
        elif msg_type == "ready":
            logger.info("Arduino ready: %s", data.get("msg", ""))
    
    # serializes payload to JSON and sends over serial
    def _send(self, payload: dict) -> None:
        if not self._connected or self._ser is None:
            raise RuntimeError("SerialBridge is not connected")
        data = json.dumps(payload) + "\n"
        with self._lock:
            self._ser.write(data.encode("utf-8"))
            self._ser.flush()
        logger.debug("Sent: %s", data.strip())
    
    #retrieves the most recent reading, as we only need the most recent one.
    def get_latest_reading(self) -> dict:
        with self._lock:
            return self._latest_reading.to_dict()
    
    # sets both electrodes to value simultanesouly
    def set_electrodes(self, a: int, b: int) -> dict:
        a = max(0, min(4095, int(a)))
        b = max(0, min(4095, int(b)))
        self._send({"type": "command", "cmd": "set_electrodes", "a": a, "b": b})
        return {"ok": True, "electrode_a": a, "electrode_b": b}
    
    # sets electrode a only
    def set_electrode_a(self, value: int) -> dict:
        value = max(0, min(4095, int(value)))
        self._send({"type": "command", "cmd" : "set_electrode_a", "value": value})
        return {"ok": True, "electrode_a": value}
    
    # electrode b only
    def set_electrode_b(self, value: int) -> dict:
        value = max(0, min(4095, int(value)))
        self._send({"type": "command", "cmd" : "set_electrode_b", "value": value})
        return {"ok": True, "electrode_b": value}

    # updates runtime parameters on Arduino
    # only fields that arent none will be changed
    def set_params(
            self,
            target_adc: Optional[int] = None,
            max_delta: Optional[int] = None,
            spike_threshold: Optional[int] = None,
            low_threshold: Optional[int] = None,
            photo_min: Optional[int] = None,
            photo_max: Optional[int] = None,
            dac_min: Optional[int] = None,
            dac_max: Optional[int] = None,
    ) -> dict:
        
        payload: dict = {"type": "command", "cmd": "set_params"}
        if target_adc is not None:
            payload["target_adc"] = target_adc
        if max_delta is not None:
            payload["max_delta"] = max_delta
        if spike_threshold is not None:
            payload["spike_threshold"] = spike_threshold
        if low_threshold is not None:
            payload["low_threshold"] = low_threshold
        if photo_min is not None:
            payload["photo_min"] = photo_min
        if photo_max is not None:
            payload["photo_max"] = photo_max
        if dac_min is not None:
            payload["dac_min"] = dac_min
        if dac_max is not None:
            payload["dac_max"] = dac_max
        
        self._send(payload)
        params_sent = {k: v for k, v in payload.items() if k not in ("type", "cmd")}

        return {"ok": True, "updated_params": params_sent}

    def set_backup(self)->dict:
        # sets arduino to auto mode in case agents f up
        self._send({"type": "backup"})
        return {"ok": True, "mode": "auto"}

    def set_ai_mode(self) -> dict:
        # ai mode brah
        self._send({"type": "ai"})
        return {"ok": True, "mode": "ai"}


        


    
        

