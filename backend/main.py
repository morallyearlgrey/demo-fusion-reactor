"""
main.py

Entry point for the Fusion Reactor AI Pipeline.

Usage:
  python main.py --port /dev/ttyUSB0 (tells which port; /dev/cu.usbmodem14101 on mac)
  python main.py --port COM3 --model gemini-2.5-flash-lite --baud 115200
  python main.py --simulate                  # full pipeline, no Arduino needed (no arduino!)
  python main.py --simulate --target-adc 700 --max-delta 100 (how to override the default or smth)
"""

# to run:
# cd to fusion reactor
# python main.py [--port <device> | --simulate] [--model <name>] [--baud <rate>] [--target-adc <int>] [--max-delta <int>] [--max-errors <int>]

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from dotenv import load_dotenv

load_dotenv(".env")

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


# ── Simulated bridge ──────────────────────────────────────────────────────────
# simulated test run, see top on how to do 
#may need to be modified, lowkey vibecode slooooop
class SimulatedBridge:
    """
    Drop-in replacement for SerialBridge when no Arduino is connected.

    Physics model:
      - ADC responds to average electrode field strength
      - Beam transmission is non-monotonic — peaks around DAC 2800 (~8.2V)
        then drops off, matching real IEC beam focusing behaviour
      - Gaussian noise on every reading (~3 counts std dev after 10-sample avg)
      - Arc discharge spike simulation: random rare spike above 800

    This is realistic enough to test the full decision + improvement LLM loop.
    The DecisionAgent will need to sweep upward, find the peak, and hold near it.
    """

    PEAK_DAC      = 2800    # DAC value where beam transmission peaks
    PEAK_ADC      = 810     # max ADC at peak (just below spike_threshold)
    NOISE_STD     = 3.0     # gaussian noise std dev (after 10-sample averaging)
    SPIKE_CHANCE  = 0.002   # 0.2% chance of arc discharge spike per reading

    def __init__(self):
        self._electrode_a = 0
        self._electrode_b = 0
        self._mode        = "auto"

    def connect(self):
        logger.info("[SIM] Simulated bridge connected")

    def disconnect(self):
        logger.info("[SIM] Simulated bridge disconnected")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    def get_latest_reading(self) -> dict:
        import time
        import random
        import math

        avg_dac = (self._electrode_a + self._electrode_b) / 2.0

        # Non-monotonic beam transmission curve — gaussian hill centred at PEAK_DAC
        # Models real focusing behaviour: too low = no beam, too high = over-focused
        sigma     = 1200.0
        peak_frac = math.exp(-((avg_dac - self.PEAK_DAC) ** 2) / (2 * sigma ** 2))
        base_adc  = int(550 + peak_frac * (self.PEAK_ADC - 550))

        # Gaussian noise
        noise   = int(random.gauss(0, self.NOISE_STD))
        raw_adc = max(0, min(1023, base_adc + noise))

        # Rare arc discharge spike
        if random.random() < self.SPIKE_CHANCE:
            raw_adc = random.randint(801, 950)
            logger.warning("[SIM] Arc discharge spike simulated: adc=%d", raw_adc)

        return {
            "raw_adc":     raw_adc,
            "electrode_a": self._electrode_a,
            "electrode_b": self._electrode_b,
            "time_ms":     int(time.time() * 1000) % (2 ** 32),
            "flag":        self._mode,
            "received_at": time.time(),
        }

    def set_electrodes(self, a: int, b: int) -> dict:
        a = max(0, min(4095, int(a)))
        b = max(0, min(4095, int(b)))
        self._electrode_a = a
        self._electrode_b = b
        logger.info("[SIM] set_electrodes a=%d b=%d", a, b)
        return {"ok": True, "electrode_a": a, "electrode_b": b}

    def set_electrode_a(self, value: int) -> dict:
        value = max(0, min(4095, int(value)))
        self._electrode_a = value
        logger.info("[SIM] set_electrode_a value=%d", value)
        return {"ok": True, "electrode_a": value}

    def set_electrode_b(self, value: int) -> dict:
        value = max(0, min(4095, int(value)))
        self._electrode_b = value
        logger.info("[SIM] set_electrode_b value=%d", value)
        return {"ok": True, "electrode_b": value}

    def set_params(self, **kwargs) -> dict:
        logger.info("[SIM] set_params %s", kwargs)
        return {"ok": True, "updated_params": kwargs}

    def set_backup(self) -> dict:
        self._mode = "auto"
        logger.info("[SIM] switched to auto mode")
        return {"ok": True, "mode": "auto"}

    def set_ai_mode(self) -> dict:
        self._mode = "ai"
        logger.info("[SIM] switched to ai mode")
        return {"ok": True, "mode": "ai"}


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fusion Reactor AI Pipeline — Google ADK + Arduino"
    )
    parser.add_argument(
        "--port", type=str, default=None,
        help="Serial port for Arduino (e.g. /dev/ttyUSB0, COM3)"
    )
    parser.add_argument(
        "--baud", type=int, default=115200,
        help="Serial baud rate (default: 115200)"
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash-lite",
        help="Gemini model name (default: gemini-2.5-flash-lite)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run with simulated hardware — no Arduino required"
    )
    parser.add_argument(
        "--target-adc", type=int, default=752,
        help="Initial target ADC setpoint (default: 752)"
    )
    parser.add_argument(
        "--max-delta", type=int, default=200,
        help="Initial max DAC step per cycle (default: 200)"
    )
    parser.add_argument(
        "--max-errors", type=int, default=5,
        help="Consecutive cycle failures before shutdown (default: 5)"
    )
    return parser.parse_args()


# ── Async entry point ─────────────────────────────────────────────────────────
# called from main!! runs the whole ass pipeline
async def run(args: argparse.Namespace) -> None:
    from pipeline.fusion_pipeline import FusionPipelineRunner, DEFAULT_STATE

    # Start WebSocket server for frontend
    try:
        from websocket_bridge import start_websocket_server
        start_websocket_server(host="0.0.0.0", port=8000)
        logger.info("WebSocket server started for frontend connection")
    except Exception as e:
        logger.warning(f"Could not start WebSocket server: {e}")

    # Override DEFAULT_STATE with CLI args
    # Only override keys that CLI args explicitly control
    # Everything else stays at the DEFAULT_STATE values
    initial_state = DEFAULT_STATE.copy()
    initial_state["target_adc"] = args.target_adc
    initial_state["max_delta"]  = args.max_delta

    # Choose bridge
    # simulated bridge
    if args.simulate:
        logger.info("Starting in SIMULATE mode — no Arduino required")
        bridge = SimulatedBridge()
        bridge.connect()

    # serial bridge!
    else:
        if not args.port:
            logger.error("--port is required unless --simulate is used")
            sys.exit(1)
        from bridge.serial_bridge import SerialBridge
        bridge = SerialBridge(port=args.port, baud=args.baud)
        bridge.connect()

    runner = FusionPipelineRunner(
        bridge=bridge,
        model=args.model,
        initial_state=initial_state,
        max_consecutive_errors=args.max_errors,
    )

    # ── Signal handlers ───────────────────────────────────────────────────────
    # Signal handlers must be synchronous — they set a flag, not call async code.
    # The flag causes the while loop to exit cleanly, which hits the finally
    # block, which calls _shutdown().
    loop = asyncio.get_running_loop()

    def _handle_signal(sig):
        logger.info("Signal %s received — stopping pipeline after current cycle", sig.name)
        runner.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: _handle_signal(s))

    try:
        await runner.run_forever()
    finally:
        # Always disconnect bridge regardless of how run_forever exits
        bridge.disconnect()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Warn if no Google credentials are set — LLM agents will fail without them
    if not os.environ.get("GOOGLE_API_KEY") and \
       not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.warning(
            "Neither GOOGLE_API_KEY nor GOOGLE_APPLICATION_CREDENTIALS is set. "
            "DecisionAgent and ImprovementAgent (Gemini LLM calls) will fail. "
            "Set GOOGLE_API_KEY before running: export GOOGLE_API_KEY=your-key"
        )

    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()