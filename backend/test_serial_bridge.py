"""
test_serial_bridge.py

Standalone test script for Arduino <-> SerialBridge communication.
No ADK, no agents, no LLM. Just raw serial comms verification.

Run with:
    python test_serial_bridge.py --port /dev/ttyUSB0
    python test_serial_bridge.py --port COM3
    python test_serial_bridge.py --simulate    # no Arduino needed

Tests covered:
    1. Connection and ready signal
    2. Reading telemetry from Arduino
    3. set_electrode_a
    4. set_electrode_b
    5. set_electrodes (both simultaneously)
    6. set_params (threshold updates)
    7. set_backup (switch to auto mode)
    8. set_ai_mode (switch to AI mode)
    9. Mode flag verification (reading["flag"] matches what we set)
   10. Spike and low threshold detection
   11. Reading rate (how many readings per second)
   12. Serial stability (sustained read over 10 seconds)
"""

from __future__ import annotations

import argparse
import sys
import time
import statistics
from typing import Optional
from bridge import serial_bridge


# ── ANSI colours for terminal output ─────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg: str):
    print(f"  {GREEN}✓ PASS{RESET}  {msg}")

def fail(msg: str):
    print(f"  {RED}✗ FAIL{RESET}  {msg}")

def warn(msg: str):
    print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")

def info(msg: str):
    print(f"  {CYAN}ℹ{RESET}  {msg}")

def header(msg: str):
    print(f"\n{BOLD}{msg}{RESET}")
    print("─" * 50)


# ── Result tracker ────────────────────────────────────────────────────────────

class TestResults:
    def __init__(self):
        self.passed  = 0
        self.failed  = 0
        self.skipped = 0

    def record(self, passed: bool, msg: str):
        if passed:
            self.passed += 1
            ok(msg)
        else:
            self.failed += 1
            fail(msg)

    def skip(self, msg: str):
        self.skipped += 1
        warn(f"SKIP — {msg}")

    def summary(self):
        total = self.passed + self.failed + self.skipped
        header("Test Summary")
        print(f"  Total:   {total}")
        print(f"  {GREEN}Passed:  {self.passed}{RESET}")
        print(f"  {RED}Failed:  {self.failed}{RESET}")
        print(f"  {YELLOW}Skipped: {self.skipped}{RESET}")
        if self.failed == 0:
            print(f"\n  {GREEN}{BOLD}All tests passed.{RESET}")
        else:
            print(f"\n  {RED}{BOLD}{self.failed} test(s) failed.{RESET}")
        return self.failed == 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def wait_for_fresh_reading(bridge, timeout: float = 3.0) -> Optional[dict]:
    """
    Wait until a new reading arrives after a command.
    Uses received_at timestamp to detect freshness.
    Returns None if timeout exceeded.
    """
    deadline = time.time() + timeout
    initial_time = bridge.get_latest_reading().get("received_at", 0)

    while time.time() < deadline:
        reading = bridge.get_latest_reading()
        if reading.get("received_at", 0) > initial_time:
            return reading
        time.sleep(0.1)

    return None


def read_stable(bridge, n: int = 5, delay: float = 0.2) -> list[dict]:
    """Collect n readings with a delay between each."""
    readings = []
    for _ in range(n):
        time.sleep(delay)
        readings.append(bridge.get_latest_reading())
    return readings


# ── Individual tests ──────────────────────────────────────────────────────────

def test_connection(bridge, results: TestResults):
    header("Test 1 — Connection")

    reading = bridge.get_latest_reading()
    info(f"Initial reading: {reading}")

    results.record(
        isinstance(reading, dict),
        "get_latest_reading() returns a dict"
    )
    results.record(
        "raw_adc" in reading,
        "reading contains 'raw_adc'"
    )
    results.record(
        "electrode_a" in reading,
        "reading contains 'electrode_a'"
    )
    results.record(
        "electrode_b" in reading,
        "reading contains 'electrode_b'"
    )
    results.record(
        "flag" in reading,
        "reading contains 'flag'"
    )
    results.record(
        "time_ms" in reading,
        "reading contains 'time_ms'"
    )
    results.record(
        "received_at" in reading,
        "reading contains 'received_at'"
    )


def test_reading_values(bridge, results: TestResults):
    header("Test 2 — Reading value ranges")

    reading = bridge.get_latest_reading()
    raw_adc    = reading["raw_adc"]
    electrode_a = reading["electrode_a"]
    electrode_b = reading["electrode_b"]
    flag        = reading["flag"]

    info(f"raw_adc={raw_adc}  electrode_a={electrode_a}  electrode_b={electrode_b}  flag={flag}")

    results.record(
        0 <= raw_adc <= 1023,
        f"raw_adc={raw_adc} is in valid range 0–1023"
    )
    results.record(
        0 <= electrode_a <= 4095,
        f"electrode_a={electrode_a} is in valid DAC range 0–4095"
    )
    results.record(
        0 <= electrode_b <= 4095,
        f"electrode_b={electrode_b} is in valid DAC range 0–4095"
    )
    results.record(
        flag in ("auto", "ai"),
        f"flag='{flag}' is one of 'auto' or 'ai'"
    )


def test_set_electrode_a(bridge, results: TestResults):
    header("Test 3 — set_electrode_a")

    test_value = 1000
    info(f"Sending set_electrode_a({test_value})")

    result = bridge.set_electrode_a(test_value)
    info(f"Bridge response: {result}")

    results.record(
        result.get("ok") is True,
        "set_electrode_a returned ok=True"
    )
    results.record(
        result.get("electrode_a") == test_value,
        f"response confirms electrode_a={test_value}"
    )

    # Wait for Arduino to echo back
    time.sleep(0.3)
    reading = bridge.get_latest_reading()
    info(f"Arduino reading after command: electrode_a={reading['electrode_a']}")

    results.record(
        reading["electrode_a"] == test_value,
        f"Arduino echoes electrode_a={test_value} in telemetry"
    )

    # Reset
    bridge.set_electrode_a(0)
    time.sleep(0.2)


def test_set_electrode_b(bridge, results: TestResults):
    header("Test 4 — set_electrode_b")

    test_value = 1500
    info(f"Sending set_electrode_b({test_value})")

    result = bridge.set_electrode_b(test_value)
    info(f"Bridge response: {result}")

    results.record(
        result.get("ok") is True,
        "set_electrode_b returned ok=True"
    )
    results.record(
        result.get("electrode_b") == test_value,
        f"response confirms electrode_b={test_value}"
    )

    time.sleep(0.3)
    reading = bridge.get_latest_reading()
    info(f"Arduino reading after command: electrode_b={reading['electrode_b']}")

    results.record(
        reading["electrode_b"] == test_value,
        f"Arduino echoes electrode_b={test_value} in telemetry"
    )

    bridge.set_electrode_b(0)
    time.sleep(0.2)


def test_set_electrodes_both(bridge, results: TestResults):
    header("Test 5 — set_electrodes (both simultaneously)")

    val_a = 2000
    val_b = 2500
    info(f"Sending set_electrodes(a={val_a}, b={val_b})")

    result = bridge.set_electrodes(val_a, val_b)
    info(f"Bridge response: {result}")

    results.record(
        result.get("ok") is True,
        "set_electrodes returned ok=True"
    )
    results.record(
        result.get("electrode_a") == val_a,
        f"response confirms electrode_a={val_a}"
    )
    results.record(
        result.get("electrode_b") == val_b,
        f"response confirms electrode_b={val_b}"
    )

    time.sleep(0.3)
    reading = bridge.get_latest_reading()
    info(f"Arduino reading: electrode_a={reading['electrode_a']}  electrode_b={reading['electrode_b']}")

    results.record(
        reading["electrode_a"] == val_a,
        f"Arduino echoes electrode_a={val_a}"
    )
    results.record(
        reading["electrode_b"] == val_b,
        f"Arduino echoes electrode_b={val_b}"
    )

    bridge.set_electrodes(0, 0)
    time.sleep(0.2)


def test_set_params(bridge, results: TestResults):
    header("Test 6 — set_params (threshold updates)")

    params = {
        "target_adc":      700,
        "max_delta":       150,
        "spike_threshold": 790,
        "low_threshold":   560,
    }
    info(f"Sending set_params: {params}")

    result = bridge.set_params(**params)
    info(f"Bridge response: {result}")

    results.record(
        result.get("ok") is True,
        "set_params returned ok=True"
    )
    results.record(
        "updated_params" in result,
        "response contains 'updated_params'"
    )

    # Verify each param is in the response
    updated = result.get("updated_params", {})
    for key, val in params.items():
        results.record(
            updated.get(key) == val,
            f"updated_params['{key}'] = {val}"
        )

    # Reset to safe defaults
    bridge.set_params(
        target_adc=752,
        max_delta=200,
        spike_threshold=800,
        low_threshold=580,
    )
    time.sleep(0.2)


def test_ai_mode(bridge, results: TestResults):
    header("Test 7 — set_ai_mode")

    info("Switching to AI mode")
    result = bridge.set_ai_mode()
    info(f"Bridge response: {result}")

    results.record(
        result.get("ok") is True,
        "set_ai_mode returned ok=True"
    )
    results.record(
        result.get("mode") == "ai",
        "response confirms mode='ai'"
    )

    # Wait for flag to appear in telemetry
    time.sleep(0.5)
    reading = bridge.get_latest_reading()
    info(f"Arduino flag after set_ai_mode: '{reading['flag']}'")

    results.record(
        reading["flag"] == "ai",
        "Arduino telemetry flag='ai' after set_ai_mode"
    )


def test_backup_mode(bridge, results: TestResults):
    header("Test 8 — set_backup")

    info("Switching to backup/auto mode")
    result = bridge.set_backup()
    info(f"Bridge response: {result}")

    results.record(
        result.get("ok") is True,
        "set_backup returned ok=True"
    )
    results.record(
        result.get("mode") == "auto",
        "response confirms mode='auto'"
    )

    time.sleep(0.5)
    reading = bridge.get_latest_reading()
    info(f"Arduino flag after set_backup: '{reading['flag']}'")

    results.record(
        reading["flag"] == "auto",
        "Arduino telemetry flag='auto' after set_backup"
    )


def test_mode_toggle(bridge, results: TestResults):
    header("Test 9 — Mode toggle (ai → auto → ai)")

    info("ai → auto → ai cycle")

    bridge.set_ai_mode()
    time.sleep(0.5)
    r1 = bridge.get_latest_reading()

    bridge.set_backup()
    time.sleep(0.5)
    r2 = bridge.get_latest_reading()

    bridge.set_ai_mode()
    time.sleep(0.5)
    r3 = bridge.get_latest_reading()

    info(f"Flags: {r1['flag']} → {r2['flag']} → {r3['flag']}")

    results.record(r1["flag"] == "ai",   "first switch: flag='ai'")
    results.record(r2["flag"] == "auto", "second switch: flag='auto'")
    results.record(r3["flag"] == "ai",   "third switch: flag='ai'")

    # Leave in auto for safety
    bridge.set_backup()
    time.sleep(0.2)


def test_electrode_clamping(bridge, results: TestResults):
    header("Test 10 — DAC value clamping (out-of-range inputs)")

    # Bridge should clamp these before sending to Arduino
    info("Sending electrode_a = -500 (should clamp to 0)")
    result = bridge.set_electrode_a(-500)
    results.record(
        result.get("electrode_a") == 0,
        "negative value clamped to 0"
    )

    info("Sending electrode_a = 9999 (should clamp to 4095)")
    result = bridge.set_electrode_a(9999)
    results.record(
        result.get("electrode_a") == 4095,
        "out-of-range value clamped to 4095"
    )

    bridge.set_electrode_a(0)
    time.sleep(0.2)


def test_reading_rate(bridge, results: TestResults):
    header("Test 11 — Reading rate")

    info("Collecting 20 readings over ~4 seconds to measure rate...")
    timestamps = []
    last_time  = bridge.get_latest_reading().get("received_at", 0)

    collected = 0
    deadline  = time.time() + 5.0

    while collected < 20 and time.time() < deadline:
        time.sleep(0.05)
        reading  = bridge.get_latest_reading()
        new_time = reading.get("received_at", 0)
        if new_time > last_time:
            timestamps.append(new_time)
            last_time = new_time
            collected += 1

    info(f"Collected {collected} readings")
    results.record(
        collected >= 10,
        f"received at least 10 readings in 5s (got {collected})"
    )

    if len(timestamps) >= 2:
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        mean_gap = statistics.mean(gaps)
        rate_hz  = 1.0 / mean_gap if mean_gap > 0 else 0
        info(f"Mean gap between readings: {mean_gap*1000:.1f}ms  ({rate_hz:.1f} Hz)")
        results.record(
            rate_hz >= 5,
            f"reading rate >= 5 Hz (measured {rate_hz:.1f} Hz)"
        )
    else:
        results.skip("not enough readings to measure rate")


def test_sustained_stability(bridge, results: TestResults):
    header("Test 12 — Sustained stability (10 seconds)")

    info("Reading for 10 seconds, checking for errors or gaps...")
    bridge.set_backup()
    time.sleep(0.2)

    readings  = []
    errors    = []
    deadline  = time.time() + 10.0
    last_time = bridge.get_latest_reading().get("received_at", 0)

    while time.time() < deadline:
        time.sleep(0.1)
        reading  = bridge.get_latest_reading()
        new_time = reading.get("received_at", 0)

        if new_time > last_time:
            readings.append(reading)
            last_time = new_time

            # Sanity checks on each reading
            if not (0 <= reading["raw_adc"] <= 1023):
                errors.append(f"out-of-range raw_adc: {reading['raw_adc']}")
            if reading["flag"] not in ("auto", "ai"):
                errors.append(f"invalid flag: {reading['flag']}")

    adc_values = [r["raw_adc"] for r in readings]
    info(f"Collected {len(readings)} readings over 10s")

    if adc_values:
        mean_adc = statistics.mean(adc_values)
        std_adc  = statistics.stdev(adc_values) if len(adc_values) > 1 else 0.0
        info(f"ADC  min={min(adc_values)}  max={max(adc_values)}  mean={mean_adc:.1f}  std={std_adc:.1f}")  

    results.record(
        len(readings) >= 50,
        f"received >= 50 readings in 10s (got {len(readings)})"
    )
    results.record(
        len(errors) == 0,
        f"no malformed readings (found {len(errors)} errors)"
    )

    if errors:
        for e in errors[:5]:
            warn(f"  error: {e}")


# ── Simulated bridge for testing without hardware ─────────────────────────────

class SimulatedBridge:
    """Mimics SerialBridge API without real serial port."""

    def __init__(self):
        self._a    = 0
        self._b    = 0
        self._mode = "auto"

    def connect(self):
        info("SimulatedBridge connected")

    def disconnect(self):
        info("SimulatedBridge disconnected")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    def get_latest_reading(self) -> dict:
        import random
        field = (self._a + self._b) / 2 / 4095
        adc   = int(550 + field * 270 + random.gauss(0, 3))
        adc   = max(0, min(1023, adc))
        return {
            "raw_adc":     adc,
            "electrode_a": self._a,
            "electrode_b": self._b,
            "time_ms":     int(time.time() * 1000) % (2**32),
            "flag":        self._mode,
            "received_at": time.time(),
        }

    def set_electrode_a(self, value: int) -> dict:
        value    = max(0, min(4095, int(value)))
        self._a  = value
        return {"ok": True, "electrode_a": value}

    def set_electrode_b(self, value: int) -> dict:
        value    = max(0, min(4095, int(value)))
        self._b  = value
        return {"ok": True, "electrode_b": value}

    def set_electrodes(self, a: int, b: int) -> dict:
        a       = max(0, min(4095, int(a)))
        b       = max(0, min(4095, int(b)))
        self._a = a
        self._b = b
        return {"ok": True, "electrode_a": a, "electrode_b": b}

    def set_params(self, **kwargs) -> dict:
        return {"ok": True, "updated_params": kwargs}

    def set_backup(self) -> dict:
        self._mode = "auto"
        return {"ok": True, "mode": "auto"}

    def set_ai_mode(self) -> dict:
        self._mode = "ai"
        return {"ok": True, "mode": "ai"}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Serial bridge test suite")
    parser.add_argument("--port",     type=str, help="Serial port e.g. /dev/ttyUSB0 or COM3")
    parser.add_argument("--baud",     type=int, default=115200)
    parser.add_argument("--simulate", action="store_true", help="Run without Arduino")
    parser.add_argument("--test",     type=int, default=None,
                        help="Run only a specific test number (1–12)")
    return parser.parse_args()


def main():
    args    = parse_args()
    results = TestResults()

    # Choose bridge
    if args.simulate:
        print(f"\n{YELLOW}Running in SIMULATE mode — no Arduino required{RESET}")
        bridge = SimulatedBridge()
    
    else:
        if not args.port:
            print(f"{RED}Error: --port is required unless --simulate is used{RESET}")
            sys.exit(1)
        from bridge.serial_bridge import SerialBridge
        bridge = SerialBridge(port=args.port, baud=args.baud)

    # All tests in order
    all_tests = [
        (1,  "Connection",                test_connection),
        (2,  "Reading value ranges",      test_reading_values),
        (3,  "set_electrode_a",           test_set_electrode_a),
        (4,  "set_electrode_b",           test_set_electrode_b),
        (5,  "set_electrodes (both)",     test_set_electrodes_both),
        (6,  "set_params",                test_set_params),
        (7,  "set_ai_mode",               test_ai_mode),
        (8,  "set_backup",                test_backup_mode),
        (9,  "Mode toggle",               test_mode_toggle),
        (10, "DAC clamping",              test_electrode_clamping),
        (11, "Reading rate",              test_reading_rate),
        (12, "Sustained stability (10s)", test_sustained_stability),
    ]

    print(f"\n{BOLD}Fusion Reactor — Serial Bridge Test Suite{RESET}")
    print(f"Port: {args.port or 'SIMULATED'}  Baud: {args.baud}")

    with bridge:
        # Give Arduino time to boot and send ready signal
        if not args.simulate:
            info("Waiting 2s for Arduino boot...")
            time.sleep(2.0)

        for num, name, fn in all_tests:
            if args.test is not None and args.test != num:
                continue
            try:
                fn(bridge, results)
            except Exception as exc:
                header(f"Test {num} — {name}")
                fail(f"Unexpected exception: {exc}")
                results.failed += 1

    passed = results.summary()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()