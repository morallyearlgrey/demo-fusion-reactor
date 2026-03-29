"""
test_action_agent.py

Standalone test for ActionAgent in isolation.
No LLM, no AnalyzeAgent, no DecisionAgent, no full pipeline.

Manually injects a fake decision into session state and runs
ActionAgent directly, verifying it calls the correct bridge
functions and writes the correct last_action back to state.

Run with:
    python test_action_agent.py --simulate
    python test_action_agent.py --port /dev/cu.usbmodem1101
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import AsyncGenerator
from unittest.mock import MagicMock

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg: str):   print(f"  {GREEN}✓ PASS{RESET}  {msg}")
def fail(msg: str): print(f"  {RED}✗ FAIL{RESET}  {msg}")
def warn(msg: str): print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")
def info(msg: str): print(f"  {CYAN}ℹ{RESET}  {msg}")

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


# ── Fake InvocationContext ────────────────────────────────────────────────────
# Mimics the ADK InvocationContext just enough for ActionAgent to run.
# ActionAgent only uses ctx.session.state — nothing else.

class FakeSession:
    def __init__(self, state: dict):
        self.state = state

class FakeContext:
    def __init__(self, state: dict):
        self.session = FakeSession(state)


# ── Run agent helper ──────────────────────────────────────────────────────────

async def run_agent(agent, state: dict) -> dict:
    ctx = FakeContext(state)
    async for event in agent._run_async_impl(ctx):
        if hasattr(event, "content") and event.content:
            parts = event.content.parts  #  access as attribute, not dict
            for part in parts:
                if hasattr(part, "text"):
                    info(f"Agent event: {part.text}")  #  same here
    return ctx.session.state


# ── Individual tests ──────────────────────────────────────────────────────────

async def test_increase_both(agent, bridge, results: TestResults):
    header("Test 1 — increase_both")

    initial_a = 1000
    initial_b = 1000
    delta     = 200

    state = {
        "electrode_a": initial_a,
        "electrode_b": initial_b,
        "max_delta":   500,
        "cycle_count": 1,
        "decision": {
            "action":            "increase_both",
            "electrode_a_delta": delta,
            "electrode_b_delta": delta,
            "reasoning":         "ADC below target, increasing both electrodes",
            "confidence":        0.9,
        },
        "last_action": {},
    }

    info(f"Decision: increase_both  delta_a=+{delta}  delta_b=+{delta}")
    info(f"Starting electrodes: a={initial_a}  b={initial_b}")

    final_state = await run_agent(agent, state)
    last_action = final_state.get("last_action", {})

    info(f"last_action: {last_action}")

    expected_a = initial_a + delta
    expected_b = initial_b + delta

    results.record(last_action.get("ok") is True,               "ok=True")
    results.record(last_action.get("action") == "increase_both", "action='increase_both'")
    results.record(final_state.get("electrode_a") == expected_a, f"electrode_a updated to {expected_a}")
    results.record(final_state.get("electrode_b") == expected_b, f"electrode_b updated to {expected_b}")

    # Verify the bridge actually received the command
    reading = bridge.get_latest_reading()
    info(f"Bridge reading after action: a={reading['electrode_a']}  b={reading['electrode_b']}")
    results.record(reading["electrode_a"] == expected_a, f"bridge electrode_a={expected_a}")
    results.record(reading["electrode_b"] == expected_b, f"bridge electrode_b={expected_b}")

    # Check last_command_cycle was updated
    results.record(
        final_state.get("last_command_cycle") == 1,
        "last_command_cycle updated to cycle_count"
    )


async def test_decrease_both(agent, bridge, results: TestResults):
    header("Test 2 — decrease_both")

    initial_a = 2000
    initial_b = 2000
    delta     = -150

    state = {
        "electrode_a": initial_a,
        "electrode_b": initial_b,
        "max_delta":   500,
        "cycle_count": 2,
        "decision": {
            "action":            "decrease_both",
            "electrode_a_delta": delta,
            "electrode_b_delta": delta,
            "reasoning":         "ADC above target, decreasing both",
            "confidence":        0.85,
        },
        "last_action": {},
    }

    info(f"Decision: decrease_both  delta={delta}")
    final_state = await run_agent(agent, state)
    last_action = final_state.get("last_action", {})

    expected_a = initial_a + delta   # 2000 + (-150) = 1850
    expected_b = initial_b + delta

    results.record(last_action.get("ok") is True,               "ok=True")
    results.record(last_action.get("action") == "decrease_both", "action='decrease_both'")
    results.record(final_state.get("electrode_a") == expected_a, f"electrode_a={expected_a}")
    results.record(final_state.get("electrode_b") == expected_b, f"electrode_b={expected_b}")

    reading = bridge.get_latest_reading()
    results.record(reading["electrode_a"] == expected_a, f"bridge electrode_a={expected_a}")
    results.record(reading["electrode_b"] == expected_b, f"bridge electrode_b={expected_b}")


async def test_increase_a_only(agent, bridge, results: TestResults):
    header("Test 3 — increase_a (electrode A only)")

    initial_a = 1000
    initial_b = 1500
    delta     = 300

    state = {
        "electrode_a": initial_a,
        "electrode_b": initial_b,
        "max_delta":   500,
        "cycle_count": 3,
        "decision": {
            "action":            "increase_a",
            "electrode_a_delta": delta,
            "electrode_b_delta": 0,
            "reasoning":         "Asymmetric correction needed on A",
            "confidence":        0.75,
        },
        "last_action": {},
    }

    info(f"Decision: increase_a  delta_a=+{delta}  b unchanged at {initial_b}")
    final_state = await run_agent(agent, state)
    last_action = final_state.get("last_action", {})

    expected_a = initial_a + delta

    results.record(last_action.get("ok") is True,             "ok=True")
    results.record(last_action.get("action") == "increase_a", "action='increase_a'")
    results.record(final_state.get("electrode_a") == expected_a, f"electrode_a={expected_a}")
    # B should be unchanged in session state
    results.record(final_state.get("electrode_b") == initial_b,  f"electrode_b unchanged at {initial_b}")

    reading = bridge.get_latest_reading()
    results.record(reading["electrode_a"] == expected_a, f"bridge electrode_a={expected_a}")


async def test_do_nothing(agent, bridge, results: TestResults):
    header("Test 4 — do_nothing")

    initial_a = 2048
    initial_b = 2048

    state = {
        "electrode_a": initial_a,
        "electrode_b": initial_b,
        "max_delta":   200,
        "cycle_count": 4,
        "decision": {
            "action":            "do_nothing",
            "electrode_a_delta": 0,
            "electrode_b_delta": 0,
            "reasoning":         "System stable at target, no action needed",
            "confidence":        0.95,
        },
        "last_action": {},
    }

    info("Decision: do_nothing")
    final_state = await run_agent(agent, state)
    last_action = final_state.get("last_action", {})

    results.record(last_action.get("ok") is True,             "ok=True")
    results.record(last_action.get("action") == "do_nothing",  "action='do_nothing'")

    # Electrodes must not change
    results.record(final_state.get("electrode_a") == initial_a, f"electrode_a unchanged at {initial_a}")
    results.record(final_state.get("electrode_b") == initial_b, f"electrode_b unchanged at {initial_b}")

    # last_command_cycle must NOT be updated on do_nothing
    results.record(
        "last_command_cycle" not in final_state or final_state.get("last_command_cycle") == 0,
        "last_command_cycle not updated on do_nothing"
    )


async def test_emergency_backup(agent, bridge, results: TestResults):
    header("Test 5 — emergency_backup")

    state = {
        "electrode_a": 3000,
        "electrode_b": 3000,
        "max_delta":   200,
        "cycle_count": 5,
        "decision": {
            "action":            "emergency_backup",
            "electrode_a_delta": 0,
            "electrode_b_delta": 0,
            "reasoning":         "Spike detected, switching to auto mode",
            "confidence":        1.0,
        },
        "last_action": {},
    }

    info("Decision: emergency_backup")
    final_state = await run_agent(agent, state)
    last_action = final_state.get("last_action", {})

    info(f"last_action: {last_action}")

    results.record(last_action.get("ok") is True,                   "ok=True")
    results.record(last_action.get("action") == "emergency_backup",  "action='emergency_backup'")
    results.record(last_action.get("mode") == "auto",                "mode='auto' in last_action")

    # Verify bridge switched to auto mode
    time.sleep(0.3)
    reading = bridge.get_latest_reading()
    info(f"Bridge flag after emergency_backup: '{reading['flag']}'")
    results.record(reading["flag"] == "auto", "bridge flag='auto' after emergency_backup")


async def test_max_delta_clamp(agent, bridge, results: TestResults):
    header("Test 6 — max_delta safety clamp")

    initial_a = 2000
    initial_b = 2000
    max_delta  = 100   # hard limit

    state = {
        "electrode_a": initial_a,
        "electrode_b": initial_b,
        "max_delta":   max_delta,
        "cycle_count": 6,
        "decision": {
            "action":            "increase_both",
            "electrode_a_delta": 500,   # LLM tried to exceed max_delta
            "electrode_b_delta": 500,
            "reasoning":         "Large correction needed",
            "confidence":        0.8,
        },
        "last_action": {},
    }

    info(f"LLM delta=500 but max_delta={max_delta} — expect clamp to {max_delta}")
    final_state = await run_agent(agent, state)

    # Should be clamped to max_delta not the full 500
    expected_a = initial_a + max_delta   # 2000 + 100 = 2100
    expected_b = initial_b + max_delta

    results.record(
        final_state.get("electrode_a") == expected_a,
        f"electrode_a clamped to {expected_a} (not {initial_a + 500})"
    )
    results.record(
        final_state.get("electrode_b") == expected_b,
        f"electrode_b clamped to {expected_b} (not {initial_b + 500})"
    )

    reading = bridge.get_latest_reading()
    results.record(reading["electrode_a"] == expected_a, f"bridge electrode_a={expected_a}")
    results.record(reading["electrode_b"] == expected_b, f"bridge electrode_b={expected_b}")


async def test_balance_electrodes(agent, bridge, results: TestResults):
    header("Test 7 — balance_electrodes")

    initial_a = 1000
    initial_b = 3000
    expected  = (initial_a + initial_b) // 2   # 2000

    state = {
        "electrode_a": initial_a,
        "electrode_b": initial_b,
        "max_delta":   500,
        "cycle_count": 7,
        "decision": {
            "action":            "balance_electrodes",
            "electrode_a_delta": 0,
            "electrode_b_delta": 0,
            "reasoning":         "Large asymmetry detected, balancing",
            "confidence":        0.88,
        },
        "last_action": {},
    }

    info(f"Decision: balance_electrodes  a={initial_a}  b={initial_b}  → expected avg={expected}")
    final_state = await run_agent(agent, state)
    last_action = final_state.get("last_action", {})

    results.record(last_action.get("ok") is True,                    "ok=True")
    results.record(last_action.get("action") == "balance_electrodes", "action='balance_electrodes'")
    results.record(final_state.get("electrode_a") == expected,        f"electrode_a balanced to {expected}")
    results.record(final_state.get("electrode_b") == expected,        f"electrode_b balanced to {expected}")

    reading = bridge.get_latest_reading()
    results.record(reading["electrode_a"] == expected, f"bridge electrode_a={expected}")
    results.record(reading["electrode_b"] == expected, f"bridge electrode_b={expected}")


async def test_dac_ceiling_clamp(agent, bridge, results: TestResults):
    header("Test 8 — DAC ceiling clamp (electrode near 4095)")

    initial_a = 4000
    initial_b = 4000
    delta     = 200   # would push past 4095

    state = {
        "electrode_a": initial_a,
        "electrode_b": initial_b,
        "max_delta":   500,
        "cycle_count": 8,
        "decision": {
            "action":            "increase_both",
            "electrode_a_delta": delta,
            "electrode_b_delta": delta,
            "reasoning":         "Trying to increase beyond DAC max",
            "confidence":        0.7,
        },
        "last_action": {},
    }

    info(f"Electrodes at {initial_a}, delta={delta} — should clamp at 4095")
    final_state = await run_agent(agent, state)

    results.record(final_state.get("electrode_a") == 4095, "electrode_a clamped at 4095")
    results.record(final_state.get("electrode_b") == 4095, "electrode_b clamped at 4095")

    reading = bridge.get_latest_reading()
    results.record(reading["electrode_a"] == 4095, "bridge electrode_a=4095")
    results.record(reading["electrode_b"] == 4095, "bridge electrode_b=4095")


async def test_dac_floor_clamp(agent, bridge, results: TestResults):
    header("Test 9 — DAC floor clamp (electrode near 0)")

    initial_a = 50
    initial_b = 50
    delta     = -200   # would push below 0

    state = {
        "electrode_a": initial_a,
        "electrode_b": initial_b,
        "max_delta":   500,
        "cycle_count": 9,
        "decision": {
            "action":            "decrease_both",
            "electrode_a_delta": delta,
            "electrode_b_delta": delta,
            "reasoning":         "Trying to decrease below DAC min",
            "confidence":        0.7,
        },
        "last_action": {},
    }

    info(f"Electrodes at {initial_a}, delta={delta} — should clamp at 0")
    final_state = await run_agent(agent, state)

    results.record(final_state.get("electrode_a") == 0, "electrode_a clamped at 0")
    results.record(final_state.get("electrode_b") == 0, "electrode_b clamped at 0")

    reading = bridge.get_latest_reading()
    results.record(reading["electrode_a"] == 0, "bridge electrode_a=0")
    results.record(reading["electrode_b"] == 0, "bridge electrode_b=0")


# ── Simulated bridge ──────────────────────────────────────────────────────────

class SimulatedBridge:
    def __init__(self):
        self._a    = 0
        self._b    = 0
        self._mode = "auto"

    def connect(self):    info("SimulatedBridge connected")
    def disconnect(self): info("SimulatedBridge disconnected")
    def __enter__(self):  self.connect();    return self
    def __exit__(self, *_): self.disconnect()

    def get_latest_reading(self) -> dict:
        import random
        field = (self._a + self._b) / 2 / 4095
        adc   = int(550 + field * 270 + random.gauss(0, 3))
        return {
            "raw_adc":     max(0, min(1023, adc)),
            "electrode_a": self._a,
            "electrode_b": self._b,
            "time_ms":     int(time.time() * 1000) % (2**32),
            "flag":        self._mode,
            "received_at": time.time(),
        }

    def set_electrode_a(self, value: int) -> dict:
        value = max(0, min(4095, int(value)))
        self._a = value
        return {"ok": True, "electrode_a": value}

    def set_electrode_b(self, value: int) -> dict:
        value = max(0, min(4095, int(value)))
        self._b = value
        return {"ok": True, "electrode_b": value}

    def set_electrodes(self, a: int, b: int) -> dict:
        a = max(0, min(4095, int(a)))
        b = max(0, min(4095, int(b)))
        self._a, self._b = a, b
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

ALL_TESTS = [
    (1, "increase_both",              test_increase_both),
    (2, "decrease_both",              test_decrease_both),
    (3, "increase_a only",            test_increase_a_only),
    (4, "do_nothing",                 test_do_nothing),
    (5, "emergency_backup",           test_emergency_backup),
    (6, "max_delta clamp",            test_max_delta_clamp),
    (7, "balance_electrodes",         test_balance_electrodes),
    (8, "DAC ceiling clamp",          test_dac_ceiling_clamp),
    (9, "DAC floor clamp",            test_dac_floor_clamp),
]


def parse_args():
    p = argparse.ArgumentParser(description="ActionAgent isolation test suite")
    p.add_argument("--port",     type=str,  help="Serial port e.g. /dev/cu.usbmodem1101")
    p.add_argument("--baud",     type=int,  default=115200)
    p.add_argument("--simulate", action="store_true", help="Run without Arduino")
    p.add_argument("--test",     type=int,  default=None, help="Run only test N (1-9)")
    return p.parse_args()


async def run_all(args):
    results = TestResults()

    # Choose bridge
    if args.simulate:
        print(f"\n{YELLOW}SIMULATE mode — no Arduino required{RESET}")
        bridge = SimulatedBridge()
    else:
        if not args.port:
            print(f"{RED}Error: --port required (or use --simulate){RESET}")
            sys.exit(1)
        from bridge.serial_bridge import SerialBridge
        bridge = SerialBridge(port=args.port, baud=args.baud)

    print(f"\n{BOLD}Fusion Reactor — ActionAgent Isolation Test Suite{RESET}")
    print(f"Port: {args.port or 'SIMULATED'}  Baud: {args.baud}")

    with bridge:
        if not args.simulate:
            info("Waiting 2s for Arduino to boot...")
            time.sleep(2.0)
            # Switch Arduino to AI mode so it accepts commands
            bridge.set_ai_mode()
            info("Arduino switched to AI mode")
            time.sleep(0.3)

        # Import and init ActionAgent with the bridge
        from pipeline.tools.reactor_tools import init_tools
        from pipeline.agents.action_agent import ActionAgent
        init_tools(bridge)
        agent = ActionAgent()

        for num, name, fn in ALL_TESTS:
            if args.test is not None and args.test != num:
                continue
            try:
                await fn(agent, bridge, results)
            except Exception as exc:
                header(f"Test {num} — {name}")
                fail(f"Unexpected exception: {exc}")
                import traceback; traceback.print_exc()
                results.failed += 1

        # Always leave Arduino in safe auto mode
        if not args.simulate:
            bridge.set_backup()
            bridge.set_electrodes(0, 0)
            info("Arduino reset to auto mode, electrodes zeroed")

    results.summary()


def main():
    args = parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()