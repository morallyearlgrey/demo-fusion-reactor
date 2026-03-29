"""
test_analyze_agent.py

Isolated unit tests for AnalyzeAgent's computation functions.
No Arduino, no ADK pipeline, no LLM, no serial bridge needed.

Tests every pure-Python function in analyze_agent.py directly:
  - ewma()
  - coefficient_of_variation()
  - integral_error()
  - command_effectiveness()
  - is_settling()
  - full analysis dict via FakeContext

Run with:
    python test_analyze_agent.py
    python test_analyze_agent.py --test 3   # run only test group 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

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


# ── Import the functions directly from analyze_agent ─────────────────────────
# We import only the pure computation functions — not the class itself.
# This means no ADK, no serial bridge, no asyncio needed for most tests.

from pipeline.agents.analyze_agent import (
    ewma,
    coefficient_of_variation,
    integral_error,
    command_effectiveness,
    is_settling,
    AnalyzeAgent,
    SETTLE_TIMEOUT,
    SETTLING_THRESHOLD,
    NUM_READINGS,
    TARGET_ADC,
    MAX_DELTA,
    SPIKE_THRESHOLD,
    LOW_THRESHOLD,
    MAX_HISTORY,
    STATS_WINDOW,
)


# ── Fake context for full agent run ───────────────────────────────────────────

class FakeSession:
    def __init__(self, state: dict):
        self.state = state

class FakeContext:
    def __init__(self, state: dict):
        self.session = FakeSession(state)


def make_history(readings: list, target: int = TARGET_ADC) -> list:
    """
    Build a history list from a plain list of raw_adc values.
    Each entry gets the same target_adc so integral_error works correctly.
    """
    return [
        {
            "raw_adc":     v,
            "electrode_a": 2048,
            "electrode_b": 2048,
            "time_ms":     i * 50,
            "flag":        "ai",
            "host_time":   time.time(),
            "target_adc":  target,
        }
        for i, v in enumerate(readings)
    ]


# ── Fake get_latest_reading for full agent run ────────────────────────────────
# Patches reactor_tools so AnalyzeAgent can run without a bridge.

def patch_get_latest_reading(raw_adc: int, electrode_a: int = 2048,
                              electrode_b: int = 2048, flag: str = "ai"):
    """
    Patches the get_latest_reading reference inside analyze_agent
    not in reactor_tools — because analyze_agent imports it directly.
    """
    import pipeline.agents.analyze_agent as agent_module
    agent_module.get_latest_reading = lambda: {
        "raw_adc":     raw_adc,
        "electrode_a": electrode_a,
        "electrode_b": electrode_b,
        "time_ms":     12345,
        "flag":        flag,
        "received_at": time.time(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1 — ewma()
# ═════════════════════════════════════════════════════════════════════════════

def test_ewma(results: TestResults):
    header("Test Group 1 — ewma()")

    # Empty list
    result = ewma([])
    results.record(result == 0.0, f"ewma([]) = 0.0  (got {result})")

    # Single value — should return that value
    result = ewma([500])
    results.record(result == 500.0, f"ewma([500]) = 500.0  (got {result})")

    # Flat list — all same values, ewma should equal that value
    result = ewma([700, 700, 700, 700, 700])
    results.record(result == 700.0, f"ewma([700]*5) = 700.0  (got {result})")

    # Rising values — ewma should be below the last value (weighted toward recent)
    values = [600, 620, 640, 660, 680, 700]
    result = ewma(values, alpha=0.3)
    info(f"ewma(rising 600→700, alpha=0.3) = {result}")
    results.record(
        600 < result < 700,
        f"ewma of rising series is between first and last value ({result})"
    )
    results.record(
        result > 650,
        f"ewma of rising series is weighted toward recent values ({result} > 650)"
    )

    # Falling values — ewma should be above the last value
    values = [700, 680, 660, 640, 620, 600]
    result = ewma(values, alpha=0.3)
    info(f"ewma(falling 700→600, alpha=0.3) = {result}")
    results.record(
        600 < result < 700,
        f"ewma of falling series is between first and last value ({result})"
    )

    # Higher alpha = more responsive to recent values
    values = [500, 500, 500, 500, 900]   # spike at end
    low_alpha  = ewma(values, alpha=0.1)
    high_alpha = ewma(values, alpha=0.9)
    info(f"spike at end: alpha=0.1 → {low_alpha}  alpha=0.9 → {high_alpha}")
    results.record(
        high_alpha > low_alpha,
        f"higher alpha more responsive to spike ({high_alpha} > {low_alpha})"
    )

    # Stable near-target readings — should return close to target
    target = TARGET_ADC   # 752
    values = [748, 751, 753, 750, 752, 754, 749, 752]
    result = ewma(values, alpha=0.3)
    info(f"ewma of stable readings near {target} = {result}")
    results.record(
        abs(result - target) < 10,
        f"ewma of stable readings within 10 of target ({result} ≈ {target})"
    )


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2 — coefficient_of_variation()
# ═════════════════════════════════════════════════════════════════════════════

def test_cv(results: TestResults):
    header("Test Group 2 — coefficient_of_variation()")

    # Less than 2 values — should return 0.0
    result = coefficient_of_variation([])
    results.record(result == 0.0, f"cv([]) = 0.0  (got {result})")

    result = coefficient_of_variation([700])
    results.record(result == 0.0, f"cv([700]) = 0.0  (got {result})")

    # All same values — std dev is 0, cv should be 0.0
    result = coefficient_of_variation([700, 700, 700, 700])
    results.record(result == 0.0, f"cv([700]*4) = 0.0 — perfectly stable  (got {result})")

    # Mean of 0 — should return 0.0 to avoid division by zero
    result = coefficient_of_variation([0, 0, 0, 0])
    results.record(result == 0.0, f"cv([0]*4) = 0.0 — zero mean guard  (got {result})")

    # Very stable readings — cv should be < 0.02
    values = [750, 751, 749, 752, 750, 751, 750, 749]
    result = coefficient_of_variation(values)
    info(f"cv of very stable readings {values} = {result}")
    results.record(result < 0.02, f"cv={result} < 0.02 for stable readings")

    # Noisy readings — cv should be > 0.10
    values = [500, 700, 550, 800, 480, 750, 520, 780]
    result = coefficient_of_variation(values)
    info(f"cv of noisy readings {values} = {result}")
    results.record(result > 0.10, f"cv={result} > 0.10 for noisy readings")

    # Normal operating variance — cv should be 0.02–0.05
    values = [740, 748, 755, 743, 758, 746, 752, 760, 745, 751]
    result = coefficient_of_variation(values)
    info(f"cv of normal variance readings = {result}")
    results.record(
        0.01 < result < 0.08,
        f"cv={result} in normal operating range for moderate variance"
    )

    # CV is dimensionless — same relative spread gives same cv regardless of scale
    low_scale  = coefficient_of_variation([100, 102, 98, 101])
    high_scale = coefficient_of_variation([700, 714, 686, 707])
    info(f"cv at scale 100: {low_scale}  cv at scale 700: {high_scale}")
    results.record(
        abs(low_scale - high_scale) < 0.005,
        f"cv is dimensionless — same relative spread gives same result"
    )


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3 — integral_error()
# ═════════════════════════════════════════════════════════════════════════════

def test_integral_error(results: TestResults):
    header("Test Group 3 — integral_error()")

    # On target perfectly — integral should be 0
    history = make_history([TARGET_ADC] * 20)
    result = integral_error(history, window=20)
    results.record(result == 0.0, f"integral_error when on target = 0.0  (got {result})")

    # Chronic undershoot — all readings below target
    # Each reading is 100 below target, over 20 readings = -2000
    history = make_history([TARGET_ADC - 100] * 20)
    result = integral_error(history, window=20)
    info(f"chronic undershoot (-100 each): integral_error = {result}")
    results.record(result == -2000.0, f"chronic undershoot = -2000.0  (got {result})")
    results.record(result < 0, "negative integral = chronic undershoot")

    # Chronic overshoot — all readings above target
    history = make_history([TARGET_ADC + 100] * 20)
    result = integral_error(history, window=20)
    info(f"chronic overshoot (+100 each): integral_error = {result}")
    results.record(result == 2000.0, f"chronic overshoot = +2000.0  (got {result})")
    results.record(result > 0, "positive integral = chronic overshoot")

    # Oscillating around target — should be near zero
    values = [TARGET_ADC + 50, TARGET_ADC - 50] * 10   # alternating
    history = make_history(values)
    result = integral_error(history, window=20)
    info(f"oscillating around target: integral_error = {result}")
    results.record(result == 0.0, f"symmetric oscillation = 0.0  (got {result})")

    # Window smaller than history — only last N entries used
    history = make_history([TARGET_ADC - 200] * 10 + [TARGET_ADC] * 10)
    result_full   = integral_error(history, window=20)
    result_recent = integral_error(history, window=10)
    info(f"window=20: {result_full}  window=10 (recent only): {result_recent}")
    results.record(
        result_recent == 0.0,
        f"window=10 covers only on-target readings = 0.0  (got {result_recent})"
    )
    results.record(
        result_full < 0,
        f"window=20 includes undershoot readings = negative  (got {result_full})"
    )

    # Moving target — each entry uses its own target_adc
    # First 10 entries: target=700, reading=700 → error=0
    # Last 10 entries: target=800, reading=700 → error=-100 each = -1000
    history = (
        make_history([700] * 10, target=700) +
        make_history([700] * 10, target=800)
    )
    result = integral_error(history, window=20)
    info(f"moving target test: integral_error = {result}")
    results.record(
        result == -1000.0,
        f"moving target: only second half contributes error = -1000.0  (got {result})"
    )


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4 — command_effectiveness()
# ═════════════════════════════════════════════════════════════════════════════

def test_command_effectiveness(results: TestResults):
    header("Test Group 4 — command_effectiveness()")

    # No last action — should be unknown
    result = command_effectiveness({}, 700, 680)
    results.record(result == "unknown", f"empty last_action → 'unknown'  (got '{result}')")

    # do_nothing action — should be unknown
    result = command_effectiveness({"action": "do_nothing"}, 700, 680)
    results.record(result == "unknown", f"do_nothing → 'unknown'  (got '{result}')")

    # emergency_backup action — should be unknown
    result = command_effectiveness({"action": "emergency_backup"}, 700, 680)
    results.record(result == "unknown", f"emergency_backup → 'unknown'  (got '{result}')")

    # Net zero change — should be unknown (balance had no effect on net field)
    last_action = {"action": "increase_a", "electrode_a": 2048, "prev_a": 2048}
    result = command_effectiveness(last_action, 700, 680)
    results.record(result == "unknown", f"net=0 change → 'unknown'  (got '{result}')")

    # Effective — increased electrodes, ADC went up
    last_action = {
        "action":      "increase_both",
        "electrode_a": 2248,   # was 2048, increased by 200
        "prev_a":      2048,
        "electrode_b": 2248,
        "prev_b":      2048,
    }
    result = command_effectiveness(last_action, current_adc=720, prev_adc=680)
    info(f"increase command, ADC rose 680→720: {result}")
    results.record(result == "effective", f"increase + ADC rose → 'effective'  (got '{result}')")

    # Effective — decreased electrodes, ADC went down
    last_action = {
        "action":      "decrease_both",
        "electrode_a": 1848,   # was 2048, decreased by 200
        "prev_a":      2048,
        "electrode_b": 1848,
        "prev_b":      2048,
    }
    result = command_effectiveness(last_action, current_adc=640, prev_adc=700)
    info(f"decrease command, ADC fell 700→640: {result}")
    results.record(result == "effective", f"decrease + ADC fell → 'effective'  (got '{result}')")

    # Ineffective — increased electrodes but ADC didn't move
    last_action = {
        "action":      "increase_both",
        "electrode_a": 2248,
        "prev_a":      2048,
        "electrode_b": 2248,
        "prev_b":      2048,
    }
    result = command_effectiveness(last_action, current_adc=700, prev_adc=700)
    info(f"increase command, ADC unchanged 700→700: {result}")
    results.record(result == "ineffective", f"increase + ADC unchanged → 'ineffective'  (got '{result}')")

    # Reversed — increased electrodes but ADC went DOWN (hardware anomaly)
    last_action = {
        "action":      "increase_both",
        "electrode_a": 2248,
        "prev_a":      2048,
        "electrode_b": 2248,
        "prev_b":      2048,
    }
    result = command_effectiveness(last_action, current_adc=660, prev_adc=700)
    info(f"increase command, ADC fell 700→660: {result}")
    results.record(result == "reversed", f"increase + ADC fell → 'reversed'  (got '{result}')")

    # Reversed — decreased electrodes but ADC went UP
    last_action = {
        "action":      "decrease_both",
        "electrode_a": 1848,
        "prev_a":      2048,
        "electrode_b": 1848,
        "prev_b":      2048,
    }
    result = command_effectiveness(last_action, current_adc=740, prev_adc=700)
    info(f"decrease command, ADC rose 700→740: {result}")
    results.record(result == "reversed", f"decrease + ADC rose → 'reversed'  (got '{result}')")

    # Single electrode change — only A changed
    last_action = {
        "action":      "increase_a",
        "electrode_a": 2248,
        "prev_a":      2048,
        "electrode_b": 2048,   # B unchanged
        "prev_b":      2048,
    }
    result = command_effectiveness(last_action, current_adc=720, prev_adc=690)
    results.record(result == "effective", f"increase_a + ADC rose → 'effective'  (got '{result}')")


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 5 — is_settling()
# ═════════════════════════════════════════════════════════════════════════════

def test_is_settling(results: TestResults):
    header("Test Group 5 — is_settling()")

    # Not enough readings yet — should be True (assume settling)
    result = is_settling(recent=[700, 702], cycles_since_command=1)
    results.record(result is True, f"len(recent)<{NUM_READINGS} → True  (got {result})")

    # Past timeout — should be False regardless of tail range
    noisy_recent = [600, 650, 700, 750, 800] * 4
    result = is_settling(
        recent=noisy_recent,
        cycles_since_command=SETTLE_TIMEOUT + 1,   # past timeout
        threshold=SETTLING_THRESHOLD,
        settle_timeout=SETTLE_TIMEOUT,
    )
    results.record(result is False, f"cycles_since > settle_timeout → False  (got {result})")

    # Within timeout, signal still moving (tail range > threshold)
    # tail = [680, 695, 710, 725], range = 45 > 5.0
    recent = [650, 660, 670, 680, 695, 710, 725]
    result = is_settling(
        recent=recent,
        cycles_since_command=2,
        threshold=5.0,
        settle_timeout=8,
    )
    info(f"tail={recent[-NUM_READINGS:]}  range={max(recent[-NUM_READINGS:]) - min(recent[-NUM_READINGS:])}")
    results.record(result is True, f"recent command + large tail range → True  (got {result})")

    # Within timeout but signal has settled (tail range < threshold)
    # tail = [750, 751, 750, 752], range = 2 < 5.0
    recent = [700, 720, 740, 748, 750, 751, 750, 752]
    result = is_settling(
        recent=recent,
        cycles_since_command=3,
        threshold=5.0,
        settle_timeout=8,
    )
    info(f"tail={recent[-NUM_READINGS:]}  range={max(recent[-NUM_READINGS:]) - min(recent[-NUM_READINGS:])}")
    results.record(result is False, f"recent command but settled tail → False  (got {result})")

    # Noise at threshold boundary — exactly at threshold should NOT settle
    # range = 5.0 is NOT > 5.0
    recent = [750, 750, 750, 750, 748, 751, 748, 753]
    tail = recent[-NUM_READINGS:]
    tail_range = max(tail) - min(tail)
    result = is_settling(
        recent=recent,
        cycles_since_command=2,
        threshold=5.0,
        settle_timeout=8,
    )
    info(f"tail range = {tail_range}, threshold = 5.0")
    results.record(
        result is (tail_range > 5.0),
        f"tail_range={tail_range} > 5.0 → {tail_range > 5.0}  (got {result})"
    )

    # No command ever fired (cycles_since = large number) — should force False
    result = is_settling(
        recent=[600, 650, 700, 750, 800] * 4,
        cycles_since_command=100,
        threshold=5.0,
        settle_timeout=8,
    )
    results.record(result is False, f"no command ever fired (cycles=100) → False  (got {result})")

    # Exactly at timeout boundary — cycles_since == settle_timeout is still False
    result = is_settling(
        recent=[600, 700, 800, 900] * 4,  # huge range, would normally be True
        cycles_since_command=SETTLE_TIMEOUT,
        threshold=5.0,
        settle_timeout=SETTLE_TIMEOUT,
    )
    results.record(result is False, f"cycles_since == settle_timeout → False  (got {result})")

    # One below timeout — should still check tail range
    recent = [700, 710, 720, 730, 740, 750, 760, 770]
    result = is_settling(
        recent=recent,
        cycles_since_command=SETTLE_TIMEOUT - 1,
        threshold=5.0,
        settle_timeout=SETTLE_TIMEOUT,
    )
    tail_range = max(recent[-NUM_READINGS:]) - min(recent[-NUM_READINGS:])
    info(f"one below timeout, tail_range={tail_range}")
    results.record(result is True, f"one below timeout + large range → True  (got {result})")


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 6 — full AnalyzeAgent run via FakeContext
# ═════════════════════════════════════════════════════════════════════════════

async def test_full_agent_run(results: TestResults):
    header("Test Group 6 — full AnalyzeAgent run (FakeContext)")

    agent = AnalyzeAgent()

    # ── Subtest A: first cycle, no history ────────────────────────────────
    info("Subtest A — first cycle, no history")
    patch_get_latest_reading(raw_adc=620, electrode_a=2048, electrode_b=2048)

    state = {
        "target_adc":         TARGET_ADC,
        "max_delta":          MAX_DELTA,
        "spike_threshold":    SPIKE_THRESHOLD,
        "low_threshold":      LOW_THRESHOLD,
        "settle_timeout":     SETTLE_TIMEOUT,
        "settling_threshold": SETTLING_THRESHOLD,
        "cycle_count":        0,
        "last_command_cycle": 0,
        "history":            [],
        "analysis":           {},
        "last_action":        {},
    }

    ctx = FakeContext(state)
    async for event in agent._run_async_impl(ctx):
        if hasattr(event.content, 'parts') and event.content.parts:
            info(f"Event: {event.content.parts[0].text}")

    analysis = ctx.session.state.get("analysis", {})

    results.record(ctx.session.state["cycle_count"] == 1,    "cycle_count incremented to 1")
    results.record(ctx.session.state["raw_adc"] == 620,      "raw_adc saved to state")
    results.record(ctx.session.state["electrode_a"] == 2048, "electrode_a saved to state")
    results.record(len(ctx.session.state["history"]) == 1,   "history has 1 entry")
    results.record("analysis" in ctx.session.state,          "analysis written to state")
    results.record(analysis.get("raw_adc") == 620,           "analysis.raw_adc=620")
    results.record(analysis.get("target_adc") == TARGET_ADC, f"analysis.target_adc={TARGET_ADC}")
    results.record(
        analysis.get("error_from_target") == 620 - TARGET_ADC,
        f"error_from_target={620 - TARGET_ADC}"
    )
    results.record(analysis.get("is_spike") is False, "is_spike=False (620 < 800)")
    results.record(analysis.get("is_low") is True,    "is_low=True (620 < 580? no...)")

    # 620 > LOW_THRESHOLD(580) so is_low should be False
    results.record(analysis.get("is_low") is False,   f"is_low=False (620 > {LOW_THRESHOLD})")

    # ── Subtest B: spike detection ─────────────────────────────────────────
    info("Subtest B — spike detection")
    patch_get_latest_reading(raw_adc=850)  # above SPIKE_THRESHOLD=800

    state2 = {
        "target_adc":         TARGET_ADC,
        "max_delta":          MAX_DELTA,
        "spike_threshold":    SPIKE_THRESHOLD,
        "low_threshold":      LOW_THRESHOLD,
        "settle_timeout":     SETTLE_TIMEOUT,
        "settling_threshold": SETTLING_THRESHOLD,
        "cycle_count":        0,
        "last_command_cycle": 0,
        "history":            [],
        "analysis":           {},
        "last_action":        {},
    }
    ctx2 = FakeContext(state2)
    async for _ in agent._run_async_impl(ctx2):
        pass

    analysis2 = ctx2.session.state.get("analysis", {})
    results.record(analysis2.get("is_spike") is True, f"is_spike=True (850 > {SPIKE_THRESHOLD})")
    results.record(analysis2.get("is_low") is False,  f"is_low=False (850 > {LOW_THRESHOLD})")

    # ── Subtest C: low signal detection ───────────────────────────────────
    info("Subtest C — low signal detection")
    patch_get_latest_reading(raw_adc=400)  # below LOW_THRESHOLD=580

    state3 = {
        "target_adc":         TARGET_ADC,
        "max_delta":          MAX_DELTA,
        "spike_threshold":    SPIKE_THRESHOLD,
        "low_threshold":      LOW_THRESHOLD,
        "settle_timeout":     SETTLE_TIMEOUT,
        "settling_threshold": SETTLING_THRESHOLD,
        "cycle_count":        0,
        "last_command_cycle": 0,
        "history":            [],
        "analysis":           {},
        "last_action":        {},
    }
    ctx3 = FakeContext(state3)
    async for _ in agent._run_async_impl(ctx3):
        pass

    analysis3 = ctx3.session.state.get("analysis", {})
    results.record(analysis3.get("is_low") is True,   f"is_low=True (400 < {LOW_THRESHOLD})")
    results.record(analysis3.get("is_spike") is False, f"is_spike=False (400 < {SPIKE_THRESHOLD})")

    # ── Subtest D: history cap ─────────────────────────────────────────────
    info(f"Subtest D — history cap at MAX_HISTORY={MAX_HISTORY}")
    patch_get_latest_reading(raw_adc=700)

    # Pre-fill history to MAX_HISTORY
    existing_history = make_history([700] * MAX_HISTORY)
    state4 = {
        "target_adc":         TARGET_ADC,
        "max_delta":          MAX_DELTA,
        "spike_threshold":    SPIKE_THRESHOLD,
        "low_threshold":      LOW_THRESHOLD,
        "settle_timeout":     SETTLE_TIMEOUT,
        "settling_threshold": SETTLING_THRESHOLD,
        "cycle_count":        MAX_HISTORY,
        "last_command_cycle": 0,
        "history":            existing_history,
        "analysis":           {},
        "last_action":        {},
    }
    ctx4 = FakeContext(state4)
    async for _ in agent._run_async_impl(ctx4):
        pass

    results.record(
        len(ctx4.session.state["history"]) == MAX_HISTORY,
        f"history capped at {MAX_HISTORY} after overflow (got {len(ctx4.session.state['history'])})"
    )

    # ── Subtest E: target change detection ────────────────────────────────
    info("Subtest E — target change resets settle timer")
    patch_get_latest_reading(raw_adc=700)

    state5 = {
        "target_adc":         800,   # new target — different from analysis snapshot
        "max_delta":          MAX_DELTA,
        "spike_threshold":    SPIKE_THRESHOLD,
        "low_threshold":      LOW_THRESHOLD,
        "settle_timeout":     SETTLE_TIMEOUT,
        "settling_threshold": SETTLING_THRESHOLD,
        "cycle_count":        10,
        "last_command_cycle": 0,
        "history":            [],
        "analysis":           {"target_adc": 750},  # old target in previous analysis
        "last_action":        {},
    }
    ctx5 = FakeContext(state5)
    async for _ in agent._run_async_impl(ctx5):
        pass

    analysis5 = ctx5.session.state.get("analysis", {})
    results.record(
        analysis5.get("target_changed") is True,
        "target_changed=True when target_adc changed from 750→800"
    )
    results.record(
        ctx5.session.state.get("last_command_cycle") == 11,
        f"last_command_cycle reset to cycle_count=11 on target change"
    )

    # ── Subtest F: analysis dict has all required keys ─────────────────────
    info("Subtest F — analysis dict completeness")
    patch_get_latest_reading(raw_adc=700)

    state6 = {
        "target_adc":         TARGET_ADC,
        "max_delta":          MAX_DELTA,
        "spike_threshold":    SPIKE_THRESHOLD,
        "low_threshold":      LOW_THRESHOLD,
        "settle_timeout":     SETTLE_TIMEOUT,
        "settling_threshold": SETTLING_THRESHOLD,
        "cycle_count":        0,
        "last_command_cycle": 0,
        "history":            [],
        "analysis":           {},
        "last_action":        {},
    }
    ctx6 = FakeContext(state6)
    async for _ in agent._run_async_impl(ctx6):
        pass

    analysis6 = ctx6.session.state.get("analysis", {})
    required_keys = [
        "timestamp", "cycle_count",
        "raw_adc", "electrode_a", "electrode_b", "electrode_diff", "flag",
        "target_adc", "target_changed",
        "error_from_target", "steady_state_error", "integral_error",
        "ewma_adc", "std_adc", "cv", "roc",
        "is_settling", "cycles_since_command", "command_effectiveness",
        "is_spike", "is_low",
        "tuning_params", "history_length",
    ]
    for key in required_keys:
        results.record(key in analysis6, f"analysis has key '{key}'")


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 7 — constants sanity check
# ═════════════════════════════════════════════════════════════════════════════

def test_constants(results: TestResults):
    header("Test Group 7 — constants sanity check")

    # Hardware bounds
    results.record(0 < TARGET_ADC <= 1023,        f"TARGET_ADC={TARGET_ADC} in valid ADC range")
    results.record(0 < MAX_DELTA <= 4095,          f"MAX_DELTA={MAX_DELTA} in valid DAC range")
    results.record(0 < SPIKE_THRESHOLD <= 1023,    f"SPIKE_THRESHOLD={SPIKE_THRESHOLD} in valid range")
    results.record(0 < LOW_THRESHOLD <= 1023,      f"LOW_THRESHOLD={LOW_THRESHOLD} in valid range")

    # Logical ordering
    results.record(
        LOW_THRESHOLD < TARGET_ADC,
        f"LOW_THRESHOLD({LOW_THRESHOLD}) < TARGET_ADC({TARGET_ADC})"
    )
    results.record(
        TARGET_ADC < SPIKE_THRESHOLD,
        f"TARGET_ADC({TARGET_ADC}) < SPIKE_THRESHOLD({SPIKE_THRESHOLD})"
    )
    results.record(
        LOW_THRESHOLD < SPIKE_THRESHOLD,
        f"LOW_THRESHOLD({LOW_THRESHOLD}) < SPIKE_THRESHOLD({SPIKE_THRESHOLD})"
    )

    # Settling config
    results.record(SETTLE_TIMEOUT > 0,       f"SETTLE_TIMEOUT={SETTLE_TIMEOUT} > 0")
    results.record(SETTLING_THRESHOLD > 0,   f"SETTLING_THRESHOLD={SETTLING_THRESHOLD} > 0")
    results.record(NUM_READINGS >= 2,        f"NUM_READINGS={NUM_READINGS} >= 2")

    # History and window
    results.record(STATS_WINDOW <= MAX_HISTORY, f"STATS_WINDOW({STATS_WINDOW}) <= MAX_HISTORY({MAX_HISTORY})")
    results.record(NUM_READINGS <= STATS_WINDOW, f"NUM_READINGS({NUM_READINGS}) <= STATS_WINDOW({STATS_WINDOW})")

    # Target is in usable photodiode range 550-820
    results.record(550 <= TARGET_ADC <= 820,
        f"TARGET_ADC={TARGET_ADC} within photodiode usable range 550-820"
    )
    # Spike threshold below photodiode max 820
    results.record(SPIKE_THRESHOLD <= 820,
        f"SPIKE_THRESHOLD={SPIKE_THRESHOLD} <= photodiode max 820"
    )
    # Low threshold above photodiode min 550
    results.record(LOW_THRESHOLD >= 550,
        f"LOW_THRESHOLD={LOW_THRESHOLD} >= photodiode min 550"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    (1, "ewma()",                   lambda r: test_ewma(r)),
    (2, "coefficient_of_variation()", lambda r: test_cv(r)),
    (3, "integral_error()",         lambda r: test_integral_error(r)),
    (4, "command_effectiveness()",  lambda r: test_command_effectiveness(r)),
    (5, "is_settling()",            lambda r: test_is_settling(r)),
    (6, "full AnalyzeAgent run",    None),   # async, handled separately
    (7, "constants sanity check",   lambda r: test_constants(r)),
]


def parse_args():
    p = argparse.ArgumentParser(description="AnalyzeAgent isolation test suite")
    p.add_argument("--test", type=int, default=None, help="Run only test group N (1-7)")
    return p.parse_args()


async def run_all(args):
    results = TestResults()

    print(f"\n{BOLD}Fusion Reactor — AnalyzeAgent Isolation Test Suite{RESET}")
    print("No Arduino, no ADK, no LLM required\n")

    for num, name, fn in ALL_TESTS:
        if args.test is not None and args.test != num:
            continue
        try:
            if num == 6:
                await test_full_agent_run(results)
            else:
                fn(results)
        except Exception as exc:
            header(f"Test Group {num} — {name}")
            fail(f"Unexpected exception: {exc}")
            import traceback; traceback.print_exc()
            results.failed += 1

    results.summary()


def main():
    args = parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()