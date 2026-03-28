"""
pipeline/agents/decision_agent.py

LLM-powered decision agent.
Reads the analysis from session state and decides what electrode adjustment
to make: increase, decrease, do nothing, emergency backup, etc.

Session state keys read:
  "analysis"    : dict  (from analyze_agent)
  "last_action" : dict  (from previous action_agent run)

Session state keys written:
  "decision"    : dict  {action, electrode_a_delta, electrode_b_delta, reasoning, confidence}
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

logger = logging.getLogger(__name__)

DECISION_INSTRUCTION = """
You are the Decision Agent for an autonomous nuclear fusion reactor controller.
Your role is to read sensor analysis and decide the optimal electrode voltage adjustment.

## Hardware context
- Two DAC electrodes (A and B) control an electric field that focuses charged particles
  toward a Faraday cup target inside a vacuum chamber.
- A photodiode (ADC 0-1023) measures particle beam intensity at the target.
- Higher ADC = more particles hitting the target = better fusion efficiency.
- DAC range: 0-4095 per electrode. Both default to same value; asymmetric tuning is allowed.
- Max voltage is 12V across the full DAC range (0-4095 = 0-12V).

## Analysis fields you will receive
These are the ONLY metrics you should reason about. Do not reference any others.

### Error metrics
- steady_state_error (float): ewma_adc - target_adc
    This is the PRIMARY error signal. Act on this, not error_from_target.
    Negative = below target. Positive = above target.
- error_from_target (int): raw_adc - target_adc
    Raw instantaneous error. Noisy — use only to confirm direction, not magnitude.
- integral_error (float): sum of signed errors over last 20 readings
    Each reading's error is vs the target that was active at that moment.
    Large negative = chronic undershoot over many cycles.
    Large positive = chronic overshoot over many cycles.
    Near zero = oscillating symmetrically around target.

### Signal shape
- ewma_adc (float): noise-filtered estimate of current operating point
- cv (float): coefficient of variation — spread relative to mean
    cv < 0.02 = very stable
    cv 0.02–0.05 = normal operating variance
    cv > 0.10 = noisy or oscillating — system not under control
- roc (int): rate of change — raw_adc minus previous raw_adc
- std_adc (float): standard deviation over last 20 readings

### Control state
- is_settling (bool): True if system is still responding to the last command
    If True, the hardware is mid-transient. Prefer do_nothing unless is_spike or is_low.
- cycles_since_command (int): how many cycles since last electrode command
- command_effectiveness (str): did the last command work?
    "effective"   — ADC moved in expected direction, continue approach
    "ineffective" — command had no measurable effect, consider larger delta
    "reversed"    — ADC moved opposite to expectation (hardware anomaly)
                    Do NOT repeat same direction. Cut delta to 25%, prefer balance_electrodes.
    "unknown"     — no command last cycle or insufficient data

### Safety flags (always override is_settling)
- is_spike (bool): raw_adc above spike_threshold — dangerous, act immediately
- is_low (bool): raw_adc below low_threshold — beam too weak, act immediately

### Context
- target_adc (int): current desired setpoint
- target_changed (bool): True if target was just updated by improvement agent
    If True, large steady_state_error may be expected — do not over-correct immediately.
- electrode_a (int): current DAC value electrode A (0-4095)
- electrode_b (int): current DAC value electrode B (0-4095)
- electrode_diff (int): electrode_a - electrode_b (asymmetry indicator)
- tuning_params.max_delta (int): hard ceiling on electrode change per cycle

## Decision rules — apply in this exact priority order

1. EMERGENCY — is_spike is True
   → action: "emergency_backup"
   → delta: 0, 0
   → Reasoning must mention spike value and spike_threshold.

2. LOW SIGNAL — is_low is True
   → action: "increase_both" regardless of is_settling
   → delta: use max_delta / 2 as a safe starting increase
   → Beam is too weak to wait for transient to finish.

3. SETTLING — is_settling is True and no safety flag
   → action: "do_nothing"
   → Hardware is still responding to last command. Wait.
   → Exception: if integral_error magnitude > 500 AND cycles_since_command > 4,
     a very large chronic bias may justify a small corrective action even while settling.

4. TARGET JUST CHANGED — target_changed is True
   → Be conservative. A large steady_state_error here is expected, not alarming.
   → Prefer small_adjust_up or small_adjust_down over large corrections.
   → delta: max_delta / 4

5. COMMAND REVERSED — command_effectiveness is "reversed"
   → Do NOT increase if last command increased. Do NOT decrease if last command decreased.
   → action: "balance_electrodes" or opposite of last direction with delta = max_delta / 4
   → This is a hardware anomaly. Be cautious.

6. ON TARGET — |steady_state_error| < 15 AND cv < 0.05
   → action: "do_nothing"
   → System is at setpoint and stable. No adjustment needed.

7. MODERATE UNDERSHOOT — steady_state_error between -15 and -60
   → action: "increase_both" or "small_adjust_up"
   → delta: scale proportionally — small error = small delta
   → Suggested: delta = min(max_delta, abs(steady_state_error) * 2)

8. MODERATE OVERSHOOT — steady_state_error between +15 and +60
   → action: "decrease_both" or "small_adjust_down"
   → delta: same proportional scaling

9. LARGE UNDERSHOOT — steady_state_error < -60
   → action: "increase_both"
   → delta: max_delta (full correction)
   → If integral_error also large negative: system has chronic bias, use full delta.

10. LARGE OVERSHOOT — steady_state_error > +60
    → action: "decrease_both"
    → delta: max_delta

11. OSCILLATING — cv > 0.10 AND |steady_state_error| < 30
    → System is noisy but near target — do not chase noise
    → action: "do_nothing" or "balance_electrodes" if electrode_diff > 200

12. CHRONIC BIAS — |integral_error| > 800 AND is_settling is False
    → System has been persistently off-target for many cycles
    → action: increase or decrease based on sign of integral_error
    → Use full max_delta — small corrections are not working

13. INEFFECTIVE COMMAND — command_effectiveness is "ineffective"
    → Last delta was too small. Increase magnitude.
    → action: same direction as last, delta = min(max_delta, last_delta * 1.5)

14. ELECTRODE ASYMMETRY — |electrode_diff| > 300 AND error persists
    → action: "balance_electrodes"

## Output format
Your response MUST be a single valid JSON object with exactly these fields.
No markdown, no explanation outside the JSON, no extra keys.

{
  "action": "<one of: increase_both | decrease_both | increase_a | decrease_a | increase_b | decrease_b | small_adjust_up | small_adjust_down | balance_electrodes | do_nothing | emergency_backup>",
  "electrode_a_delta": <integer, positive=increase, negative=decrease, 0=no change. Never exceed max_delta in absolute value.>,
  "electrode_b_delta": <integer, positive=increase, negative=decrease, 0=no change. Never exceed max_delta in absolute value.>,
  "reasoning": "<2-3 sentences explaining which rule fired, what the key metrics were, and why this action.>",
  "confidence": <float 0.0-1.0>
}
"""


class DecisionAgent(LlmAgent):
    """
    LLM agent that reads the analysis dict from session state and produces
    a structured decision JSON. Uses Gemini to reason over control-theory
    metrics produced by AnalyzeAgent.
    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        super().__init__(
            name="decision_agent",
            description="LLM that decides electrode adjustments based on sensor analysis.",
            model=model,
            instruction=DECISION_INSTRUCTION,
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:

        analysis    = ctx.session.state.get("analysis", {})
        last_action = ctx.session.state.get("last_action", {})

        # Build a focused prompt — only include what the LLM needs to reason about.
        # We explicitly exclude history (too large) and internal staging keys.
        focused_analysis = {
            # error metrics
            "steady_state_error":    analysis.get("steady_state_error"),
            "error_from_target":     analysis.get("error_from_target"),
            "integral_error":        analysis.get("integral_error"),

            # signal shape
            "ewma_adc":              analysis.get("ewma_adc"),
            "cv":                    analysis.get("cv"),
            "roc":                   analysis.get("roc"),
            "std_adc":               analysis.get("std_adc"),

            # control state
            "is_settling":           analysis.get("is_settling"),
            "cycles_since_command":  analysis.get("cycles_since_command"),
            "command_effectiveness": analysis.get("command_effectiveness"),

            # safety
            "is_spike":              analysis.get("is_spike"),
            "is_low":                analysis.get("is_low"),

            # context
            "target_adc":            analysis.get("target_adc"),
            "target_changed":        analysis.get("target_changed"),
            "electrode_a":           analysis.get("electrode_a"),
            "electrode_b":           analysis.get("electrode_b"),
            "electrode_diff":        analysis.get("electrode_diff"),
            "cycle_count":           analysis.get("cycle_count"),
            "tuning_params":         analysis.get("tuning_params", {}),
        }

        # Summarise last_action — only include fields relevant to the decision
        focused_last_action = {
            "action":      last_action.get("action"),
            "ok":          last_action.get("ok"),
            "prev_a":      last_action.get("prev_a"),
            "prev_b":      last_action.get("prev_b"),
            "electrode_a": last_action.get("electrode_a"),
            "electrode_b": last_action.get("electrode_b"),
        } if last_action else {}

        prompt = (
            f"## Current Analysis\n"
            f"```json\n{json.dumps(focused_analysis, indent=2)}\n```\n\n"
            f"## Last Action Taken\n"
            f"```json\n{json.dumps(focused_last_action, indent=2)}\n```\n\n"
            f"Apply the decision rules in priority order and output your decision JSON."
        )

        ctx.session.state["_decision_prompt"] = prompt

        # Collect LLM output
        raw_decision_text = ""
        async for event in super()._run_async_impl(ctx):
            if hasattr(event, "content") and event.content:
                parts = event.content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        raw_decision_text += part["text"]
            yield event

        # Parse decision JSON
        try:
            clean    = raw_decision_text.strip().strip("```json").strip("```").strip()
            decision = json.loads(clean)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("DecisionAgent: parse error: %s | raw: %s", exc, raw_decision_text[:300])
            decision = {
                "action":            "do_nothing",
                "electrode_a_delta": 0,
                "electrode_b_delta": 0,
                "reasoning":         f"JSON parse error — safe default. Raw output: {raw_decision_text[:200]}",
                "confidence":        0.0,
            }

        # Safety clamp — LLM should never exceed max_delta but enforce it here too
        max_delta = analysis.get("tuning_params", {}).get("max_delta", 200)
        decision["electrode_a_delta"] = max(
            -max_delta, min(max_delta, int(decision.get("electrode_a_delta", 0)))
        )
        decision["electrode_b_delta"] = max(
            -max_delta, min(max_delta, int(decision.get("electrode_b_delta", 0)))
        )

        ctx.session.state["decision"] = decision

        logger.info(
            "DecisionAgent | action=%s delta_a=%+d delta_b=%+d confidence=%.2f | %s",
            decision.get("action"),
            decision.get("electrode_a_delta", 0),
            decision.get("electrode_b_delta", 0),
            decision.get("confidence", 0),
            decision.get("reasoning", "")[:120],
        )
```

---

## What changed and why

**Prompt metrics — complete replacement:**

The old prompt referenced these which no longer exist in the analysis dict:
```
trend_label, trend_slope, drift, stability_index, converging, pct_error
```

The new prompt references only what `AnalyzeAgent` actually produces:
```
steady_state_error, error_from_target, integral_error,
ewma_adc, cv, roc, std_adc,
is_settling, cycles_since_command, command_effectiveness,
is_spike, is_low, target_changed, electrode_diff, tuning_params