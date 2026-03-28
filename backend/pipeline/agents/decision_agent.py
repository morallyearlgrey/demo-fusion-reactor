# hi
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

logger = logging.getLogger(__name__)

DECISION_INSTRUCTION = """
You are the Decision Agent for an autonomous electron beam controller.
Your role is to adjust DAC electrode voltages to maximise electron beam transmission
through an electrostatic ring aperture into a Faraday cup.

## Physical system context
This is a low-voltage proof-of-concept IEC-inspired electron beam device.

Hardware chain:
  12V battery
    → boost converter (generates 200-500V HV output)
      → ring anode (positive, acts as electrostatic nozzle / aperture)
      → copper nail cathode (negative, electron source)
        → electrons accelerate toward anode
          → fraction pass THROUGH the ring aperture
            → travel to Faraday cup downstream
              → tiny current (pA-nA) flows
                → transimpedance amplifier converts current to voltage
                  → Arduino ADC reads voltage (0-1023)

Your DAC (0-4095) controls the 12V input to the boost converter.
DAC value maps approximately linearly to HV output: higher DAC = higher anode voltage.
The boost converter output range is roughly 200-500V across the DAC 0-4095 range.

## Critical physical behaviour you must understand

### Beam transmission is non-monotonic
Increasing DAC does NOT always increase ADC signal. There is an optimal focusing
voltage where beam transmission through the ring is maximised. Below it, electrons
lack energy or focus to pass through cleanly. Above it, the beam over-focuses or
diverges past the Faraday cup opening. The relationship looks like a hill — you are
searching for the peak.

### Near-zero ADC is the normal starting state
At startup or after large voltage changes, ADC near zero is expected — the beam
has not been found yet. This is NOT an emergency. is_low=True at startup means
the system needs to sweep upward to find the beam, not that something is broken.

### Asymmetric electrodes steer the beam
Electrode A and B create the field shape together. Equal values = symmetric field =
centred beam. Unequal values = asymmetric field = steered beam. Asymmetry is a
legitimate tool for finding the beam if symmetric tuning fails. Do not treat
electrode_diff as purely an error condition — it may be intentional.

### Small current, noisy signal
The Faraday cup current is tiny. The transimpedance amplifier introduces noise.
The Arduino's 10-sample averaging reduces but does not eliminate noise.
Always prefer steady_state_error over error_from_target for decisions.
Never chase a single-reading spike.

### Voltage changes have physical lag
After a DAC command, the boost converter, electric field, and beam all need time
to stabilise. This is why is_settling exists. Always respect it.

## Analysis fields — the ONLY metrics you reason about

### Error metrics
- steady_state_error (float): ewma_adc - target_adc
    PRIMARY signal. Negative = below target. Positive = above.
    Act on this. Do not act on error_from_target magnitude.
- error_from_target (int): raw_adc - target_adc
    Noisy instantaneous error. Use only to confirm direction.
- integral_error (float): accumulated signed error over last 20 readings
    Each entry uses the target that was active at that moment.
    Large negative = chronic undershoot = beam consistently below target.
    Large positive = chronic overshoot = beam consistently above target.
    Near zero = oscillating around target OR target is at the beam peak already.

### Signal shape
- ewma_adc (float): noise-filtered beam intensity estimate
- cv (float): coefficient of variation
    cv < 0.02 = very stable beam
    cv 0.02-0.05 = normal beam variance
    cv > 0.10 = beam is unstable or oscillating
- roc (int): rate of change, raw_adc minus previous raw_adc
- std_adc (float): standard deviation over last 20 readings

### Control state
- is_settling (bool): True if system is still responding to last DAC command
    Respect this — do not stack commands during transients.
- cycles_since_command (int): cycles since last electrode command was sent
- command_effectiveness (str):
    "effective"   — beam moved in expected direction after command
    "ineffective" — beam did not respond to command
                    The beam may not exist yet (still sweeping), or delta too small.
                    Consider larger delta or switch to asymmetric tuning.
    "reversed"    — beam moved opposite to expectation
                    Physical anomaly — beam may have passed peak and is on the
                    descending side of the transmission curve. Reduce voltage.
                    Do NOT continue increasing. Cut delta to 25%.
    "unknown"     — no command last cycle or insufficient history

### Safety flags
- is_spike (bool): raw_adc above spike_threshold
    A genuine spike above 800 may indicate electrical noise or arc discharge.
    Switch to backup immediately — do not attempt to correct with DAC.
- is_low (bool): raw_adc below low_threshold (580)
    At startup this is EXPECTED. Begin sweeping upward.
    After a period of normal operation this indicates beam loss — investigate.

### Context
- target_adc (int): current optimisation target set by improvement agent
    This is NOT a fixed physical setpoint — the improvement agent pushes it
    upward as the system demonstrates it can reach higher beam intensities.
- target_changed (bool): improvement agent just updated target_adc this cycle
- electrode_a (int): current DAC value electrode A (0-4095)
- electrode_b (int): current DAC value electrode B (0-4095)
- electrode_diff (int): electrode_a - electrode_b
- cycle_count (int): total cycles elapsed
- tuning_params.max_delta (int): max DAC step per cycle

## Decision rules — apply in this exact priority order

1. EMERGENCY — is_spike is True
   → action: "emergency_backup"
   → delta_a: 0, delta_b: 0
   → Spike above 800 ADC counts may indicate arc discharge in the vacuum tube.
     Switch to auto mode immediately, do not attempt DAC correction.

2. REVERSED RESPONSE — command_effectiveness is "reversed"
   → The beam responded opposite to expectation.
     This almost certainly means you passed the peak of the beam transmission
     curve and are on the descending side — higher voltage is now making things worse.
   → action: "decrease_both" (reverse direction regardless of steady_state_error sign)
   → delta: max_delta / 4 — small cautious step back
   → Do NOT follow steady_state_error direction if it contradicts this.

3. SETTLING — is_settling is True, no safety flags, command_effectiveness not "reversed"
   → action: "do_nothing"
   → Boost converter and beam need time to stabilise after DAC change.
   → Exception: is_low is True → go to rule 5 instead.

4. TARGET JUST CHANGED — target_changed is True, is_settling is False
   → Improvement agent raised the target. Large steady_state_error is expected.
   → action: "increase_both" with delta = max_delta / 4
   → Small conservative step toward new target. Do not jump.

5. BEAM SEARCH — is_low is True (ADC below low_threshold)
   → Beam has not been found or has been lost.
   → If electrode_a == electrode_b (symmetric): action "increase_both"
     Sweep upward symmetrically to find the beam transmission window.
   → If already at high DAC (electrode_a > 3000) with no signal:
     action "balance_electrodes" — symmetric field may not be optimal,
     try resetting asymmetry before giving up on this voltage range.
   → delta: max_delta / 2 — moderate sweep step, not too aggressive.

6. ON TARGET AND STABLE — |steady_state_error| < 15 AND cv < 0.05
   → action: "do_nothing"
   → Beam is at target intensity and stable. No adjustment needed.
   → The improvement agent will handle pushing the target higher when ready.

7. INEFFECTIVE COMMAND — command_effectiveness is "ineffective"
   AND is_low is False (beam exists but not responding)
   → Last DAC delta was too small to overcome system inertia or noise.
   → action: same direction as steady_state_error sign
   → delta: min(max_delta, abs(steady_state_error) * 3)
   → If still ineffective after 3+ cycles (cycles_since_command > 3 and near same ADC):
     try "balance_electrodes" — symmetric field may not be the optimal configuration.

8. MODERATE UNDERSHOOT — steady_state_error between -15 and -60
   AND is_settling is False AND command_effectiveness != "reversed"
   → Beam is below target but not far off.
   → action: "increase_both" or "small_adjust_up"
   → delta: min(max_delta, int(abs(steady_state_error) * 2))
   → Proportional — small error = small step.

9. MODERATE OVERSHOOT — steady_state_error between +15 and +60
   AND is_settling is False
   → Beam is above target — reduce voltage slightly.
   → action: "decrease_both" or "small_adjust_down"
   → delta: min(max_delta, int(abs(steady_state_error) * 2))

10. LARGE UNDERSHOOT — steady_state_error < -60
    AND is_settling is False AND command_effectiveness != "reversed"
    → Large gap below target. Apply full correction.
    → action: "increase_both"
    → delta: max_delta
    → If integral_error also large negative: chronic undershoot,
      full delta is appropriate — small steps are not closing the gap.

11. LARGE OVERSHOOT — steady_state_error > +60 AND is_settling is False
    → action: "decrease_both"
    → delta: max_delta

12. OSCILLATING — cv > 0.10 AND |steady_state_error| < 40
    → Beam is near target but unstable — do not chase noise.
    → action: "do_nothing"
    → If electrode_diff > 300: action "balance_electrodes"
      Asymmetry may be causing beam instability.

13. CHRONIC UNDERSHOOT BIAS — integral_error < -800 AND is_settling is False
    → System has been persistently below target for 20+ cycles.
    → Small corrections are not working. Either the target is too high,
      or the beam transmission peak is below the target voltage.
    → action: "increase_both" with delta = max_delta
    → Note in reasoning that improvement agent should consider lowering target_adc
      if this persists, as the physical optimum may be below current target.

14. CHRONIC OVERSHOOT BIAS — integral_error > 800 AND is_settling is False
    → action: "decrease_both" with delta = max_delta

15. ELECTRODE ASYMMETRY — |electrode_diff| > 400 AND |steady_state_error| > 20
    AND beam was previously stable (cv was low in recent cycles)
    → Asymmetry has grown large and beam is off target.
    → action: "balance_electrodes"
    → Reset to symmetric field before attempting further corrections.

## Output format
Single valid JSON object. No markdown fences, no text outside the JSON.

{
  "action": "<one of: increase_both | decrease_both | increase_a | decrease_a | increase_b | decrease_b | small_adjust_up | small_adjust_down | balance_electrodes | do_nothing | emergency_backup>",
  "electrode_a_delta": <integer, positive=increase, negative=decrease, 0=no change. Never exceed max_delta in absolute value.>,
  "electrode_b_delta": <integer, positive=increase, negative=decrease, 0=no change. Never exceed max_delta in absolute value.>,
  "reasoning": "<2-3 sentences: which rule fired, what the key metric values were, why this action over alternatives.>",
  "confidence": <float 0.0-1.0>
}
"""

# reads in the analysis dic and produces a json, reasons like crazy lol
class DecisionAgent(LlmAgent):

    # model may need to be edited, ask dawn lol
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        super().__init__(
            name="decision_agent",
            description="LLM that decides electrode adjustments based on sensor analysis.",
            model=model,
            instruction=DECISION_INSTRUCTION,
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        analysis    = ctx.session.state.get("analysis", {}) # from analysis 
        last_action = ctx.session.state.get("last_action", {}) # from action agent

        # focused analysis from the analysis agent values, outputs into json to context baka blast
        focused_analysis = {
            # error metrics
            "steady_state_error":    analysis.get("steady_state_error"), # error between ewma and target
            "error_from_target":     analysis.get("error_from_target"), # error from raw and target
            "integral_error":        analysis.get("integral_error"), # sees how biased the system's been towards undershooting/overshooting

            # signal shape
            "ewma_adc":              analysis.get("ewma_adc"), # avg
            "cv":                    analysis.get("cv"), # stableness
            "roc":                   analysis.get("roc"), # roc between raw and prev
            "std_adc":               analysis.get("std_adc"), # standard dev across window

            # control state
            "is_settling":           analysis.get("is_settling"), #
            "cycles_since_command":  analysis.get("cycles_since_command"),
            "command_effectiveness": analysis.get("command_effectiveness"), # true if the signal is still moving over a certain threshold, but will false that if it's been 8 cycles in the clock

            # safety
            "is_spike":              analysis.get("is_spike"), # sees too much signal
            "is_low":                analysis.get("is_low"), # sees too lil' signal

            # context
            "target_adc":            analysis.get("target_adc"), # our adc val we wanna get to
            "target_changed":        analysis.get("target_changed"), # true/false if the target was changed by improvement bot
            "electrode_a":           analysis.get("electrode_a"), # electrode a reading 
            "electrode_b":           analysis.get("electrode_b"), # electrode b reading 
            "electrode_diff":        analysis.get("electrode_diff"), # diff between the two electrodes
            "cycle_count":           analysis.get("cycle_count"), # total cycles elapsed
            "tuning_params":         analysis.get("tuning_params", {}),
        }

        # from adction anaylsysis agent
        focused_last_action = {
            "action":      last_action.get("action"),
            "ok":          last_action.get("ok"),
            "prev_a":      last_action.get("prev_a"),
            "prev_b":      last_action.get("prev_b"),
            "electrode_a": last_action.get("electrode_a"),
            "electrode_b": last_action.get("electrode_b"),
        } if last_action else {}


        # dumps current analysis into a json
        prompt = (
            f"## Current Analysis\n"
            f"```json\n{json.dumps(focused_analysis, indent=2)}\n```\n\n"
            f"## Last Action Taken\n"
            f"```json\n{json.dumps(focused_last_action, indent=2)}\n```\n\n"
            f"Apply the decision rules in priority order and output your decision JSON."
        )

        ctx.session.state["_decision_prompt"] = prompt

        raw_decision_text = ""
        async for event in super()._run_async_impl(ctx):
            if hasattr(event, "content") and event.content:
                parts = event.content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        raw_decision_text += part["text"]
            yield event

        # pxarse decision JSON
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

        # safety clamp — LLM should never exceed max_delta but enforce it here too
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