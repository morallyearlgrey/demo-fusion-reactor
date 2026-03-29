# hi
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events.event_actions import EventActions

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

### DAC ceiling — do not sweep past 4095
The DAC hardware maximum is 4095. If electrode_a is already at or near 4095 and
the beam has not been found, STOP sweeping up. The beam does not exist at this
voltage, or the hardware conditions are not suitable. Switch to balance_electrodes
or do_nothing and wait. Continuing to request increases beyond 4095 wastes cycles.

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
    IMPORTANT: if is_low=False and roc is consistently negative over multiple
    cycles while you are still increasing electrodes, the beam has PASSED its
    transmission peak. You must switch to decrease_both immediately.
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

## MANDATORY PRE-CHECK — do this BEFORE applying any rule

Read the GROUND TRUTH block above. Write down the exact value of `is_low`.
If `is_low = False`, the beam IS present. You MUST NOT invoke Rule 5 (beam search).
Rule 5 only fires when `is_low = True` — this is a binary hardware fact, not an inference.

If you find yourself about to cite "beam search", "startup sweep", or "is_low" as
justification for increasing electrodes, STOP and verify the GROUND TRUTH block again.
If GROUND TRUTH says `is_low = False`, choose a different rule (6–14) instead.

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

3. DAC CEILING WITH NO BEAM — electrode_a >= 4000 AND is_low is True
   → You have swept to the top of the DAC range without finding the beam.
   → Do NOT continue requesting increase_both — it will be clamped to 4095 anyway.
   → action: "balance_electrodes" to reset asymmetry and try a different field shape.
   → If already balanced, action: "do_nothing" — beam cannot be found at this voltage.

4. SETTLING — is_settling is True, no safety flags, command_effectiveness not "reversed"
   → action: "do_nothing"
   → Boost converter and beam need time to stabilise after DAC change.
   → Exception: is_low is True → go to rule 5 instead (beam search does not wait).

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

# reads in the analysis dict and produces a json, reasons like crazy lol
class DecisionAgent(LlmAgent):

    # model may need to be edited, ask dawn lol
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        super().__init__(
            name="decision_agent",
            description="LLM that decides electrode adjustments based on sensor analysis.",
            model=model,
            instruction=DECISION_INSTRUCTION,
            # include_contents='none' stops the LLM from seeing the full ADK
            # conversation history. Without this, it reads prior cycle events
            # (e.g. old AnalyzeAgent text saying is_low=True) and hallucinates
            # stale state. With 'none' it only sees the current turn — our
            # freshly-built prompt with the actual current analysis data.
            include_contents="none",
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        analysis    = ctx.session.state.get("analysis", {})
        last_action = ctx.session.state.get("last_action", {})

        # focused analysis from the analysis agent values, outputs into json to context baka blast
        focused_analysis = {
            # error metrics
            "steady_state_error":    analysis.get("steady_state_error"),
            "error_from_target":     analysis.get("error_from_target"),
            "integral_error":        analysis.get("integral_error"),

            # signal shape
            "raw_adc":               analysis.get("raw_adc"),
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

        # from action agent
        focused_last_action = {
            "action":      last_action.get("action"),
            "ok":          last_action.get("ok"),
            "prev_a":      last_action.get("prev_a"),
            "prev_b":      last_action.get("prev_b"),
            "electrode_a": last_action.get("electrode_a"),
            "electrode_b": last_action.get("electrode_b"),
        } if last_action else {}

        # Build recent_decisions from history — last 5 entries with ADC + action context.
        # This stops the LLM from reasoning as if it's cycle 1 when it's cycle 20+.
        history = ctx.session.state.get("history", [])
        decision_history = ctx.session.state.get("decision_history", [])
        recent_decisions = []
        tail = history[-5:] if len(history) >= 5 else history
        for i, entry in enumerate(tail):
            # pair each history entry with the corresponding decision if available
            dh_offset = len(decision_history) - len(tail) + i
            past_decision = decision_history[dh_offset] if 0 <= dh_offset < len(decision_history) else {}
            recent_decisions.append({
                "raw_adc":     entry.get("raw_adc"),
                "electrode_a": entry.get("electrode_a"),
                "electrode_b": entry.get("electrode_b"),
                "target_adc":  entry.get("target_adc"),
                "action":      past_decision.get("action", "unknown"),
                "delta_a":     past_decision.get("electrode_a_delta", 0),
                "delta_b":     past_decision.get("electrode_b_delta", 0),
            })

        # Log what the LLM is actually about to see — use this to catch stale data
        logger.info(
            "DecisionAgent | feeding LLM: cycle=%s is_low=%s is_settling=%s sse=%s electrode_a=%s",
            focused_analysis.get("cycle_count"),
            focused_analysis.get("is_low"),
            focused_analysis.get("is_settling"),
            focused_analysis.get("steady_state_error"),
            focused_analysis.get("electrode_a"),
        )

        # ── GROUND TRUTH HEADER ───────────────────────────────────────────────
        # These values are HARDWARE-MEASURED FACTS computed by AnalyzeAgent.
        # They are not estimates. Do not contradict them in your reasoning.
        # The JSON below encodes the same values — use these plain-text lines
        # as the authoritative reference if there is any ambiguity.
        is_low_val       = focused_analysis.get("is_low")
        is_spike_val     = focused_analysis.get("is_spike")
        is_settling_val  = focused_analysis.get("is_settling")
        sse_val          = focused_analysis.get("steady_state_error")
        effectiveness_val = focused_analysis.get("command_effectiveness")
        electrode_a_val  = focused_analysis.get("electrode_a")

        ground_truth_header = (
            f"## GROUND TRUTH — hardware-measured this cycle (do not contradict)\n"
            f"  is_low        = {is_low_val}   ← beam ABSENT if True, PRESENT if False\n"
            f"  is_spike      = {is_spike_val}\n"
            f"  is_settling   = {is_settling_val}\n"
            f"  steady_state_error = {sse_val}  ← negative=below target, positive=above\n"
            f"  command_effectiveness = {effectiveness_val}\n"
            f"  electrode_a   = {electrode_a_val}  ← DAC max is 4095, do not sweep past it\n\n"
        )

        cycle_now = focused_analysis.get("cycle_count", 0)
        prompt = (
            ground_truth_header
            + f"## SYSTEM CONTEXT — you are at cycle {cycle_now}, NOT cycle 1\n"
            + f"This system has been running for {cycle_now} cycles. "
            + f"Do NOT reason as if no data exists. Use the history below.\n\n"
            + f"## Recent History (last 5 cycles — oldest to newest)\n"
            + f"```json\n{json.dumps(recent_decisions, indent=2)}\n```\n\n"
            + f"## Current Analysis\n"
            + f"```json\n{json.dumps(focused_analysis, indent=2)}\n```\n\n"
            + f"## Last Action Taken\n"
            + f"```json\n{json.dumps(focused_last_action, indent=2)}\n```\n\n"
            + f"Apply the decision rules in priority order and output your decision JSON.\n"
            + f"Your reasoning MUST be consistent with the GROUND TRUTH values above.\n"
            + f"IMPORTANT: Output ONLY the JSON object. Do not repeat, echo, or summarise "
            + f"any part of this prompt. Do not output 'GROUND TRUTH' or analysis fields. "
            + f"Your entire response must start with {{ and end with }}."
        )

        ctx.session.state["_decision_prompt"] = prompt

        raw_decision_text = ""
        async for event in super()._run_async_impl(ctx):
            if hasattr(event, "content") and event.content:
                parts = event.content.parts or []
                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        raw_decision_text += text
            yield event

        # parse decision JSON — robust extraction handles markdown fences and
        # extra text around the JSON object
        import re
        decision = None
        try:
            # First try: find the first {...} block in the response
            match = re.search(r'\{.*\}', raw_decision_text, re.DOTALL)
            if match:
                decision = json.loads(match.group(0))
            else:
                raise ValueError("No JSON object found in response")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("DecisionAgent: parse error: %s | raw: %s", exc, raw_decision_text[:300])
            decision = {
                "action":            "do_nothing",
                "electrode_a_delta": 0,
                "electrode_b_delta": 0,
                "reasoning":         f"JSON parse error — safe default. Raw output: {raw_decision_text[:200]}",
                "confidence":        0.0,
            }

        # ── POST-PARSE SAFETY OVERRIDES ───────────────────────────────────────
        # Hard rules that Python enforces regardless of what the LLM decided.
        # These catch the cases where the LLM prompt guidance wasn't enough.
        # ORDER MATTERS — higher overrides win. Spike is always checked first.

        max_delta = analysis.get("tuning_params", {}).get("max_delta", 200)

        # ── OVERRIDE 0: SPIKE — highest priority, non-negotiable safety. ──────
        # is_spike=True means raw_adc > spike_threshold (800), which may indicate
        # arc discharge in the vacuum tube. Switch to backup IMMEDIATELY.
        # This fires regardless of what the LLM decided — do not move past this
        # check if the spike flag is set.
        actual_is_spike = bool(analysis.get("is_spike", False))
        if actual_is_spike:
            logger.warning(
                "DecisionAgent | OVERRIDE (spike): is_spike=True (adc=%s > threshold=%s) "
                "— forcing emergency_backup, ignoring LLM action=%s",
                analysis.get("raw_adc"), analysis.get("tuning_params", {}).get("spike_threshold", 800),
                decision.get("action"),
            )
            decision["action"]            = "emergency_backup"
            decision["electrode_a_delta"] = 0
            decision["electrode_b_delta"] = 0
            decision["confidence"]        = 1.0
            decision["reasoning"] = (
                f"[OVERRIDE] is_spike=True: raw_adc={analysis.get('raw_adc')} exceeds "
                f"spike_threshold={analysis.get('tuning_params', {}).get('spike_threshold', 800)}. "
                "Arc discharge risk — switching to backup mode immediately. "
                "LLM action was overridden."
            )
            # Skip all further overrides — emergency action is final.
            ctx.session.state["decision"] = decision
            decision_history = list(ctx.session.state.get("decision_history", []))
            decision_history.append({
                "cycle":             focused_analysis.get("cycle_count"),
                "action":            decision.get("action"),
                "electrode_a_delta": 0,
                "electrode_b_delta": 0,
                "confidence":        1.0,
                "effectiveness":     analysis.get("command_effectiveness", "unknown"),
            })
            if len(decision_history) > 10:
                decision_history = decision_history[-10:]
            ctx.session.state["decision_history"] = decision_history
            logger.info(
                "DecisionAgent | action=%s delta_a=%+d delta_b=%+d confidence=%.2f | %s",
                decision.get("action"), 0, 0, 1.0, decision["reasoning"][:120],
            )
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={
                    "decision":         decision,
                    "decision_history": decision_history,
                }),
            )
            return

        # ── OVERRIDE 1: REVERSED EFFECTIVENESS — beam is past its peak. ───────
        # command_effectiveness="reversed" means ADC moved opposite to the DAC
        # change — the system is on the descending side of the beam transmission
        # hill. Any increase at this point makes things worse.
        #
        # TWO-TIER GUARD:
        #   Tier A: Single-cycle reversed + LLM chose increase → force decrease.
        #           (existing logic, kept for single-cycle response)
        #   Tier B: 2+ consecutive reversed cycles → force decrease regardless of
        #           what the LLM chose (even do_nothing). The LLM sometimes picks
        #           do_nothing on the reversed cycle then tries to increase next
        #           cycle. A streak of 2 means we are definitively past the peak.

        effectiveness_now = analysis.get("command_effectiveness", "unknown")

        # Count how many of the last N decision_history entries had reversed effectiveness
        prev_history = ctx.session.state.get("decision_history", [])
        reversed_streak = 0
        for entry in reversed(prev_history):
            if entry.get("effectiveness") == "reversed":
                reversed_streak += 1
            else:
                break  # streak broken — stop counting

        increase_actions = ("increase_both", "increase_a", "increase_b", "small_adjust_up")
        safe_delta = max(10, max_delta // 4)

        # Tier B: streak of 2+ reversed cycles — force decrease unconditionally
        if reversed_streak >= 2:
            logger.warning(
                "DecisionAgent | OVERRIDE (reversed streak=%d): %d consecutive reversed "
                "cycles — forcing decrease_both delta=%d regardless of LLM action=%s",
                reversed_streak, reversed_streak, safe_delta, decision.get("action"),
            )
            decision["action"]            = "decrease_both"
            decision["electrode_a_delta"] = -safe_delta
            decision["electrode_b_delta"] = -safe_delta
            decision["reasoning"] = (
                f"[OVERRIDE] {reversed_streak} consecutive reversed-effectiveness cycles — "
                f"beam is past transmission peak. Forcing decrease_both delta={safe_delta} "
                f"to walk back toward the hill. LLM action was '{decision.get('action')}'. "
                + decision.get("reasoning", "")
            )
        # Tier A: single reversed cycle AND LLM chose increase → force decrease
        elif (effectiveness_now == "reversed"
                and decision.get("action") in increase_actions):
            logger.warning(
                "DecisionAgent | OVERRIDE: LLM chose increase despite reversed effectiveness "
                "— forcing decrease_both delta=%d",
                safe_delta,
            )
            decision["action"]            = "decrease_both"
            decision["electrode_a_delta"] = -safe_delta
            decision["electrode_b_delta"] = -safe_delta
            decision["reasoning"] = (
                f"[OVERRIDE] command_effectiveness=reversed but LLM chose increase. "
                f"Forcing decrease_both delta={safe_delta}. "
                + decision.get("reasoning", "")
            )

        # 0. IS_LOW HALLUCINATION GUARD — most common LLM failure mode.
        #    The model often acts as if is_low=True even when hardware says False.
        #    Previous fix used keyword-sniffing on reasoning text — unreliable because
        #    the LLM sometimes omits beam-search keywords but still fires Rule 5 logic.
        #
        #    NEW APPROACH: gate purely on hardware truth, not on what the LLM wrote.
        #    If is_low=False (beam is present), an unsolicited increase is wrong when:
        #      (a) system is still settling — wait for it to stabilise, OR
        #      (b) |sse| < 15 — beam is within the target band, hold position.
        #    If sse < -15 (real undershoot), the increase may be legitimately correct
        #    under Rules 8/10 — allow it but strip any hallucinated beam-search reasoning.
        actual_is_low   = bool(analysis.get("is_low", False))
        actual_settling = bool(analysis.get("is_settling", False))
        sse_now         = float(analysis.get("steady_state_error", 0))
        action_now      = decision.get("action", "do_nothing")
        reasoning_now   = decision.get("reasoning", "")

        increase_actions = ("increase_both", "increase_a", "increase_b", "small_adjust_up")

        if not actual_is_low and action_now in increase_actions:
            if actual_settling and abs(sse_now) < 30:
                # Beam present, still settling — never stack commands mid-transient.
                logger.warning(
                    "DecisionAgent | OVERRIDE (spurious increase): is_low=False and "
                    "is_settling=True — forcing do_nothing"
                )
                decision["action"]            = "do_nothing"
                decision["electrode_a_delta"] = 0
                decision["electrode_b_delta"] = 0
                decision["reasoning"] = (
                    "[OVERRIDE] Beam present (is_low=False) and system is still settling. "
                    "Holding position until transient clears. " + reasoning_now
                )
            elif abs(sse_now) < 15:
                # Beam present and within target band — hold, no increase needed.
                logger.warning(
                    "DecisionAgent | OVERRIDE (spurious increase): is_low=False and "
                    "|sse|=%.1f < 15 — beam on target, forcing do_nothing",
                    abs(sse_now),
                )
                decision["action"]            = "do_nothing"
                decision["electrode_a_delta"] = 0
                decision["electrode_b_delta"] = 0
                decision["reasoning"] = (
                    f"[OVERRIDE] Beam present (is_low=False) and |sse|={sse_now:.1f} is within "
                    "±15 target band. No increase warranted. " + reasoning_now
                )
            else:
                # sse < -15: real undershoot with beam present — increase is correct
                # under Rules 8/10. Allow it, but clean up any beam-search language
                # in the reasoning so it doesn't mislead downstream agents.
                if "beam search" in reasoning_now.lower() or (
                        "is_low" in reasoning_now.lower() and "true" in reasoning_now.lower()):
                    logger.warning(
                        "DecisionAgent | WARN (hallucinated beam-search language): is_low=False "
                        "but LLM cited beam-search; sse=%.1f is real undershoot — allowing "
                        "increase but correcting reasoning",
                        sse_now,
                    )
                    decision["reasoning"] = (
                        "[CORRECTED] Beam IS present (is_low=False). Increase justified by "
                        f"real undershoot steady_state_error={sse_now:.1f}, not beam search. "
                        + reasoning_now
                    )

        # 1. Clamp deltas to max_delta
        decision["electrode_a_delta"] = max(
            -max_delta, min(max_delta, int(decision.get("electrode_a_delta", 0)))
        )
        decision["electrode_b_delta"] = max(
            -max_delta, min(max_delta, int(decision.get("electrode_b_delta", 0)))
        )

        # 2. If LLM is trying to increase but electrode is already at DAC ceiling,
        #    override to do_nothing to stop the infinite sweep-at-4095 loop.
        electrode_a_now = analysis.get("electrode_a", 0)
        electrode_b_now = analysis.get("electrode_b", 0)
        action = decision.get("action", "do_nothing")
        if action in ("increase_both", "increase_a", "small_adjust_up"):
            if electrode_a_now >= 4090 and electrode_b_now >= 4090:
                logger.warning(
                    "DecisionAgent | OVERRIDE: LLM requested increase at DAC ceiling "
                    "(a=%d b=%d) — forcing do_nothing",
                    electrode_a_now, electrode_b_now,
                )
                decision["action"]            = "do_nothing"
                decision["electrode_a_delta"] = 0
                decision["electrode_b_delta"] = 0
                decision["reasoning"] = (
                    f"[OVERRIDE] Electrodes at DAC ceiling ({electrode_a_now}/{electrode_b_now}). "
                    "Increase blocked. Beam not found in full sweep range. "
                    + decision.get("reasoning", "")
                )

        ctx.session.state["decision"] = decision

        # Append this cycle's decision to decision_history so future cycles
        # can show the LLM what actually happened in recent cycles.
        decision_history = list(ctx.session.state.get("decision_history", []))
        decision_history.append({
            "cycle":               focused_analysis.get("cycle_count"),
            "action":              decision.get("action"),
            "electrode_a_delta":   decision.get("electrode_a_delta", 0),
            "electrode_b_delta":   decision.get("electrode_b_delta", 0),
            "confidence":          decision.get("confidence", 0),
            "effectiveness":       analysis.get("command_effectiveness", "unknown"),
        })
        # cap to last 10 entries — only need enough to cover the recent_decisions window
        if len(decision_history) > 10:
            decision_history = decision_history[-10:]
        ctx.session.state["decision_history"] = decision_history

        logger.info(
            "DecisionAgent | action=%s delta_a=%+d delta_b=%+d confidence=%.2f | %s",
            decision.get("action"),
            decision.get("electrode_a_delta", 0),
            decision.get("electrode_b_delta", 0),
            decision.get("confidence", 0),
            decision.get("reasoning", "")[:120],
        )

        # Emit state_delta so ADK flushes the decision back to the storage
        # session via append_event. Without this, ImprovementAgent reads an
        # empty decision dict every cycle.
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={
                "decision":          decision,
                "decision_history":  decision_history,
            }),
        )