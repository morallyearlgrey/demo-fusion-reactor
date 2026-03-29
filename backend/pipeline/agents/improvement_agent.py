from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events.event_actions import EventActions

logger = logging.getLogger(__name__)

TARGET_ADC = 752
MAX_DELTA = 200
SPIKE_THRESHOLD = 800 # below photo max of 820
LOW_THRESHOLD = 580 # above photo max of 550

SETTLE_TIMEOUT = 8 # how many cycles to wait before forcing decision
SETTLING_THRESHOLD = 5.0 # how much movement counts as still settling
#  should be set above your sensor noise floor (~3 counts after Arduino's 10-sample averaging).


IMPROVEMENT_INSTRUCTION = """
You are the Improvement Agent for an autonomous electron beam controller.
Your role is meta-learning: after observing the full analyze→decide→act cycle,
you decide whether to adjust the control parameters to make the system
more efficient and stable over time.

## Physical context
This is a low-voltage proof-of-concept electron beam system inspired by inertial
electrostatic confinement (IEC) devices. It is focused purely on beam control and
transmission efficiency — not fusion itself. The goal is to demonstrate how
intelligent feedback can steer charged particles more efficiently through an
electrostatic structure.

### Hardware layout
Three components are aligned along a central axis inside a vacuum tube:

  Cathode (copper nail, negative)
    → electrons emitted from surface
      → accelerated toward anode by high voltage field
        → fraction pass THROUGH the ring aperture
          → travel downstream to Faraday cup
            → tiny current (pA-nA) collected
              → transimpedance amplifier (op-amp + feedback resistor)
                converts current to measurable voltage
                  → Arduino ADC reads voltage (0-1023)

  Anode (small copper ring, positive)
    → acts as electrostatic nozzle / aperture
    → electrons that hit the ring are lost
    → only electrons passing through the aperture reach the Faraday cup

  Faraday cup (downstream collector)
    → larger opening than ring to capture diverging beam
    → separation from ring: ~2cm
    → current collected here is the signal we are maximising

### Power and control chain
  12V DC source
    → high-voltage boost converter (200-500V adjustable output)
      → positive terminal → ring anode
      → negative terminal → cathode (system ground)

  Two DAC electrodes (A and B, 0-4095 each) control the boost converter input.
  Higher DAC = higher anode voltage = stronger accelerating field.
  The DAC range 0-4095 maps approximately to 200-500V at the anode.

### What higher ADC means
Higher ADC = more electrons passing through the ring aperture = better beam
transmission efficiency. This is the quantity we are trying to maximise.
The system is NOT trying to hit a fixed ADC setpoint — target_adc is a
moving optimisation target that should be pushed upward as the system
demonstrates it can sustain higher transmission rates stably.

### Why beam transmission is non-monotonic
There is an optimal anode voltage where electron beam transmission through the
ring aperture is maximised:
  - Too low voltage: electrons lack sufficient energy to form a focused beam
    that passes cleanly through the small ring aperture
  - Optimal voltage: beam geometry and energy level produce maximum throughput
  - Too high voltage: beam over-focuses or diverges, electrons miss the
    Faraday cup opening (~2cm away), ADC drops despite higher field strength

The ADC response to DAC is therefore a hill — the system must find and hold
the peak, not simply maximise voltage.

### Vacuum and pressure conditions
The system only works under appropriate vacuum pressure. At atmospheric pressure
electrons collide with gas molecules and scatter. The pressure conditions are
set externally and are not controlled by the DAC — they are assumed stable
during a run. If ADC is near zero despite correct voltage range, pressure
conditions may be the cause rather than voltage settings.

### Noise characteristics
The Faraday cup current is extremely small (pA-nA). The transimpedance amplifier
introduces noise. The Arduino performs 10-sample averaging to reduce but not
eliminate noise. Readings have a noise floor of approximately ±3-5 ADC counts
at rest. The cv metric captures this — cv < 0.02 means the beam is genuinely
stable, not just quiet.

## Parameters you can modify

- target_adc (int, 580-820):
    The current optimisation target — the ADC value the system is trying to reach.
    This is NOT a fixed physical setpoint. It is a moving goalpost.
    Push upward (by 5-15 counts) when:
      cv < 0.02 AND |steady_state_error| < 15 for several cycles
      (beam is stable at current target — ready to push higher)
    Lower (by 10-20 counts) when:
      integral_error < -800 consistently over many cycles
      (target is above the physical beam peak — unreachable)
    Lower when:
      command_effectiveness is "reversed" repeatedly
      (beam has passed its transmission peak, higher voltage makes it worse)
    Never set above 820 (hardware ADC saturation / spike risk).
    Never set below 580 (below this is beam-absent baseline noise).

- max_delta (int, 10-400):
    Maximum DAC step change per cycle. Controls how aggressively the system
    corrects toward target.
    Increase (by 10-20%) when:
      integral_error is large and steady_state_error is not closing
      (corrections too small to overcome system inertia)
    Decrease (by 10-20%) when:
      cv > 0.10 AND |steady_state_error| < 30
      (system is oscillating near target — steps too large, causing overshoot)
    Never below 10 (would make progress impossibly slow).
    Never above 400 (would cause large voltage swings and instability).

- spike_threshold (int):
    ADC level above which a reading is treated as an arc discharge or electrical
    noise spike, triggering emergency_backup immediately.
    Only lower if: genuine arc discharge spikes are being missed.
    Only raise if: false alarms are triggering emergency_backup too frequently.
    Never set below low_threshold + 50 (must stay well above beam floor).
    Default and recommended value: 800.

- low_threshold (int):
    ADC level below which the beam is considered absent or lost, triggering
    beam search mode in the decision agent.
    At startup, ADC below this is EXPECTED — it means beam has not been found yet.
    Only raise if: false beam-loss alarms are too frequent during normal operation.
    Only lower if: genuine beam-loss events are being missed.
    Never set above spike_threshold - 50.
    Default and recommended value: 580.

## You will receive
1. focused_analysis: key metrics from AnalyzeAgent this cycle
2. decision: what DecisionAgent chose this cycle
3. recent_history: last 10 history entries, each with raw_adc and the
   target_adc that was active at that moment
4. current_params: live values of all four parameters currently in session state

## Decision guidelines — apply carefully, changes compound over cycles

PUSH TARGET HIGHER when all of these are true:
  cv < 0.02 (beam very stable)
  |steady_state_error| < 15 (beam at current target)
  integral_error near zero (no chronic bias)
  is_settling is False (not mid-transient)
  history_length >= 15 (enough data to be confident)
  → push target_adc up by 5-15 counts

LOWER TARGET when any of these are true:
  integral_error < -800 across many cycles (target chronically unreachable)
  command_effectiveness is "reversed" in multiple recent history entries
  (beam is past its physical transmission peak)
  → lower target_adc by 10-20 counts

INCREASE MAX_DELTA when:
  |steady_state_error| > 60 AND integral_error large negative
  AND max_delta is already being used fully each cycle
  → increase by 10-20%

DECREASE MAX_DELTA when:
  cv > 0.10 AND |steady_state_error| < 30
  → decrease by 10-20%

DO NOTHING when:
  system is settling (is_settling is True)
  insufficient history (history_length < 10)
  system is already stable and near target
  any uncertainty — bad changes compound, stability is more valuable than speed

## Output format — single valid JSON object, no other text, no markdown fences
{
  "target_adc":      <int or null to leave unchanged>,
  "max_delta":       <int or null>,
  "spike_threshold": <int or null>,
  "low_threshold":   <int or null>,
  "reasoning":       "<2-3 sentences: what evidence you saw, what you changed and why, or why you changed nothing>",
  "changes_made":    <true|false>
}
"""


class ImprovementAgent(LlmAgent):

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        super().__init__(
            name="improvement_agent",
            description="LLM that meta-learns and updates tuning parameters in session state.",
            model=model,
            instruction=IMPROVEMENT_INSTRUCTION,
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:

        analysis  = ctx.session.state.get("analysis", {})
        decision  = ctx.session.state.get("decision", {})
        history   = ctx.session.state.get("history", [])
        recent_10 = history[-10:] if len(history) >= 10 else history

        current_params = {
            "target_adc":      ctx.session.state.get("target_adc",      TARGET_ADC),
            "max_delta":       ctx.session.state.get("max_delta",        MAX_DELTA),
            "spike_threshold": ctx.session.state.get("spike_threshold",  SPIKE_THRESHOLD),
            "low_threshold":   ctx.session.state.get("low_threshold",    LOW_THRESHOLD),
        }

        # focused subset — only what the LLM needs to reason about
        focused_analysis = {
            "steady_state_error":    analysis.get("steady_state_error"),
            "integral_error":        analysis.get("integral_error"),
            "cv":                    analysis.get("cv"),
            "is_settling":           analysis.get("is_settling"),
            "command_effectiveness": analysis.get("command_effectiveness"),
            "is_spike":              analysis.get("is_spike"),
            "is_low":                analysis.get("is_low"),
            "target_adc":            analysis.get("target_adc"),
            "cycle_count":           analysis.get("cycle_count"),
            "history_length":        analysis.get("history_length"),
        }

        prompt = (
            f"## Analysis\n```json\n{json.dumps(focused_analysis, indent=2)}\n```\n\n"
            f"## Decision This Cycle\n```json\n{json.dumps(decision, indent=2)}\n```\n\n"
            f"## Recent History (last 10)\n```json\n{json.dumps(recent_10, indent=2)}\n```\n\n"
            f"## Current Parameters\n```json\n{json.dumps(current_params, indent=2)}\n```\n\n"
            "Output your parameter update JSON now."
        )

        ctx.session.state["_improvement_prompt"] = prompt

        raw_text = ""
        async for event in super()._run_async_impl(ctx):
            if hasattr(event, "content") and event.content:
                parts = event.content.parts or []
                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        raw_text += text
            yield event

        # parse LLM response
        try:
            clean        = raw_text.strip().strip("```json").strip("```").strip()
            improvements = json.loads(clean)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("ImprovementAgent: parse error: %s", exc)
            improvements = {
                "target_adc":      None,
                "max_delta":       None,
                "spike_threshold": None,
                "low_threshold":   None,
                "reasoning":       f"Parse error: {exc}",
                "changes_made":    False,
            }

        # write updates directly to session state
        # session state is the sole source of truth for all tuning params
        # AnalyzeAgent reads these at the top of every cycle automatically
        # Arduino backup mode is intentionally left stagnant — not synced
        params_changed: dict = {}
        for key in ("target_adc", "max_delta", "spike_threshold", "low_threshold"):
            val = improvements.get(key)
            if val is not None:
                try:
                    val = int(val)
                    ctx.session.state[key] = val
                    params_changed[key] = val
                except (TypeError, ValueError):
                    pass

        ctx.session.state["improvement_result"] = {
            "improvements":  improvements,
            "params_changed": params_changed,   # what actually changed in state
        }

        logger.info(
            "ImprovementAgent | changes_made=%s params_changed=%s | %s",
            improvements.get("changes_made"),
            params_changed,
            improvements.get("reasoning", "N/A")[:120],
        )

        # Emit state_delta so ADK flushes these writes back to storage
        # via append_event. Any key ImprovementAgent may have updated goes here.
        improvement_state_delta: dict = {
            "improvement_result": {
                "improvements":  improvements,
                "params_changed": params_changed,
            },
        }
        # Only include tuning params that actually changed
        for key in ("target_adc", "max_delta", "spike_threshold", "low_threshold"):
            if key in params_changed:
                improvement_state_delta[key] = params_changed[key]

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=improvement_state_delta),
            content={
                "parts": [{
                    "text": (
                        f"[ImprovementAgent] changes_made={improvements.get('changes_made')} | "
                        f"params_changed={params_changed} | "
                        f"reasoning: {improvements.get('reasoning', 'N/A')}"
                    )
                }]
            },
        )