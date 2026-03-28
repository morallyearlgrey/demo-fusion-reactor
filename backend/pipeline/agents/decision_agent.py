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


"""

# reads in the analysis dic and produces a json, reasons like crazy lol
class DecisionAgent(LlmAgent):

    def __init__(self, model: str = "gemini-2.5-flash"):
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