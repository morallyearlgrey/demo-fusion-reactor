import logging
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from pipeline.tools.reactor_tools import {
    set_electrodes,
    set_electrode_a,
    set_electrode_b,
    set_backup,
    set_ai_mode,
}

logger = logging.getLogger(__name__)

# DAC constants
DAC_MIN = 0
DAC_MAX = 4095

## simple safety function, if any calculation yields outside of range value, clamp will just pull it back in
def _clamp(value: int) -> int:
    return max(DAC_MIN, min(DAC_MAX,value))

# class definition
class ActionAgent(BaseAgent):

    name: str = "action_agent"
    description: str = "Executes electrode adjustments based on the decision agent output."

    def __init__(self):
        super().__init__(name=self.name, description = self.description)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:

        # reads from most recent session state, and saved to local variables
        decision = ctx.session.state.get("decision", {})
        action = decision.get("action", "do_nothing")
        delta_a = int(decision.get("electrode_a_delta", 0))
        delta_b = int(decision.get("electrode_b_delta", 0))
        max_delta = int(ctx.session.state.get("max_delta", 200))
        electrode_a = int(ctx.session.state.get("electrode_a", 0))
        electrode_b = int(ctx.session.state.get("electrode_a", 0))

        # safety clamp as requested by llm
        delta_a = max(-max_delta, min(max_delta, delta_a))
        delta_b = max(-max_delta, min(max_delta, delta_b))

        # default result
        result: dict = {"action": action, "ok": False}

        # bunch of if else dealing with which action to take based off action result seen in session state
        if action == " emergency_backup":
            result = set_backup()
            result["action"] = "emergency_backup"
            logger.warning("ActionAgent | EMERGENCY BACKUP triggered")

        elif action == "do_nothing":
            result = {"action": "do_nothing", "ok": True, "msg": "No Change Applied"}

        elif action in ("increase_both", "decrease_both", "small_adjust_up", "small_adjust_down"):
            new_a = _clamp(electrode_a + delta_a)
            new_b = _clamp(electrode_b + delta_b)
            result = set_electrodes(new_a,new_b)
            result["action"] = action
            result["prev_a"] = electrode_a
            result["prev_b"] = electrode_b

            ctx.session.state["electrode_a"] = new_a
            ctx.session.state["electrode_b"] = new_b
        
        elif action == "increase_a":
            new_a = _clamp(electrode_a+ abs(delta_a))
            result = set_electrode_a(new_a)
            result["action"] = action
            result["prev_a"] = electrode_a
            ctx.session.state["electrode_a"] = new_a
        
        elif action == "decrease_a":
            new_a = _clamp(electrode_a - abs(delta_a))
            result = set_electrode_a(new_a)
            result["action"] = action
            result["prev_a"] = electrode_a
            ctx.session.state["electrode_a"] = new_a

        elif action == "increase_b":
            new_a = _clamp(electrode_b + abs(delta_b))
            result = set_electrode_b(new_b)
            result["action"] = action
            result["prev_b"] = electrode_b
            ctx.session.state["electrode_b"] = new_b
        
        elif action == "decrease_b":
            new_a = _clamp(electrode_b - abs(delta_b))
            result = set_electrode_b(new_b)
            result["action"] = action
            result["prev_b"] = electrode_b
            ctx.session.state["electrode_b"] = new_b
        
        elif action == "balance_electrodes":
            avg = _clamp((electrode_a + electrode_b) // 2)
            result = set_electrodes(avg, avg)
            result["action"] = "balance_electrodes"
            result["prev_a"] = electrode_a
            result["prev_b"] = electrode_b
            ctx.session.state["electrode_a"] = avg
            ctx.session.state["electrode_b"] = avg
        
        else:
            logger.warning("ActionAgent | Unknown action: %s — doing nothing", action)
            result = {"action": action, "ok": False, "msg": "Unknown action"}

        ctx.session.state["last_action"] = result

        logger.info(
            "ActionAgent | action=%s ok=%s a=%s→%s b=%s→%s",
            action, result.get("ok"),
            electrode_a, ctx.session.state.get("electrode_a"),
            electrode_b, ctx.session.state.get("electrode_b"),
            )

        yield Event(
            author=self.name,
            content={
                "parts": [{
                    "text": (
                        f"[ActionAgent] Executed '{action}' | "
                        f"electrode_a: {electrode_a} → {ctx.session.state.get('electrode_a', electrode_a)} | "
                        f"electrode_b: {electrode_b} → {ctx.session.state.get('electrode_b', electrode_b)} | "
                        f"ok={result.get('ok')}"
                    )
                }]
            },
        )

