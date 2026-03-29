import asyncio
import logging
import statistics
import time
from typing import AsyncGenerator
 
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
 
from pipeline.tools.reactor_tools import get_latest_reading

logger = logging.getLogger(__name__)
# state is curr snapshot of info needed for pipeline
# events is time ordered history of the convo thread, including agent responses and user input
# state is updated continuously, kinda temp memory
# events are accumulated, for the llms to record every little of its info
# we NEED history tho and cant use events because of the specific structure we require, the cap, and is easibly readable by other agents


# CONSTANTS
# tuning parameter const values
# 12 v max
TARGET_ADC = 752
MAX_DELTA = 200
SPIKE_THRESHOLD = 800 # below photo max of 820
LOW_THRESHOLD = 580 # above photo max of 550

SETTLE_TIMEOUT = 8 # how many cycles to wait before forcing decision
SETTLING_THRESHOLD = 5.0 # how much movement counts as still settling
#  should be set above your sensor noise floor (~3 counts after Arduino's 10-sample averaging).


NUM_READINGS = 4 # num of readings to look at to see if the adc is still settling

MAX_HISTORY = 50
CYCLE_DELAY = 1.0
STATS_WINDOW = 20

# session state:
# raw_adc (adc average iirc)
# electrode_a (dac value)
# electrode_b (dac value)
# flag (auto or ai)

# so instead of a flat rolling mean, we're gonna use a weighted one because the voltage readings should be very stagnant, not linear like orig thought, orig design also woudnt be able to tell if we're drifting away or responding to a correction and settling

# where is adc mean right now
def ewma(values: list, alpha: float = 0.3) -> float:
    """
    expoentially weighted moving average
    """
    if not values:
        return 0.0
    result = float(values[0])
    for v in values[1:]:
        result = alpha * v + (1 - alpha) * result
    return round(result, 2)

# how stable is this, whether system is ready to be optimized further
# for the improvement agent
# high cv means system is not under control, low cv means system is under control
# cv is spread or standard deviation of adc relative to its own mean across past stat window 
# cv < 0.02 = very stable
#  cv > 0.10 = noisy / unstable
def coefficient_of_variation(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return round(statistics.stdev(values) / mean, 4)

# diagnoses whether or not the system has been biased towards undershooting or overshooting
# considers target value that was active in each point of time and calculates the diff between that and the curr target
# sums and then a neg means the gap isnt being closed enough, or says if target adc is too high

# large neg = chronic undershoot, increase max delta or lower target adc
# large pos = chronic overshoot, decrease max delta or raise low threshold
# near 0 is around target
def integral_error(history: list, window: int = 20) -> float:
    recent = history[-window:]
    return round(sum(r["raw_adc"] - r["target_adc"] for r in recent), 2)

# looks at what happened at last cycle, looks at last action taken, raw_adc, prev_adc and sees if it was moved in expected direction
#   Returns:
#       "effective"   — ADC moved in expected direction
#       "ineffective" — ADC did not move despite command
#       "reversed"    — ADC moved opposite to expectation (hardware anomaly)
#       "unknown"     — no command was sent last cycle or no data
def command_effectiveness(
    last_action: dict,
    current_adc: int,
    prev_adc: int,
) -> str:
    if not last_action or last_action.get("action") in ("do_nothing", "emergency_backup", None):
        return "unknown"

    delta_a = last_action.get("electrode_a", 0) - last_action.get("prev_a", 0)
    delta_b = last_action.get("electrode_b", 0) - last_action.get("prev_b", 0)
    net = delta_a + delta_b

    if net == 0:
        return "unknown"

    adc_change = current_adc - prev_adc
    expected   = 1 if net > 0 else -1
    actual     = 1 if adc_change > 0 else (-1 if adc_change < 0 else 0)

    if actual == expected:  return "effective"
    elif actual == 0:       return "ineffective"
    else:                   return "reversed"


# sees if the adc is still moving
# however, in the system photodiode should have some noise so we're gonna have cycles and cycle count and timeout to continue going even if it thinks it's not settled 
# will say it is settling if signal is still moving over a certain threshold, set to SETTLING_THRESHOLD rn

# true when:
# command recently issued, or signal is still moving (tail range > threshold saying that it moves lol)

# Once settle_timeout cycles have elapsed since the last command we force
def is_settling(
    recent: list,
    cycles_since_command: int,
    threshold: float = SETTLING_THRESHOLD,
    settle_timeout: int = SETTLE_TIMEOUT,
) -> bool:
    # timeout: been too long since last command, declare settled
    if cycles_since_command >= settle_timeout:
        return False

    # not enough history yet
    if len(recent) < NUM_READINGS:
        return True

    tail = recent[-NUM_READINGS:]
    return (max(tail) - min(tail)) > threshold


# basic class lol
class AnalyzeAgent(BaseAgent):
    def __init__(self):
        super().__init__(
        name="analyze_agent",
        description=(
            "Reads sensor data in, saves to session state, updates history "
            "with all session states/reading, and computes the drift/error "
            "analysis based off previous history"
        ),
    )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        
        # need to delay so not too overloaded
        await asyncio.sleep(CYCLE_DELAY)

        # need to get tuning parameters
        target_adc         = ctx.session.state.get("target_adc",          TARGET_ADC)
        max_delta          = ctx.session.state.get("max_delta",            MAX_DELTA)
        spike_threshold    = ctx.session.state.get("spike_threshold",      SPIKE_THRESHOLD)
        low_threshold      = ctx.session.state.get("low_threshold",        LOW_THRESHOLD)
        settle_timeout     = ctx.session.state.get("settle_timeout",       SETTLE_TIMEOUT)
        settling_threshold = ctx.session.state.get("settling_threshold",   SETTLING_THRESHOLD)

        # inc cycle, 

        cycle_count = ctx.session.state.get("cycle_count", 0) + 1
        ctx.session.state["cycle_count"] = cycle_count
 
        #get latest readings, gotta save so the session.state isnt altered mid pipeline or some shit
        reading     = get_latest_reading()
        raw_adc     = reading["raw_adc"]
        electrode_a = reading["electrode_a"]
        electrode_b = reading["electrode_b"]
        flag        = reading.get("flag", "auto")
 
        # save to state
        ctx.session.state["raw_adc"]     = raw_adc
        ctx.session.state["electrode_a"] = electrode_a
        ctx.session.state["electrode_b"] = electrode_b
 
        #save to history
        history: list = list(ctx.session.state.get("history", []))
        history.append({
            "raw_adc":     raw_adc,
            "electrode_a": electrode_a,
            "electrode_b": electrode_b,
            "time_ms":     reading.get("time_ms", 0),
            "flag":        flag,
            "host_time":   time.time(),
            "target_adc":  target_adc,  # snapshot so integral_error stays valid
        })                              # even after ImprovementAgent changes target
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
        ctx.session.state["history"] = history


        # analyzing steps:
    

        # see if target was JUST moved, indicates that it's still settling and sets settle timer 
        prev_target    = ctx.session.state.get("analysis", {}).get("target_adc", target_adc)
        target_changed = target_adc != prev_target
        #michael was also here :)
        if target_changed:
            # Treat target change like a new command — give system time to reach
            # the new setpoint before acting again
            ctx.session.state["last_command_cycle"] = cycle_count
            logger.info(
                "AnalyzeAgent | target changed %d → %d, resetting settle timer",
                prev_target, target_adc,
            )


        # 2. compute:
        # - ewma (average that weighs recent readings more)
        # - coefficient_of_variation, which basically says how spread out the current adc is according to prev readings
        # - integral error, looks at target value rn versus prev ones to see if we are off topic and how much
        # - issettling, indicates if system is still changing
        # - command_effectiveness, says if the command worked last, if we moved in the right direction

        recent = [r["raw_adc"] for r in history[-STATS_WINDOW:]]
        prev_adc = history[-2]["raw_adc"] if len(history) >= 2 else raw_adc
        last_action = ctx.session.state.get("last_action", {})

        # EWMA — noise-filtered estimate of current operating point
        ewma_adc = ewma(recent, alpha=0.3)

        # Std dev over window
        std_adc = round(statistics.stdev(recent), 2) if len(recent) > 1 else 0.0

        # Coefficient of variation — dimensionless stability (replaces 1-std/target)
        cv = coefficient_of_variation(recent)

        # Instantaneous error (raw, for logging + spike/low decisions)
        error_from_target = raw_adc - target_adc

        # Steady-state error — smoothed, this is what DecisionAgent acts on
        steady_state_error = round(ewma_adc - target_adc, 2)

        # Integral error — uses per-entry target so moving target doesn't corrupt it
        integral_err = integral_error(history, window=STATS_WINDOW)

        # Rate of change between last two readings
        roc = raw_adc - prev_adc

        # Settling state — uses cycle-based timeout to avoid noise locking it True
        cycles_since_command = cycle_count - ctx.session.state.get("last_command_cycle", 0)
        settling = is_settling(
            recent,
            cycles_since_command=cycles_since_command,
            threshold=settling_threshold,
            settle_timeout=settle_timeout,
        )

        # Beam search override: if the beam hasn't been found yet (is_low=True),
        # force is_settling=False. During a continuous upward sweep the settle timer
        # resets every cycle (because ActionAgent stamps last_command_cycle each time),
        # which would lock is_settling=True forever and confuse the DecisionAgent.
        # The beam search phase doesn't need settling logic — just keep sweeping.
        if raw_adc < low_threshold:
            settling = False

        # Did last command actually work?
        effectiveness = command_effectiveness(last_action, raw_adc, prev_adc)

        # Spike / low-signal (always checked regardless of settling state)
        is_spike = raw_adc > spike_threshold
        is_low   = raw_adc < low_threshold

        # Electrode asymmetry
        electrode_diff = electrode_a - electrode_b

        # saves in analysis dict to session state :D
        analysis = {
            "timestamp":           time.time(), # exact timestamp
            "cycle_count":         cycle_count, # num cycle we're on

            # Raw readings
            "raw_adc":             raw_adc, # curr reading
            "electrode_a":         electrode_a, 
            "electrode_b":         electrode_b,
            "electrode_diff":      electrode_diff, # calculated !!!! BOOm
            "flag":                flag,

            # Target (snapshot for context)
            "target_adc":          target_adc,
            "target_changed":      target_changed, # true or false

            # Error metrics
            "error_from_target":   error_from_target,    # raw, noisy
            "steady_state_error":  steady_state_error,   # smoothed, act on this
            "integral_error":      integral_err,         # chronic bias detector

            # Signal shape
            "ewma_adc":            ewma_adc, # avg
            "std_adc":             std_adc, # standard dev
            "cv":                  cv,  # stableness !! < 0.02 stable, > 0.10 noisy
            "roc":                 roc, # roc between raw and prev

            # Control state
            "is_settling":         settling,             # True = wait, transient in progress
            "cycles_since_command": cycles_since_command,
            "command_effectiveness": effectiveness,

            # Safety flags (override settling — always act on these)
            "is_spike":            is_spike,
            "is_low":              is_low,

            # Active tuning params (snapshot so LLMs have full context)
            "tuning_params": {
                "target_adc":         target_adc,
                "max_delta":          max_delta,
                "spike_threshold":    spike_threshold,
                "low_threshold":      low_threshold,
                "settle_timeout":     settle_timeout,
                "settling_threshold": settling_threshold,
            },

            "history_length": len(history),
        }

        ctx.session.state["analysis"] = analysis


        logger.info(
            "AnalyzeAgent | cycle=%d adc=%d ewma=%.1f sse=%+.1f "
            "int_err=%.0f cv=%.3f settling=%s effectiveness=%s spike=%s low=%s",
            cycle_count, raw_adc, ewma_adc, steady_state_error,
            integral_err, cv, settling, effectiveness, is_spike, is_low,
        )

        # Emit state_delta so ADK's append_event flushes these writes
        # back to the storage session. Without this, ctx.session.state writes
        # only live on the in-memory copy and are lost between run_async calls.
        state_delta = {
            "raw_adc":            raw_adc,
            "electrode_a":        electrode_a,
            "electrode_b":        electrode_b,
            "cycle_count":        cycle_count,
            "last_command_cycle": ctx.session.state.get("last_command_cycle", 0),
            "history":            history,
            "analysis":           analysis,
        }

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state_delta),
            content={
                "parts": [{
                    "text": (
                        f"[AnalyzeAgent] cycle={cycle_count} "
                        f"adc={raw_adc} ewma={ewma_adc} "
                        f"sse={steady_state_error:+.1f} int_err={integral_err:.0f} "
                        f"cv={cv:.3f} settling={settling} "
                        f"effectiveness={effectiveness} "
                        f"spike={is_spike} low={is_low}"
                    )
                }]
            },
        )