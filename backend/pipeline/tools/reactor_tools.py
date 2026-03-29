# hi (;

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_bridge = None

def init_tools(bridge) -> None:
    global _bridge
    _bridge = bridge

def _require_bridge():
    if _bridge is None:
        raise RuntimeError("Tools not intialised - call init_tool(bridge) first")
    return _bridge

def get_latest_reading() -> dict:
    return _require_bridge().get_latest_reading()

def set_electrodes(a: int, b: int) -> dict:
    return _require_bridge().set_electrodes(a,b)

def set_electrode_a(value: int) -> dict:
    return _require_bridge().set_electrode_a(value)

def set_electrode_b(value: int) -> dict:
    return _require_bridge().set_electrode_b(value)

def set_params(
    target_adc: Optional[int] = None,
    max_delta: Optional[int] = None,
    spike_threshold: Optional[int] = None,
    low_threshold: Optional[int] = None,
    photo_min: Optional[int] = None,
    photo_max: Optional[int] = None,
    dac_min: Optional[int] = None,
    dac_max: Optional[int] = None,
) -> dict:
    
    return _require_bridge().set_params(
        target_adc=target_adc,
        max_delta=max_delta,
        spike_threshold=spike_threshold,
        low_threshold=low_threshold,
        photo_min=photo_min,
        photo_max=photo_max,
        dac_min=dac_min,
        dac_max=dac_max,
    )

def set_backup() -> dict:

    return _require_bridge().set_backup()

def set_ai_mode() -> dict:

    return _require_bridge().set_ai_mode()


    
