"""
WebSocket Bridge

Connects the Fusion Pipeline to the WebSocket broadcaster.
Runs the WebSocket server in a separate thread so it doesn't block the pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Global reference to the broadcaster
_broadcaster = None
_server_thread: Optional[threading.Thread] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def start_websocket_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the WebSocket server in a separate thread"""
    global _broadcaster, _server_thread, _event_loop

    async def run_server_and_keep_loop_running():
        """Run the server and keep event loop running for broadcasts"""
        from websocket_server import app, broadcaster as bc
        import uvicorn

        global _broadcaster
        _broadcaster = bc

        logger.info(f"Starting WebSocket server on {host}:{port}")

        # Create uvicorn config
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        # Run server as a task (doesn't block the event loop)
        await server.serve()

    def run_in_thread():
        global _event_loop
        # Create a new event loop for this thread
        _event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_event_loop)

        try:
            # Run the server coroutine on this loop
            _event_loop.run_until_complete(run_server_and_keep_loop_running())
        except Exception as e:
            logger.error(f"WebSocket server error: {e}", exc_info=True)

    _server_thread = threading.Thread(target=run_in_thread, daemon=True, name="websocket-server")
    _server_thread.start()

    # Give the server time to start
    import time
    time.sleep(1)
    logger.info("WebSocket server thread started")


def broadcast_state(state: dict):
    """
    Broadcast reactor state to all connected WebSocket clients.
    Safe to call from the pipeline (main thread).
    """
    global _broadcaster, _event_loop

    logger.info(f"[BROADCAST] broadcast_state called, broadcaster={_broadcaster is not None}, loop={_event_loop is not None}")

    if _broadcaster is None or _event_loop is None:
        logger.warning("[BROADCAST] Broadcaster or event loop not initialized!")
        return

    # Prepare the message
    message = {
        "type": "state_update",
        "data": {
            "raw_adc": state.get("raw_adc", 0),
            "electrode_a": state.get("electrode_a", 0),
            "electrode_b": state.get("electrode_b", 0),
            "target_adc": state.get("target_adc", 752),
            "cycle_count": state.get("cycle_count", 0),
            "system_mode": _determine_system_mode(state),
            "analysis": state.get("analysis", {}),
            "decision": state.get("decision", {}),
            "last_action": state.get("last_action", {}),
        }
    }

    logger.info(f"[BROADCAST] Prepared message with adc={message['data']['raw_adc']}, electrodes={message['data']['electrode_a']}/{message['data']['electrode_b']}")

    # Schedule the broadcast in the WebSocket event loop
    try:
        asyncio.run_coroutine_threadsafe(
            _broadcaster.broadcast(message),
            _event_loop
        )
        logger.info("[BROADCAST] Successfully scheduled broadcast")
    except Exception as e:
        logger.error(f"[BROADCAST] Could not broadcast state: {e}", exc_info=True)


def _determine_system_mode(state: dict) -> str:
    """Determine system mode from state"""
    # Check if emergency
    spike_threshold = state.get("spike_threshold", 800)
    raw_adc = state.get("raw_adc", 0)

    if raw_adc > spike_threshold:
        return "emergency"

    # Check if in auto or manual mode
    # For now, assume auto mode during normal operation
    return "auto"


def broadcast_agent_log(agent: str, message: str, payload: dict = None, confidence: float = None):
    """
    Broadcast an agent log entry to frontend.
    """
    global _broadcaster, _event_loop

    if _broadcaster is None or _event_loop is None:
        return

    log_message = {
        "type": "agent_log",
        "data": {
            "agent": agent,
            "message": message,
            "actionPayload": payload,
            "confidence": confidence,
        }
    }

    try:
        asyncio.run_coroutine_threadsafe(
            _broadcaster.broadcast(log_message),
            _event_loop
        )
    except Exception as e:
        logger.debug(f"Could not broadcast agent log: {e}")
