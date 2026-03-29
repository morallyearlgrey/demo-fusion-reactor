"""
WebSocket Server for Fusion Reactor Frontend

Broadcasts real-time reactor state to connected frontend clients.
Runs alongside the main pipeline in a separate thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logger = logging.getLogger(__name__)

# Global state broadcaster
class ReactorBroadcaster:
    def __init__(self):
        self.connections: Set[WebSocket] = set()
        self._latest_state = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.connections)}")

        # Send latest state immediately upon connection
        if self._latest_state:
            try:
                await websocket.send_json(self._latest_state)
            except Exception as e:
                logger.error(f"Failed to send initial state: {e}")

    def disconnect(self, websocket: WebSocket):
        self.connections.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        self._latest_state = message
        disconnected = set()

        logger.info(f"[WEBSOCKET_SERVER] Broadcasting to {len(self.connections)} clients: type={message.get('type')}")

        for connection in self.connections:
            try:
                await connection.send_json(message)
                logger.info(f"[WEBSOCKET_SERVER] Successfully sent message to client")
            except Exception as e:
                logger.error(f"[WEBSOCKET_SERVER] Failed to send to client: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.connections.discard(conn)

    def update_state(self, state: dict):
        """Update state (called from pipeline thread)"""
        self._latest_state = state


# Global broadcaster instance
broadcaster = ReactorBroadcaster()

# FastAPI app
app = FastAPI(title="Fusion Reactor WebSocket API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "online", "service": "fusion-reactor-websocket"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "connected_clients": len(broadcaster.connections)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await broadcaster.connect(websocket)
    try:
        # Keep connection alive and handle incoming messages
        while True:
            # We don't expect messages from frontend, but keep connection alive
            data = await websocket.receive_text()
            # Could handle control commands from frontend here
            logger.debug(f"Received from client: {data}")
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        broadcaster.disconnect(websocket)


async def run_server_async(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server asynchronously"""
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    asyncio.run(run_server_async(host, port))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
