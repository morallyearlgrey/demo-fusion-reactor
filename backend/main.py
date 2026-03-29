from __future__ import annotations
 
import argparse
import asyncio
import logging
import os
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace: