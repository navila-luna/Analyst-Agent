import json
import logging
from dataclasses import asdict

from app.agent.models import TraceEvent


class LoggingTraceRecorder:
    def __init__(
        self,
        logger: logging.Logger | None = None,
    ):
        self._logger = logger or logging.getLogger(
            "analyst_agent"
        )

    def record(self, event: TraceEvent) -> None:
        self._logger.info(json.dumps(asdict(event)))