"""
overwatch.py

Utility class for creating a centralized/standardized logger (built on Rich) and accelerate handler.
"""
import logging
import logging.config
import os
from logging import LoggerAdapter
from typing import Union

# Overwatch Default Format String
RICH_FORMATTER, DATEFMT = "| >> %(message)s", "%m/%d [%H:%M:%S]"

# Set Logging Configuration
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"simple-console": {"format": RICH_FORMATTER, "datefmt": DATEFMT}},
    "handlers": {
        "console": {
            "class": "rich.logging.RichHandler",
            "formatter": "simple-console",
            "markup": True,
            "rich_tracebacks": True,
            "show_level": True,
            "show_path": True,
            "show_time": True,
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}
logging.config.dictConfig(LOG_CONFIG)


# === Custom Contextual Logging Logic ===
class ContextAdapter(LoggerAdapter):
    def __init__(self, logger, extra):
        super().__init__(logger, extra)

    CTX_PREFIXES = {0: "[*] "} | {idx: "|=> ".rjust(4 + (idx * 4)) for idx in [1, 2, 3]}

    def process(self, msg, kwargs):
        ctx_level = kwargs.pop("ctx_level", 0)
        return f"{self.CTX_PREFIXES[ctx_level]}{msg}", kwargs


class DistributedOverwatch:
    def __init__(self, name: str) -> None:
        """Initializer for an Overwatch object that wraps logging & `accelerate.PartialState`."""
        from accelerate import PartialState

        # Note that PartialState is always safe to initialize regardless of `accelerate launch` or `torchrun`
        #   =>> However, might be worth actually figuring out if we need the `accelerate` dependency at all!
        extra_info = {}  # You can provide additional contextual information here if needed
        print("logging.getLogger(name)", logging.getLogger(name))
        self.logger, self.distributed_state = ContextAdapter(logging.getLogger(name), extra_info), PartialState()

        # Logger Delegation (for convenience; would be nice to just compose & dynamic dispatch eventually)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> only Log `INFO` on Main Process, `ERROR` on others!
        self.logger.setLevel(logging.INFO if self.distributed_state.is_main_process else logging.ERROR)


class PureOverwatch:
    def __init__(self, name: str) -> None:
        """Initializer for an Overwatch object that just wraps logging."""

        print("logging.getLogger(name)", logging.getLogger(name))
        extra_info = {}  # You can provide additional contextual information here if needed
        self.logger = ContextAdapter(logging.getLogger(name), extra_info)

        # Logger Delegation (for convenience; would be nice to just compose & dynamic dispatch eventually)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> INFO
        self.logger.setLevel(logging.INFO)


def initialize_overwatch(name: str) -> Union[DistributedOverwatch, PureOverwatch]:
    return DistributedOverwatch(name) if int(os.environ.get("WORLD_SIZE", -1)) > 1 else PureOverwatch(name)
