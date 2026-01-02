"""Logging configuration for BioLM."""

import logging
import sys
import warnings

# Suppress Hydra working directory warning
warnings.filterwarnings(
    "ignore",
    message="Future Hydra versions will no longer change working directory at job runtime by default.",
    category=UserWarning,
)

# Suppress transformers deprecation warnings
warnings.filterwarnings(
    "ignore",
    message="`evaluation_strategy` is deprecated",
    category=FutureWarning,
)

# Suppress MLflow filesystem backend warning
warnings.filterwarnings(
    "ignore",
    message="Filesystem tracking backend.*is deprecated",
    category=FutureWarning,
)


def setup_logging(log_file=None):
    """Configure logging for clean, real-time output."""
    # Configure root logger to use stdout and flush after each emit to preserve ordering
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    # StreamHandler for stdout
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    orig_emit = stream_handler.emit

    def emit_and_flush(record):
        orig_emit(record)
        try:
            stream_handler.flush()
        except Exception:
            pass

    stream_handler.emit = emit_and_flush
    root_logger.addHandler(stream_handler)

    # FileHandler for logging to a file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Check if the log file remains empty and remove it
        import os

        def remove_empty_log_file():
            if os.path.exists(log_file) and os.path.getsize(log_file) == 0:
                os.remove(log_file)

        # Schedule removal of empty log file at program exit
        import atexit

        atexit.register(remove_empty_log_file)

    # Redirect stdout to the log file if specified
    if log_file:

        class StreamToLogger:
            def __init__(self, logger, level):
                self.logger = logger
                self.level = level

            def write(self, message):
                if message.strip():
                    self.logger.log(self.level, message.strip())

            def flush(self):
                pass

        stdout_logger = logging.getLogger("stdout")
        sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
        # Map stderr to INFO to avoid misleading "ERROR" prefixes for libraries that print to stderr
        sys.stderr = StreamToLogger(stdout_logger, logging.INFO)

        # Add a flush method to ensure all outputs are written immediately
        class FlushStreamToLogger(StreamToLogger):
            def flush(self):
                for handler in self.logger.handlers:
                    handler.flush()

        sys.stdout = FlushStreamToLogger(stdout_logger, logging.INFO)
        # Keep stderr at INFO level for cleaner logs; genuine errors still go through the root handler
        sys.stderr = FlushStreamToLogger(stdout_logger, logging.INFO)

    root_logger.setLevel(logging.INFO)

    # Make sure transformers loggers propagate to root handler and don't install extra handlers
    import transformers.utils.logging as tf_logging

    # Disable transformers' own default handler to avoid duplicate logs
    try:
        tf_logging.disable_default_handler()
    except Exception:
        pass
    for name in (
        "transformers",
        "transformers.trainer",
        "transformers.trainer_callback",
    ):
        lg = logging.getLogger(name)
        lg.propagate = True
        lg.setLevel(logging.NOTSET)

    class SuppressSavingMessages(logging.Filter):
        def filter(self, record):
            message = record.getMessage()
            # Allow evaluation results and training logs
            if (
                "'eval_" in message
                or message.startswith("{")
                or "Running Evaluation" in message
                or "Num examples" in message
                or "'loss'" in message
                or "'learning_rate'" in message
            ):
                return True
            # Block saving messages
            return not (
                "Saving model checkpoint" in message
                or "Deleting older checkpoint" in message
                or "Configuration saved in" in message
                or "Model weights saved in" in message
                or "tokenizer config file saved in" in message
                or "Special tokens file saved in" in message
            )

    # Suppress saving-related messages but keep trainer logging
    trainer_logger = logging.getLogger("transformers.trainer")
    trainer_logger.setLevel(logging.INFO)
    trainer_logger.addFilter(SuppressSavingMessages())

    # Allow all logging from trainer_callback (metrics)
    callback_logger = logging.getLogger("transformers.trainer_callback")
    callback_logger.setLevel(logging.INFO)

    logging.getLogger("transformers.modeling_utils").setLevel(logging.CRITICAL)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.CRITICAL)
    logging.getLogger("transformers.configuration_utils").setLevel(logging.CRITICAL)
    logging.getLogger("transformers.generation.configuration_utils").setLevel(
        logging.WARNING
    )
    logging.getLogger("transformers.models").setLevel(logging.CRITICAL)
    logging.getLogger("transformers.integrations.tensor_parallel").setLevel(
        logging.ERROR
    )
    logging.getLogger("accelerate.accelerator").setLevel(logging.WARNING)
