import logging

import torch
import torch.nn as nn
from transformers import Trainer


class RegressionTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        targets = inputs.pop("labels")
        inputs.pop("qualities", None)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        targets = targets.type(logits.dtype)
        loss = torch.nn.functional.mse_loss(logits.squeeze(), targets.squeeze())
        return (loss, outputs) if return_outputs else loss

    def log_metrics(self, split, metrics):
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        super().log_metrics(split, metrics)
        logging.info(f"[{timestamp}] {split} metrics: {metrics}")

    def log(self, logs, start_time=None):
        """Override the log method to include timestamps for metrics only."""
        from datetime import datetime

        # Call the parent log method to preserve default logging behavior
        # Pass start_time if provided (for transformers>=4.x compatibility)
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)

        # Add custom logging for metrics
        if "loss" in logs or "learning_rate" in logs:  # Metrics-related logs
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_logs = {
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in logs.items()
            }
            logging.info(f"[{timestamp}] Training metrics: {formatted_logs}")


class WeightedRegressionTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        targets = inputs.pop("labels")
        qualities = inputs.pop("qualities")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = torch.nn.functional.mse_loss(
            logits.squeeze(), targets.squeeze(), reduction="none"
        )
        loss = torch.mean(qualities * loss)
        return (loss, outputs) if return_outputs else loss


class WeightedSamplingTrainer(Trainer):
    def __init__(self, weights, **args):
        self.weights = weights
        super().__init__(**args)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.weights.to(self.args.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
