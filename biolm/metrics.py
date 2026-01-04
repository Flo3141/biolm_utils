"""Metrics and scaling helpers factored out of train_utils."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    precision_recall_fscore_support,
)


class LogScaler:
    def fit_transform(self, data):
        return np.log(data)

    def inverse_transform(self, data):
        return np.exp(data)


class IdentityScaler:
    def fit_transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


def compute_metrics_for_regression(dataset, savepath):
    def _compute_metrics(pred):
        logits, labels = pred
        logits = logits.squeeze().tolist()
        labels = labels.squeeze().tolist()
        mse = mean_squared_error(labels, logits)
        spearman_rho, _ = spearmanr(logits, labels)
        return {
            "mse": mse,
            "spearman rho": spearman_rho,
        }

    return _compute_metrics


def compute_metrics_for_classification(dataset, savepath):
    def _compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )
        acc = accuracy_score(labels, preds)
        target_names = dataset.LE.classes_.tolist()
        used_labels = list(range(len(target_names)))
        report = classification_report(
            labels,
            preds,
            output_dict=True,
            target_names=target_names,
            labels=used_labels,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(savepath / "classification_report.csv")
        logging.info(
            classification_report(
                labels,
                preds,
                target_names=target_names,
                labels=used_labels,
                zero_division=0,
            )
        )
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    return _compute_metrics
