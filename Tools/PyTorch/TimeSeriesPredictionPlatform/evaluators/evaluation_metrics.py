# SPDX-License-Identifier: Apache-2.0

import os
import pickle
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class AbstractMetric(ABC):
    @staticmethod
    @abstractmethod
    def __call__(label, pred, weights):
        pass


class SMAPE(AbstractMetric):
    name = "SMAPE"

    @staticmethod
    def __call__(labels, preds, weights):
        if weights.shape == (0, 0):
            weights = None
        return 100 * np.average(2 * np.abs(preds - labels) / (np.abs(labels) + np.abs(preds)), weights=weights)


def numpy_normalised_quantile_loss(y, y_pred, quantile):
    prediction_underflow = y - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.0) + (1.0 - quantile) * np.maximum(
        -prediction_underflow, 0.0
    )
    loss = weighted_errors.mean()
    normaliser = abs(y).mean()
    return 2 * loss / normaliser


class P50_loss(AbstractMetric):
    name = "P50"
    selector = 1

    @staticmethod
    def __call__(labels, preds, weights):
        if weights.shape != (0, 0):
            raise ValueError("Weights not currently supported for quantile metrics")
        return numpy_normalised_quantile_loss(labels, preds, 0.5)


class P90_loss(AbstractMetric):
    name = "P90"
    selector = 2

    @staticmethod
    def __call__(labels, preds, weights):
        if weights.shape != (0, 0):
            raise ValueError("Weights not currently supported for quantile metrics")
        return numpy_normalised_quantile_loss(labels, preds, 0.9)


# Normalized Deviation
class ND(AbstractMetric):
    name = "ND"

    @staticmethod
    def __call__(labels, preds, weights, return_individual=False):
        if weights.shape == (0, 0):
            if return_individual:
                return np.abs(labels - preds) / np.abs(labels)
            return np.sum(np.abs(labels - preds)) / np.sum(np.abs(labels))
        else:

            values = np.abs(labels - weights)
            if return_individual:
                return values * weights / np.sum(np.abs(labels))
            return np.sum(values * weights) / np.sum(np.abs(labels) * weights)


class MAE(AbstractMetric):
    name = "MAE"

    @staticmethod
    def __call__(labels, preds, weights, return_individual=False):
        if weights.shape == (0, 0):
            if return_individual:
                return mean_absolute_error(preds, labels, multioutput="raw_values")
            return mean_absolute_error(labels, preds)
        else:

            values = mean_absolute_error(preds, labels, multioutput="raw_values")
            if return_individual:
                return values * weights
            return np.sum(values * weights) / np.sum(weights)


class MSE(AbstractMetric):
    name = "MSE"

    @staticmethod
    def __call__(labels, preds, weights, return_individual=False):
        if weights.shape == (0, 0):
            if return_individual:
                return mean_squared_error(preds, labels, multioutput="raw_values")
            return mean_squared_error(labels, preds)
        else:

            values = mean_squared_error(preds, labels, multioutput="raw_values")
            if return_individual:
                return values * weights
            return np.sum(values * weights) / np.sum(weights)


class RMSE(AbstractMetric):
    name = "RMSE"

    @staticmethod
    def __call__(labels, preds, weights):

        if weights.shape == (0, 0):
            return np.sqrt(mean_squared_error(labels, preds))
        else:

            values = mean_squared_error(preds, labels, multioutput="raw_values")
            return np.sqrt(np.sum(values * weights) / np.sum(weights))


class R_Squared(AbstractMetric):
    name = "R_Squared"

    @staticmethod
    def __call__(labels, preds, weights, return_individual=False):
        if weights.shape == (0, 0):
            if return_individual:
                return r2_score(preds, labels, multioutput="raw_values")
            return r2_score(preds, labels)
        else:
            values = r2_score(preds, labels, multioutput="raw_values")
            if return_individual:
                return values * weights
            return np.sum(values * weights) / np.sum(weights)


class WMSMAPE(AbstractMetric):
    name = "WMSMAPE"

    @staticmethod
    def __call__(labels, preds, weights, return_individual=False):
        if weights.shape != (0, 0):
            if return_individual:
                return 2 * weights * np.abs(preds - labels) / (np.maximum(labels, 1) + np.abs(preds))
            else:
                return (
                    100.0
                    / np.sum(weights)
                    * np.sum(2 * weights * np.abs(preds - labels) / (np.maximum(labels, 1) + np.abs(preds)))
                )
        if return_individual:
            return 2 * np.abs(preds - labels) / (np.maximum(labels, 1) + np.abs(preds))
        else:
            return 100.0 / len(labels) * np.sum(2 * np.abs(preds - labels) / (np.maximum(labels, 1) + np.abs(preds)))


mapping = {
    "SMAPE": SMAPE,
    "WMSMAPE": WMSMAPE,
    "MSE": MSE,
    "MAE": MAE,
    "P50": P50_loss,
    "P90": P90_loss,
    "RMSE": RMSE,
    "R_Squared": R_Squared,
    "ND": ND,
}


class MetricEvaluator:
    def __init__(self, config):
        self.output_selector = config.evaluator.get("output_selector", None)
        self.label_selector = config.evaluator.get("label_selector", None)
        self.metrics = []
        self.visualize_path = config.evaluator.get("visualize_path", None)
        self.visualize_num = config.evaluator.get("visualize_num", 0)

        self.scalers = pickle.load(open(os.path.join(config.dataset.dest_path, "composite_scaler.bin"), "rb"))

        for metric in config.evaluator.metrics:
            if metric not in mapping.keys():
                raise ValueError("No metric of name: {metric}".format(metric=metric))
            self.metrics.append(mapping[metric]())
        self.precision = config.evaluator.precision
        self.config = config

    def __call__(self, labels, preds, weights=np.zeros((0, 0)), ids=np.zeros((0, 0))):
        results = {}

        if len(weights.shape) > 2:
            weights = np.squeeze(weights, axis=2)
        for metric in [metric for metric in self.metrics if metric.name in ["P50", "P90"]]:
            q_preds, q_labels = self.select(preds, labels, metric.selector, None)

            q_preds = self.scalers.inverse_transform_targets(q_preds.copy(), ids)
            q_labels = self.scalers.inverse_transform_targets(q_labels.copy(), ids)
            results[metric.name] = (
                np.round(metric(q_labels, q_preds, weights), self.precision) if np.all(np.isfinite(q_preds)) else np.NaN
            )

        preds, labels = self.select(preds, labels, self.output_selector, self.label_selector)
        preds = self.scalers.inverse_transform_targets(preds, ids)
        if self.config.evaluator.get("test_prediction_path", None):
            np.savetxt(self.config.evaluator.get("test_prediction_path"), preds, delimiter=",")
        labels = self.scalers.inverse_transform_targets(labels, ids)

        # naively, we are going to assume that all calls to visualize will be conducted on horizon 1 prediction for now
        # this is a stopgap, and will take some work to generalize to horizon=N

        if self.visualize_num:
            individual_losses = self.metrics[0](labels, preds, weights, True)
            loss_df = (
                pd.DataFrame({"id": ids[:, 0], "loss": individual_losses[:, 0]}, index=range(len(individual_losses)))
                .groupby("id")
                .agg("sum")
                .sort_values("loss")
                .reset_index()
            )
            min_ids = loss_df["id"].loc[: self.visualize_num]
            min_losses = loss_df["loss"].loc[: self.visualize_num]
            max_ids = loss_df["id"].loc[len(loss_df) - self.visualize_num - 1 :]
            max_losses = loss_df["loss"].loc[len(loss_df) - self.visualize_num - 1 :]
            group_ids = max_ids.append(min_ids)
            group_losses = max_losses.append(min_losses)
            idx = 0
            for group_id in group_ids:
                plt.figure(idx)
                rows = ids == group_id

                id_preds = preds[rows]
                id_labels = labels[rows]
                plt.plot(id_preds, label="Predicted")
                plt.plot(id_labels, label="Actual")
                plt.legend(loc="best")
                plt.title("ID: {} Loss: {}".format(group_id, group_losses.iloc[idx]))
                plt.savefig(self.visualize_path + "{}.png".format(group_id))
                idx += 1

        for metric in [metric for metric in self.metrics if metric.name not in ["P50", "P90"]]:
            results[metric.name] = (
                np.round(metric(labels, preds, weights), self.precision) if np.all(np.isfinite(preds)) else np.NaN
            )
        targets = self.config.evaluator.get("targets", None)
        if targets is not None:
            missed_targets = {}
            for target in targets:
                if target.objective == "MIN":
                    if target.value < results[target.name]:
                        missed_targets[target.name] = results[target.name]
                if target.objective == "MAX":
                    if target.value > results[target.name]:
                        missed_targets[target.name] = results[target.name]
            if len(missed_targets) > 0:
                raise ValueError("Target metrics not achieved: %s" % str(missed_targets))

        return results

    def select(self, preds, labels, output_selector=None, label_selector=None):
        if len(preds.shape) > 2:
            if output_selector is not None:
                preds = preds[:, :, output_selector]
            else:
                preds = np.squeeze(preds, axis=2)
        if len(labels.shape) > 2:
            if label_selector is not None:
                labels = labels[:, :, label_selector]
            else:
                labels = np.squeeze(labels, axis=2)
        return preds, labels
