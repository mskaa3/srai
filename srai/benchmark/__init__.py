"""The benchmark module contains classes for evaluating the performance of a model on a dataset."""

from ._base import BaseEvaluator
from .hex_regression_evaluator import HexRegressionEvaluator

__all__ = ["BaseEvaluator", "HexRegressionEvaluator"]
