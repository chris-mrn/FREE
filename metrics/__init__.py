# Backward-compat shim — canonical location is now evaluation.metrics
from evaluation.metrics import InceptionMetrics

__all__ = ['InceptionMetrics']
