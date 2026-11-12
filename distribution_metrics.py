import numpy as np


def normalize_distribution(values):
    """Return a normalized finite, nonnegative probability vector, or None."""
    if values is None:
        return None
    try:
        probs = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if probs.ndim != 1 or probs.size == 0 or not np.isfinite(probs).all():
        return None
    if np.any(probs < 0):
        return None
    total = float(probs.sum())
    if total <= 0:
        return None
    return probs / total


def calculate_distribution_metrics(target, prediction):
    """Calculate KL, Jensen-Shannon, Bhattacharyya coefficient, and R².

    KL and JS use natural logarithms and are reported in nats. R² is computed
    per utterance over the class probabilities; it is undefined for a constant
    target vector and is then returned as None.
    """
    target_probs = normalize_distribution(target)
    prediction_probs = normalize_distribution(prediction)
    if (target_probs is None or prediction_probs is None or
            target_probs.shape != prediction_probs.shape):
        return {'kl': None, 'js': None, 'bc': None, 'r2': None}

    eps = 1e-10
    nonzero_target = target_probs > 0
    kl = np.sum(
        target_probs[nonzero_target]
        * np.log(target_probs[nonzero_target] /
                 np.maximum(prediction_probs[nonzero_target], eps))
    )

    midpoint = 0.5 * (target_probs + prediction_probs)
    js = 0.0
    for distribution in (target_probs, prediction_probs):
        nonzero = distribution > 0
        js += 0.5 * np.sum(
            distribution[nonzero]
            * np.log(distribution[nonzero] / np.maximum(midpoint[nonzero], eps))
        )

    bc = np.sum(np.sqrt(target_probs * prediction_probs))
    target_variance = np.sum((target_probs - np.mean(target_probs)) ** 2)
    r2 = None
    if target_variance > eps:
        r2 = 1.0 - np.sum((target_probs - prediction_probs) ** 2) / target_variance

    return {
        'kl': max(float(kl), 0.0),
        'js': max(float(js), 0.0),
        'bc': float(bc),
        'r2': float(r2) if r2 is not None else None,
    }
