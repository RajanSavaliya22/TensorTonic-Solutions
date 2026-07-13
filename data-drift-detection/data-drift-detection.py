import numpy as np
def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    reference_counts = np.asarray(reference_counts, dtype=int)
    production_counts = np.asarray(production_counts, dtype=int)
    norm_rc = reference_counts / np.sum(reference_counts)
    norm_pc = production_counts / np.sum(production_counts)

    tvd = 0.5*np.sum(abs((norm_rc-norm_pc)))

    return {'score': float(tvd), 'drift_detected': bool(tvd > threshold)}