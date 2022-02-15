import numpy as np
import scipy.stats

def confidence_interval(mu, std, confidence=0.9):
    h = std * scipy.stats.norm.ppf((1 + confidence) / 2)
    low_interval = mu - h
    up_interval = mu + h
    return mu, low_interval, up_interval

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), scipy.stats.sem(a, axis=0)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h