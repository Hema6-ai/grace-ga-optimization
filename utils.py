import numpy as np

def make_equal_area_grid(m=4551):
    """
    Create a simple equal-area grid placeholder.
    We return lat-lon cell centers for m cells approx uniformly distributed.
    This is a surrogate grid (paper uses 3-deg equal-area; we use spherical Fibonacci sampling).
    """
    # spherical Fibonacci points for near-uniform distribution
    indices = np.arange(0, m, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/m) - np.pi/2
    theta = np.pi * (1 + 5**0.5) * indices
    lats = np.degrees(phi)
    lons = np.degrees((theta % (2*np.pi)))
    return np.vstack([lats, lons]).T  # shape (m,2)

def gini_coefficient(arr):
    """
    Compute Gini coefficient for 1D non-negative array.
    Formula adapted to match paper's usage.
    """
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return 0.0
    # zero or negative guard
    a = a - a.min() + 1e-12
    a_sorted = np.sort(a)
    n = a.size
    index = np.arange(1, n+1)
    return (2.0 * np.sum(index * a_sorted) / (n * np.sum(a_sorted))) - (n + 1) / n

def normalize01(x):
    x = np.asarray(x, dtype=float)
    if x.max() - x.min() < 1e-12:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())
