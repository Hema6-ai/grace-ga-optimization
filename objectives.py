# objectives.py
import numpy as np
from utils import gini_coefficient, make_equal_area_grid

def compute_groundtracks_for_constellation(constel, m_cells, T_days=29, samples_per_rev=300):
    """
    Surrogate: For each satellite pair in constel, create groundtrack samples across cells.
    constel: list of pairs, each pair is a dict {'incl':deg,'raan':deg,'M':deg}
    m_cells: number of spatial cells (we'll use numpy indices)
    Returns:
      observations_per_cell: counts of how many times a cell is observed (size m_cells)
      temporal_matrix: binary matrix shape (m_cells, T_days * temporal_bins_per_day) indicating visits
      local_vel_vectors: for each cell a small set of (east,north) vectors aggregated
    NOTE: This is a simplified surrogate ground-track simulator — it does NOT propagate orbits physically.
    """
    m = m_cells
    # initialize
    obs_counts = np.zeros(m, dtype=int)
    # for temporal we choose 16 bins/day
    bins_per_day = 16
    TV = np.zeros((m, bins_per_day * T_days), dtype=int)
    # for directionality, gather random direction vectors biased by inclination
    local_vels = [[] for _ in range(m)]
    rng = np.random.default_rng(0)

    # treat each pair as producing many samples; map samples to cell indices via hashing of orbit parameters
    for p in constel:
        incl = p['incl']
        raan = p['raan']
        M = p['M']
        # number of total samples ~ T_days * samples_per_rev
        total_samples = int(T_days * samples_per_rev)
        # create pseudo groundtrack sample cell indices
        # deterministic pseudo-random mapping so results reproducible for same constellation parameters
        seed = int((incl*1000 + raan*10 + M) % (2**31-1))
        local_rng = np.random.default_rng(seed)
        cell_indices = local_rng.integers(0, m, size=total_samples)
        times = local_rng.integers(0, bins_per_day * T_days, size=total_samples)
        # update counts
        for ci, t in zip(cell_indices, times):
            obs_counts[ci] += 1
            TV[ci, t] = 1
            # directionality vector — inclination controls north/south bias
            # map inclination to a direction: polar (incl ~90) -> strong N component, low incl -> strong E component
            inc_rad = np.deg2rad(incl)
            # base velocity vector in local east,north coordinates (unit)
            east = np.cos(inc_rad) + local_rng.normal(scale=0.05)
            north = np.sin(inc_rad) + local_rng.normal(scale=0.05)
            # normalize
            vnorm = max(1e-6, (east**2 + north**2)**0.5)
            local_vels[ci].append((east / vnorm, north / vnorm))
    return obs_counts, TV, local_vels

def compute_Jso(constel, m_cells=4551, weights=None, T_days=29):
    """
    Compute spatial objective J_so broken into subobjectives Job, Jro, Jew, Jns
    weights: dict with keys 'Wob','Wro','Wew','Wns'
    """
    if weights is None:
        weights = {'Wob':100.0, 'Wro':1.0, 'Wew':1.0, 'Wns':2.0}

    obs_counts, TV, local_vels = compute_groundtracks_for_constellation(constel, m_cells, T_days=T_days)

    # Job: fraction of unobserved cells
    m = m_cells
    N = np.count_nonzero(obs_counts)  # number of observed cells
    Job = 1.0 - N / m

    # Jro: Gini of repeat-observations (use obs_counts for M)
    M = np.where(obs_counts > 1, obs_counts, 0)
    Jro = gini_coefficient(M)

    # Jew, Jns: compute Bew (east) and Bns (north) for each cell
    Bew = np.zeros(m)
    Bns = np.zeros(m)
    for i in range(m):
        vs = local_vels[i]
        if len(vs) == 0:
            Bew[i] = 1.0  # worst
            Bns[i] = 1.0
            continue
        # form 2xn matrix A where each column is [east; north]
        A = np.array(vs).T  # shape 2 x n
        # compute B = I - (A A^T) / n  (paper uses 1 - A^T A / n; we use analogous scalar measures)
        # We will use the average projection energy on east and north
        cov = np.cov(A)
        # use diag inversely: higher variance -> better coverage -> lower B
        Bew[i] = 1.0 / (1e-6 + (cov[0,0]))
        Bns[i] = 1.0 / (1e-6 + (cov[1,1]))
    Bew = normalize01(Bew)
    Bns = normalize01(Bns)

    # compute Jew/Jns as mean energy + Gini uniformity term (surrogate)
    def combined_metric(B):
        meanterm = np.mean(B)
        giniterm = gini_coefficient(B)
        return meanterm + 0.5 * giniterm

    Jew = combined_metric(Bew)
    Jns = combined_metric(Bns)

    # weighted sum
    Jso = weights['Wob']*Job + weights['Wro']*Jro + weights['Wew']*Jew + weights['Wns']*Jns
    # normalized to [0,1] by a heuristic scaling for display stability
    Jso_norm = Jso / (weights['Wob'] + weights['Wro'] + weights['Wew'] + weights['Wns'] + 1e-9)
    return {
        'Jso': float(Jso_norm),
        'Job': float(Job),
        'Jro': float(Jro),
        'Jew': float(Jew),
        'Jns': float(Jns),
        'obs_counts': obs_counts,
        'TV': TV
    }

def compute_Jto_from_TV(TV):
    """
    Compute temporal objective J_to from temporal visit matrix TV (m_cells x total_bins)
    Use the paper's Gini-based formulation across temporal vectors.
    """
    # TV is binary presence per temporal bin. For each spatial cell build TV_i vector sums (over days)
    # Following paper's approach: compute Gini of each cell's temporal vector sum and then combined metric
    m = TV.shape[0]
    # sum across bins per cell = number of temporal bins observed for that cell over T days
    tv_sums = TV.sum(axis=1)
    # compute Gini across cells on temporal coverage per cell -> G_TV
    G_TV = gini_coefficient(tv_sums)
    # combined metric: mean + 0.5*gini of G_TV across cells (surrogate)
    mean_term = np.mean(tv_sums) / (TV.shape[1] + 1e-9)
    Jto = G_TV + 0.5 * mean_term
    # normalized
    Jto_norm = float(np.clip(Jto, 0.0, 1.0))
    return {'Jto': Jto_norm, 'G_TV': float(G_TV), 'tv_sums': tv_sums}
