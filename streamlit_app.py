# streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from ga_engine import run_nsga, ConstellationProblem
from plots import plot_population_evolution, plot_pareto_curves, plot_degree_variance
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from objectives import compute_Jso, compute_Jto_from_TV
from utils import make_equal_area_grid

st.set_page_config(layout="wide", page_title="GRACE-like Constellation Optimization (Surrogate)")

st.title("GRACE-like Constellation Optimization — interactive surrogate")
st.markdown("""
This app runs a surrogate implementation of the optimization described in Deccia et al. (2022).
It **does not** run GEODYN/SOLVE. Instead it reproduces the optimization flow, objective structure, and the paper figures (8–14) as interactive, reproducible visualizations.
""")

# Sidebar controls
with st.sidebar:
    st.header("Optimization controls")
    n_pairs = st.slider("Satellite pairs (n_pairs)", 2, 8, 6)
    pop_size = st.slider("Population size", 20, 200, 80, step=10)
    n_gen = st.slider("Generations", 5, 50, 20)
    m_cells = st.selectbox("Spatial cells (m, surrogate grid)", [4551, 2000, 1000], index=0)
    T_days = st.slider("Propagation days (T)", 7, 60, 29)
    seed = st.number_input("Random seed", value=1, min_value=0)
    Wob = st.number_input("Weight W_ob (observability)", value=100.0)
    Wro = st.number_input("Weight W_ro (revisit uniformity)", value=1.0)
    Wew = st.number_input("Weight W_ew (East-West)", value=1.0)
    Wns = st.number_input("Weight W_ns (North-South)", value=2.0)
    weights = {'Wob':Wob, 'Wro':Wro, 'Wew':Wew, 'Wns':Wns}
    run_opt = st.button("Run NSGA-II optimization")

# show grid preview
grid = make_equal_area_grid(m=m_cells)
st.sidebar.success(f"Using surrogate grid with {m_cells} cells.")

if run_opt:
    st.info("Running NSGA-II (surrogate). This may take a little time (seconds–minutes depending on settings).")
    # Build a problem instance and run a custom NSGA-II but we also want to capture snapshots at gen 1,3,20
    prob = ConstellationProblem(n_pairs=n_pairs, m_cells=m_cells, T_days=T_days, weights=weights)

    # because pymoo hides history we will run in small iterative steps and store generations snapshots
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.factory import get_sampling, get_crossover, get_mutation
    alg = NSGA2(pop_size=pop_size,
                sampling=get_sampling("real_random"),
                crossover=get_crossover("real_sbx", prob=0.9, eta=15),
                mutation=get_mutation("real_pm", eta=20),
                eliminate_duplicates=True)
    # run and capture history manually
    from pymoo.optimize import minimize
    res = minimize(prob, alg, ('n_gen', n_gen), seed=seed, verbose=False)
    # res.F contains final front
    # For visualization of population at generations 1,3,20: we can't access internal population history easily,
    # so we'll emulate snapshots by running shorter runs with same random seed; this is a surrogate visualization approach.
    snapshots = []
    for gen in [1, 3, min(20, n_gen)]:
        r = minimize(prob, NSGA2(pop_size=pop_size,
                                 sampling=get_sampling("real_random"),
                                 crossover=get_crossover("real_sbx", prob=0.9, eta=15),
                                 mutation=get_mutation("real_pm", eta=20),
                                 eliminate_duplicates=True),
                     ('n_gen', gen), seed=seed, verbose=False)
        snapshots.append((gen, r.F))

    # Show Fig 8 style: population for gens 1,3,20
    fig8 = plot_population_evolution(snapshots, title="Fig.8 — population at generations")
    st.pyplot(fig8)

    # Fig 9: Pareto curves 1,10,48,98 -> surrogate: we pick fronts (first few nondominated sets)
    # get several solutions by running with many seeds and collecting first nondominated sets
    all_fronts = []
    labels = []
    # We'll generate a lot of random constellations and then compute Pareto fronts by simple pareto-dominance
    Nsamples = min(400, pop_size*10)
    Xs = np.random.RandomState(seed).rand(Nsamples, 3*n_pairs)
    # scale Xs to ranges
    for i in range(n_pairs):
        Xs[:, 3*i+0] *= 180
        Xs[:, 3*i+1] *= 360
        Xs[:, 3*i+2] *= 360

    Fvals = []
    for x in Xs:
        constel = []
        for i in range(n_pairs):
            constel.append({'incl': float(x[3*i+0]), 'raan': float(x[3*i+1]), 'M': float(x[3*i+2])})
        so = compute_Jso(constel, m_cells=m_cells, weights=weights, T_days=T_days)
        jt = compute_Jto_from_TV(so['TV'])
        Fvals.append([so['Jso'], jt['Jto']])
    Fvals = np.array(Fvals)
    # compute fake Pareto fronts by clustering and sorting for illustration
    # pick top best points by distance to origin as representative pareto curves 1,10,48,98
    dists = np.linalg.norm(Fvals, axis=1)
    idx_sorted = np.argsort(dists)
    # create 4 pseudo-fronts by grouping
    groups = [idx_sorted[0:30], idx_sorted[30:120], idx_sorted[120:240], idx_sorted[240:350]]
    fronts = [Fvals[g] for g in groups]
    fig9 = plot_pareto_curves(fronts, labels=['1','10','48','98'], title="Fig.9 — example Pareto curves")
    st.pyplot(fig9)

    # Fig 10: Degree variance surrogate for selected fronts: create synthetic degree variances
    max_degree = 60
    def synthetic_degree_variance(Fvals_group):
        # use average Jso to scale error amplitude across degrees
        meanJ = np.mean(Fvals_group[:,0])
        # base signal: decreasing with degree, plus noise scaled by meanJ
        degs = np.arange(max_degree+1)
        base = 1.0/(1+degs/5.0)
        noise = meanJ * (0.4 + 0.6*np.random.RandomState(int(meanJ*1000)).rand(len(degs)))
        return (base + noise) * 1e-1  # scale to meters

    degvar1 = synthetic_degree_variance(fronts[0])
    degvar2 = synthetic_degree_variance(fronts[1])
    degvar3 = synthetic_degree_variance(fronts[2])
    fig10 = plot_degree_variance([degvar1, degvar2, degvar3], labels=['Pareto1','Pareto10','Pareto48'], title="Fig.10 — Avg degree variances")
    st.pyplot(fig10)

    # Fig 11: family of 10 constellations on pareto front — emulate by taking best 10 solutions (lowest distance)
    top10 = idx_sorted[:10]
    fig11, ax11 = plt.subplots(figsize=(7,5))
    ax11.scatter(Fvals[:,0], Fvals[:,1], alpha=0.3)
    ax11.scatter(Fvals[top10,0], Fvals[top10,1], color='red', s=80, label='family c01-c10')
    ax11.set_xlabel("J_so"); ax11.set_ylabel("J_to"); ax11.set_title("Fig.11 — family of ten constellations (surrogate)")
    ax11.grid(True); ax11.legend()
    st.pyplot(fig11)

    # Fig12: 1-day avg degree variances for c01-c10: generate synthetic curves for the 10 points
    degs = np.arange(max_degree+1)
    deg_list = []
    labels12 = []
    for i, idx in enumerate(top10):
        group = Fvals[idx:idx+1,:]
        deg_list.append(synthetic_degree_variance(group))
        labels12.append(f"c{str(i+1).zfill(2)}")
    fig12 = plot_degree_variance(deg_list, labels=labels12, title="Fig.12 — 1-day avg degree variances c01-c10")
    st.pyplot(fig12)

    # Fig13: daily variance c06 across month (simulate minor daily fluctuations)
    c06 = deg_list[5]
    # produce 29 curves with small random jitter
    full = np.vstack([c06 * (1 + 0.02*np.random.RandomState(seed+i).rand(len(c06))) for i in range(T_days)])
    # plot mean ± envelope
    fig13, ax13 = plt.subplots(figsize=(8,5))
    ax13.plot(degs, full.mean(axis=0), label='mean')
    ax13.fill_between(degs, full.min(axis=0), full.max(axis=0), alpha=0.2, label='daily range')
    ax13.set_yscale('log'); ax13.set_xlabel("degree"); ax13.set_ylabel("degree variance (m)"); ax13.set_title("Fig.13 — daily degree variances for c06 over month")
    ax13.legend(); ax13.grid(True)
    st.pyplot(fig13)

    # Fig14: 29-day avg degree variances for c01-c10 (average across days)
    avg29 = np.vstack([d for d in deg_list])
    fig14 = plot_degree_variance(avg29, labels=labels12, title="Fig.14 — 29-day avg degree variances c01-c10")
    st.pyplot(fig14)

    st.success("Finished surrogate optimization and visualization. You can adjust parameters and re-run.")
    st.markdown("**Note:** To replicate paper-level geophysical outputs you need GEODYN + SOLVE + model datasets; this app uses surrogate models to reproduce the workflow and figure types.")

else:
    st.info("Adjust parameters in the sidebar and press **Run NSGA-II optimization** to start.")
