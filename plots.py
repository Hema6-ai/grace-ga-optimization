# plots.py
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def plot_population_evolution(hist_F_list, title="Population evolution"):
    """
    hist_F_list: list of tuples (gen_index, F_array) where F_array shape (n_pop,2)
    returns matplotlib fig
    """
    fig, ax = plt.subplots(figsize=(7,5))
    colors = ['C0','C1','C2']
    markers = ['o','s','^']
    for (gi, F), c, m in zip(hist_F_list, colors, markers):
        ax.scatter(F[:,0], F[:,1], c=c, marker=m, alpha=0.7, label=f"gen {gi}")
    ax.set_xlabel("J_so (spatial)")
    ax.set_ylabel("J_to (temporal)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig

def plot_pareto_curves(fronts, labels=None, title="Pareto curves"):
    # fronts: list of arrays Nx2
    fig, ax = plt.subplots(figsize=(7,5))
    for i, f in enumerate(fronts):
        if f is None or len(f)==0:
            continue
        ax.plot(f[:,0], f[:,1], '-o', label=f"pareto {labels[i] if labels else i}")
    ax.set_xlabel("J_so")
    ax.set_ylabel("J_to")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return fig

def plot_degree_variance(degree_arrays, labels=None, title="Degree variance"):
    """
    degree_arrays: list of 1D arrays (degrees start at 0..N)
    """
    fig, ax = plt.subplots(figsize=(8,5))
    degs = np.arange(len(degree_arrays[0]))
    for arr, lab in zip(degree_arrays, labels):
        ax.plot(degs, arr, label=lab)
    ax.set_yscale('log')
    ax.set_xlabel("Spherical harmonic degree n")
    ax.set_ylabel("Degree variance (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', ls=':')
    return fig
