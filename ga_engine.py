# ga_engine.py
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from objectives import compute_Jso, compute_Jto_from_TV

# we will represent a single constellation as a flat vector of length 3 * n_pairs
# for each pair: [incl (0..180), raan (0..360), M (0..360)]

class ConstellationProblem(Problem):
    def __init__(self, n_pairs=6, m_cells=4551, T_days=29, weights=None):
        n_var = 3 * n_pairs
        xl = np.tile([0.0, 0.0, 0.0], n_pairs)
        xu = np.tile([180.0, 360.0, 360.0], n_pairs)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, elementwise_evaluation=True)
        self.n_pairs = n_pairs
        self.m_cells = m_cells
        self.T_days = T_days
        self.weights = weights

    def _evaluate(self, x, out, *args, **kwargs):
        # decode x into constellation
        constel = []
        for i in range(self.n_pairs):
            incl = float(x[3*i+0])
            raan = float(x[3*i+1])
            M = float(x[3*i+2])
            constel.append({'incl': incl, 'raan': raan, 'M': M})
        # compute objectives
        so = compute_Jso(constel, m_cells=self.m_cells, weights=self.weights, T_days=self.T_days)
        jt = compute_Jto_from_TV(so['TV'])
        # objective: minimize both Jso and Jto
        out["F"] = np.column_stack([so['Jso'], jt['Jto']])
        # attach diagnostics for later retrieval
        out["G"] = np.array([so['Job'], so['Jro'], so['Jew'], so['Jns'], jt['G_TV']])
        out["misc"] = (constel, so, jt)

def run_nsga(n_pairs=6, pop_size=80, n_gen=20, m_cells=4551, T_days=29, weights=None, seed=1):
    prob = ConstellationProblem(n_pairs=n_pairs, m_cells=m_cells, T_days=T_days, weights=weights)
    alg = NSGA2(pop_size=pop_size,
                sampling=get_sampling("real_random"),
                crossover=get_crossover("real_sbx", prob=0.9, eta=15),
                mutation=get_mutation("real_pm", eta=20),
                eliminate_duplicates=True)
    res = minimize(prob, alg, ('n_gen', n_gen), seed=seed, verbose=False)
    # collect pop histories (res.history exists in newer pymoo versions)
    # For plotting generation 1,3,20 we'll run smaller separate batches to record
    return res
