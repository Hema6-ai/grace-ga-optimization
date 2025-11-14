# GRACE-like Constellation Optimization - interactive surrogate Web App

This repository provides an interactive Streamlit web app that:
- Runs a multi-objective GA (NSGA-II) to search constellation parameter space
- Uses spatial (J_so) and temporal (J_to) surrogate objective calculations (based on paper Deccia et al. 2022)
- Produces Figures 8–14 like outputs (population evolution, Pareto fronts, degree-variance surrogates)
- Interactive controls (altitude, population size, generations, grid resolution, number of pairs, weightings)

**Important**: This is a *surrogate* implementation to reproduce the *workflow and figures* from the paper. Exact paper-level geophysical outputs require NASA GSFC GEODYN & SOLVE and real geophysical model data.

Run locally:
```bash
python -m venv venv
# activate venv (Windows)
venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
