# (Nonlinear) Bayesian Networks for Cognitive Modeling

This repository contains a few test notebooks for non-linear Bayesian network modeling. The overall idea is a two-step process:

1. Conduct (nonlinear) Bayesian Network Structure learning to fit a *Data Model* of participants' survey responses (continuous responses)
2. Transorm the *Data Model* into a *Cognitive Model* of the propositions measured in the survey (binary states-of-affairs)

From there, we can use the *cognitive model* to predict how beliefs will change following interventions.

The structure of the initial *data model* can also be used to better understand people's thinking about the domain, by appreciating which beliefs are directly connected with one another.

## Notebooks and code
- `funcs.py` implments some custom functions
- `dagma-cv-vacc.ipynb` implements an example of cross validation to select model fitting hyperparameters. From this, a best approach can be selected and a final model fit. Also includes some code for plotting
- `dagma-vacc.ipynb` initial experimental notebook, a rough draft more-or-less
- `dibs_joint.ipynb` an experimental notebook applying the `Dibs` package instead of Dagma

---

The project makes use of a few packages:
- `Dagma`: for structure learning (also optionally, `Dibs`)
- `PGMPY`: for BN plotting and inference
- and notably on the backend/under the hood:
    - `numpy`, `scipy`, `sklearn`
    - `torch`
    - `pandas`