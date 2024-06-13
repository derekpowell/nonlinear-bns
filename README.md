# (Nonlinear) Bayesian Networks for Cognitive Modeling

This repository contains a few test notebooks for non-linear Bayesian network modeling. The overall idea is a two-step process:

1. Conduct (nonlinear) Bayesian Network Structure learning to fit a *Data Model* of participants' survey responses (continuous responses)
2. Transorm the *Data Model* into a *Cognitive Model* of the propositions measured in the survey (binary states-of-affairs)

From there, we can use the *cognitive model* to predict how beliefs will change following interventions.

The structure of the initial *data model* can also be used to better understand people's thinking about the domain, by appreciating which beliefs are directly connected with one another.

---

The project makes use of a few packages:
- `Dagma`: for structure learning (also optionally, `Dibs`)
- `PGMPY`: for BN plotting and inference
- and notably on the backend/under the hood:
    - `numpy`, `scipy`, `sklearn`
    - `torch`
    - `pandas`