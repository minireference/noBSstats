# Chapter 2 — Probability theory

- Discrete random variables [21_discrete_random_vars.ipynb](./21_discrete_random_vars.ipynb)
- Multiple random variable [22_multiple_random_vars.ipynb](./22_multiple_random_vars.ipynb)
- Inventory of discrete distributions [23_inventory_discrete_dists.ipynb](./23_inventory_discrete_dists.ipynb)
- Calculus prerequisites [24_calculus_prerequisites.ipynb](./24_calculus_prerequisites.ipynb)
- Continuous random variables [25_continuous_random_vars.ipynb](./25_continuous_random_vars.ipynb)
- Inventory of continuous distributions [26_inventory_continuous_dists.ipynb](./26_inventory_continuous_dists.ipynb)
- Probability models for random samples [25_random_samples.ipynb](./25_random_samples.ipynb)




### What is probability theory?

- definition: **probability theory is a language for describing uncertainty, variability, and randomness**

- Originally the study of "randomness" and "expectations" started to quantify random events in gambling.

- Later extended to as a general purpose tool to model any process that contains uncertainty:
  - random variables = described by probability distribution (CDF, pmf/pdf) modelled as a math function with parameters $\theta$
  - noise = can be modelled as a random variable
  - sampling = variations due to random selection of a subset from the population
  - beliefs = can be described as probability distributions


- Probability theory is an essential tool for statistics.

- Probability theory is also a foundational subject that used in physics, machine learning, biology, optimization, algorithms, etc. (side note: in terms of usefulness, I'd say probability theory is up there with linear algebra—you need to know this shit!)


## Probability models

In probability theory, we model data as instances of a **random variable** $X$ described by a **probability distribution** $f_X$ (a math function) with particular parameters (usually denoted with Greek letters like $\theta$, $\mu$, $\sigma$, etc.).

Multiple different ways to specify and interact with probability distributions:
- exact math model function (CDF from which we can extract pdf density function or pmf mass function). Math models allow us the most options: `rvs`, `cdf`, `pdf/pmf`, and stats like `mean`, `median`, `var/std`, `quartiles`.
- random draws form a generative process
- random draws from a real world process
- data for an entire population (census)
- sample data from a population
- synthetic data obtained by resampling
  - bootstrap estimation for any distribution
  - permutation test for no-difference-between-groups hypotheses