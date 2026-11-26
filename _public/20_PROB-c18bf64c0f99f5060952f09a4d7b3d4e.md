# Chapter 2 — Probability theory

Probability theory is a language for describing randomness, variability, and uncertainty.
You need to know probability theory to understand statistics.
This is why Part 1 of the book includes ~200 pages of formulas,
explanations, figures, and code examples
to get you up to speed on all the probability theory topics that you need to know.

Below,
you'll find the computational notebooks from Chapter 2 of the book.
Going through these notebooks would be an interesting even if you don't have the book,
since the notebooks are mostly self contained.

## Notebooks

Each notebook contains the code examples from corresponding section.
If you're reading the book,
you can follow along by running the commands in the these notebooks,
to run all the probability calculations for yourself.

- Discrete random variables [21_discrete_random_vars.ipynb](./21_discrete_random_vars.ipynb)
- Multiple random variables [22_multiple_random_vars.ipynb](./22_multiple_random_vars.ipynb)
- Inventory of discrete distributions [23_inventory_discrete_dists.ipynb](./23_inventory_discrete_dists.ipynb)
- Continuous random variables [24_continuous_random_vars.ipynb](./24_continuous_random_vars.ipynb)
- Multiple continuous random variables [25_multiple_continuous_random_vars.ipynb](./25_multiple_continuous_random_vars.ipynb)
- Inventory of continuous distributions [26_inventory_continuous_dists.ipynb](./26_inventory_continuous_dists.ipynb)
- Random variable generation [27_simulations.ipynb](./27_simulations.ipynb)
- Probability models for random samples [28_random_samples.ipynb](./28_random_samples.ipynb)


## Exercises

Each section contains the exercises to help you practice probability calculations
covered explained in that section.




### Probability models for real world data

Here is a list of the different domains that can be usefully described using probability distributions:
- math models for r.v. $X$ (defined as probability distribution function $f_X$)
- computer models like `rvX` created from one of the model families in `scipy.stats`
  initialized with appropriate parameters.
  The computer model `rvX` for the random variable $X$ has
  methods like: `rvX.rvs()`, `rvX.cdf(b)`, `pdf/pmf`,
  and stats like `rvX.mean()`, `rvX.median()`, `rvX.var()`, `rvX.std()`, `rvQ.ppf(q)`, etc.
- random draws form a generative process (computer simulation that generates random numbers, see [Section 2.7](./27_simulations.ipynb) for examples)
- random draws from a real world process (e.g. factory that produces a new item)
- data for an entire population (census)
- sample data from a population (the data type we learned about in Chapter 1)
- synthetic data obtained by resampling:
  - bootstrap samples = sampling from the empirical distribution
  - permutation test that forget group membership


