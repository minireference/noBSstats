# Chapter 2 — Probability theory

Probability theory is a language for describing uncertainty, variability, and randomness.
You need to know probability theory to understand statistics.
This is why the book includes ~200 pages of formulas, explanations, figures, and code examples to
get you up to speed on all the probability theory topics that you need to know.


## Notebooks

Each notebook contains the code examples from corresponding section in book.
If you're reading the book, you should follow along by running the commands in the these notebooks,
to check all the probability calculations for yourself.
You don't need to copy-paste the code examples manually—I've already copy-pasted them for you!

- Discrete random variables [21_discrete_random_vars.ipynb](./21_discrete_random_vars.ipynb)
- Multiple random variables [22_multiple_random_vars.ipynb](./22_multiple_random_vars.ipynb)
- Inventory of discrete distributions [23_inventory_discrete_dists.ipynb](./23_inventory_discrete_dists.ipynb)
- Calculus prerequisites [24_calculus_prerequisites.ipynb](./24_calculus_prerequisites.ipynb)
- Continuous random variables [25_continuous_random_vars.ipynb](./25_continuous_random_vars.ipynb)
- Inventory of continuous distributions [26_inventory_continuous_dists.ipynb](./26_inventory_continuous_dists.ipynb)
- Random variable generation [27_random_var_generation.ipynb](./27_random_var_generation.ipynb)
- Probability models for random samples [28_random_samples.ipynb](./28_random_samples.ipynb)

Going through these notebooks would be an interesting even if you don't have the book,
since the notebooks are mostly self contained.
In particular, in combinations with the video tutorials like the [Stats overview](https://www.youtube.com/watch?v=oXy-sZwkn9E&list=PLGmu4KtWiH680gMQnSbSADBuLnoyBUVFg) series,
you could probably learn a good bit of statistics without ever buying the [**No Bullshit Guide to Statistics**](https://nobsstats.com).



## Exercises

Learning the Python code used for doing probability calculations will be very helpful
for solving the exercises and problems in the book.
I've prepared notebooks in the [../notebooks/](../notebooks/) folder,
that you can use to start working on the exercises and problems.



## What is probability theory?

Probability theory is a language for describing uncertainty in measurements,
variability in observations, and randomness in outcomes.

- Originally the study of "randomness" and "expectations" started to quantify random events in gambling.
- Later extended to as a general purpose tool to model any process that contains uncertainty:
  - random variables = described by probability distribution (CDF, pmf/pdf) modelled as a math function with parameters $\theta$
  - noise = can be modelled as a random variable
  - sampling = variations due to random selection of a subset from the population
  - beliefs = can be described as probability distributions
- Probability theory is an essential tool for statistics.
- Probability theory is also a foundational subject that used in physics, machine learning,
  biology, optimization, algorithms, etc. (side note: in terms of usefulness,
  I'd say probability theory is up there with linear algebra—you need to know this shit!)



### Random variables

In probability theory, a **random variable** $X$ is a tool for describing variability of some quantity.
Unlike a regular variable $x$ that is a placeholder for a single value,
the random variable $X$ can have different values *outcomes* every time it is observed.
The random variables $X$ is described by a **probability distribution** $f_X$
(a function that assigns the probabilities to different possible outcomes of the random variable $X$).
The probability distribution (probability model) $f_X$ is usually described
as $f_X = \mathcal{M}(\theta)$, where $\mathcal{M}$ is a probability model family (e.g. uniform, normal, exponential, Poisson, etc.), and $\theta$ are the model parameters (denoted by Greek letters like $\theta$, $\mu$, $\sigma$, etc.).


### Probability models for real world data

The main reason why you might wand to learn probability theory,
is because it provides a unified language for talking about the
uncertainty, variability, and randomness we observe in real-world data.

Here is a short list of the different domains that can be usefully described using probability distributions:
- math models for r.v. $X$ (defined as probability distribution function $f_X$)
- computer models like `rvX` created from one of the model families in `scipy.stats`
  initialized with appropriate parameters.
  The computer model `rvX` for the random variable $X$ has
  methods like: `rvX.rvs()`, `rvX.cdf(b)`, `pdf/pmf`,
  and stats like `rvX.mean()`, `rvX.median()`, `rvX.var()`, `rvX.std()`, `rvQ.ppf(q)`, etc.
- random draws form a generative process (computer simulation that generates random numbers, see [Section 2.7](./27_random_var_generation.ipynb) for examples)
- random draws from a real world process (e.g. factory that produces a new item)
- data for an entire population (census)
- sample data from a population (the data type we learned about in Chapter 1)
- synthetic data obtained by resampling:
  - bootstrap samples = sampling from the empirical distribution
  - permutation test that forget group membership


