

## What is probability theory?

Probability theory is a language for describing measurement noise,
variability in observations, and randomness in outcomes.

- Originally the study of "randomness" and "expectations" started to quantify random events in gambling.
- Later extended to as a general purpose tool to model any process that contains uncertainty:
  - random variables = described by probability distribution (CDF, pmf/pdf) modelled as a math function with parameters $\theta$
  - noise = can be modelled as a random variable = uncertainty in measurement
  - sampling = variations due to random selection of a subset from the population
  - uncertainty in knowledge = can be described as probability distributions  (ALT. beliefs)
- Probability theory is an essential tool for statistics.
- Probability theory is also a foundational subject that used in physics, machine learning,
  biology, optimization, algorithms, etc. (side note: in terms of usefulness,
  I'd say probability theory is up there with linear algebra—you need to know this shit!)



### Random variables

In probability theory, a **random variable** $X$ is a tool for describing variability of some quantity.
Unlike a regular variable $x$ that is a placeholder for a single value,
the random variable $X$ can have different values *outcomes* every time it is observed.
The random variables $X$ is described by a **probability distribution** $f_X$
(a function that assigns probabilities to different possible outcomes of the random variable $X$).
The probability distribution (probability model) $f_X$ is usually described
as $f_X = \mathcal{M}(\theta)$, where $\mathcal{M}$ is a probability model family (e.g. uniform, normal, exponential, Poisson, etc.), and $\theta$ are the model parameters.
e use capital letters like $X,Y,Z$ for random variables,
lowercase letters for like $x,y,z$ for particular values,
and Greek letters like $\theta,\alpha,\mu,\sigma$ for model parameters.


