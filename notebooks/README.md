Notebooks
=========

This folder contains the notebooks created for the **No Bullshit Guide to Statistics**.

The purpose of these notebooks is to allow you
to experiment and play with all the code examples presented in the book.
Each notebook contains numerous empty code cells,
which are an invitation for you to try some commands on your own.
For example, you can copy-paste some of the neighbouring commands
and try modifying them to see what outputs you get.




## Chapter 1: Data

### 1.1 Introduction to data
This is a very short notebook that gives some examples of random selection and random assignment.

- View notebook: [11_intro_to_data.ipynb](./11_intro_to_data.ipynb)
- Binder link: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main?labpath=notebooks%2F11_intro_to_data.ipynb)
- Colab link:
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstatsnotebooks/blob/main/notebooks/11_intro_to_data.ipynb)


### 1.2 Data in practice
This notebook explains practical aspects of data manipulations using Pandas
and talks about data pre-processing steps like data cleaning and outlier removal.

- View notebook: [12_data_in_practice.ipynb](./12_data_in_practice.ipynb)
- Binder link: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main?labpath=notebooks%2F12_data_in_practice.ipynb)
- Colab link: TODO

TODO: finish the notebook with Alice, Bob, and Charlotte pre-processing steps:
[12b_background_stories.ipynb](./12b_background_stories.ipynb).



### 1.3 Descriptive statistics
This notebook explains how to compute numerical summaries (mean, standard deviation, quartiles, etc.)
and how to generate data visualizations like histograms, box plots, bar plots, etc.

- View notebook: [13_descriptive_statistics.ipynb](./13_descriptive_statistics.ipynb)
- Binder link: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main?labpath=notebooks%2F13_descriptive_statistics.ipynb)
- Colab link:
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstatsnotebooks/blob/main/notebooks/13_descriptive_statistics.ipynb)











## Chapter 2: Probability

In this chapter you'll learn about random variables and probability models.


### 2.1 Discrete random variables
This notebook contains a complete introduction to probability theory,
including definitions, formulas, and lots of examples of discrete
random variables like coin toss, die roll, and other.

- View notebook: [21_discrete_random_vars.ipynb](./21_discrete_random_vars.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main?labpath=notebooks%2F21_discrete_random_vars.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstatsnotebooks/blob/main/notebooks/21_discrete_random_vars.ipynb)


### 2.2 Multiple random variables
This section will introduce you to the concept of a joint probability distribution.
For example, the pair of random variables $(X,Y)$ can be described by the joint probability distribution function $f_{XY}$.

- View notebook: [22_multiple_random_vars.ipynb](./22_multiple_random_vars.ipynb)  
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main?labpath=notebooks%2F22_multiple_random_vars.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstatsnotebooks/blob/main/notebooks/22_multiple_random_vars.ipynb)


### 2.3 Inventory of discrete distributions
The Python module `scipy.stats` contains pre-defined probability models that you
can use for modeling tasks. These are like LEGOs for the XXIst century.

- View notebook: [23_inventory_discrete_dists.ipynb](./23_inventory_discrete_dists.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main?labpath=notebooks%2F23_inventory_discrete_dists.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstatsnotebooks/blob/main/notebooks/23_inventory_discrete_dists.ipynb)



### 2.4 Calculus prerequisites
You need to know a bit of calculus to understand the math machinery
for calculating probabilities of continuous random variables.
Don't worry—there is only one new concept to learn: the integral $\int_{x=a}^{x=b} f(x)dx$,
which corresponds to computing the area under the graph of $f(x)$ between $x=a$ and $x=b$.

- View notebook: [24_calculus_prerequisites.ipynb](./24_calculus_prerequisites.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main?labpath=notebooks%2F24_calculus_prerequisites.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstatsnotebooks/blob/main/notebooks/24_calculus_prerequisites.ipynb)


### 2.5 Continuous random variables
In this notebook we'll revisit all the probability concepts we learned for discrete
random variables, and learn the analogous concepts for continuous random variables.
You can think of Section 2.5 as the result of taking Section 2.1
and replacing every occurrence $\textrm{Pr}(a \leq X \leq b)=\sum_{x=a}^{x=b}f_X(x)$
with $\textrm{Pr}(a \leq X \leq b)=\int_{x=a}^{x=b}f_X(x)dx$.

- View notebook: [25_continuous_random_vars.ipynb](./25_continuous_random_vars.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main?labpath=notebooks%2F25_continuous_random_vars.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstatsnotebooks/blob/main/notebooks/25_continuous_random_vars.ipynb)


### 2.6 Inventory of continuous distributions
In this section we'll complete the inventory of probability distributions by
introducing the "continuous LEGOs" distributions like `uniform`, `norm`, `expon`,
`t`, `f`, `chi2`, `gamma`, `beta`, etc.

- View notebook: [26_inventory_continuous_dists.ipynb](./26_inventory_continuous_dists.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main?labpath=notebooks%2F26_inventory_continuous_dists.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstatsnotebooks/blob/main/notebooks/26_inventory_continuous_dists.ipynb)



### 2.7 Random variable generation
How can we use computers to generation observations from random variables?
In this notebooks, we'll describe some practical techniques for generating
observations from any probability distribution, and develop math tools to verify
that the random generation process is workin as expected.

- View notebook: [27_random_var_generation.ipynb](./27_random_var_generation.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main?labpath=notebooks%2F27_random_var_generation.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstatsnotebooks/blob/main/notebooks/27_random_var_generation.ipynb)



### 2.8 Probability models for random samples
Consider a random variable $X$ with a known probability distribution $f_X$.
What can we say about the characteristics of $n$ copies of the random variable
$\mathbf{X} = X_1X_2X_3\cdots X_n \sim f_{X_1X_2\cdots X_n}$.
Each $X_i$ is independent copy of the random variable $X$.
This is called the independent, identically distributed (iid) setting.
Understanding the properties of $\mathbf{X}$ is important for all the 
statistics operations we'll be doing in the next two chapters.

- View notebook: [28_random_samples.ipynb](./28_random_samples.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main?labpath=notebooks%2F28_random_samples.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstatsnotebooks/blob/main/notebooks/28_random_samples.ipynb)












## Chapter 3: Inferential statistics

(coming late 2022 or early 2023)

In the meantime,
you check out the [Chapter 3 outline](https://docs.google.com/document/d/1fwep23-95U-w1QMPU31nOvUnUXE2X3s_Dbk5JuLlKAY/edit#heading=h.w1m7v7b5wie3) to see what will be covered.












## Chapter 4: Linear models

(coming late 2022 or early 2023)

In the meantime,
you check out the [Chapter 4 outline](https://docs.google.com/document/d/1fwep23-95U-w1QMPU31nOvUnUXE2X3s_Dbk5JuLlKAY/edit#heading=h.9etj7aw4al9w) to see what will be covered.



____


## Math and code conventions used in notebooks

Datasets:

- `eprices`: the dataframe loaded from `datasets/eprices.csv`
- `eprices{W}`: the subset of the rows from dataframe for group `{W}`
- `x{A}`: data from group `{A}` (notation chosen to match math symbol $\mathbf{x}_A$)


Sampling and resampling conventions:

- `{x}sample`: generated sample from `rvX` by simulation.
  Optionally include sample size, e.g. `{x}sample20` when $n=20$.
  - `{x}samples_df`: dataframe that contains `N` `{x}samples`
- `rsample`: sample generated using resampling (e.g. permutation test) 
  - `rsamples_df`: a DataFrame that contains data from `R` `rsample`s
    and has an extra column `rep=1:n` (to imitate `replicate` col used by `infer`).
- `bsample`: bootstrap sample generated from `sample`
  - `bsamples_df`: a DataFrame that contains data from `B` `bsample`s

