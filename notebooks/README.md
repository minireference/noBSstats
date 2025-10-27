Computational notebooks
=======================

This folder contains the computational notebooks
that accompany the the **No Bullshit Guide to Statistics**.

The purpose of these notebooks is to allow you
to experiment and play with all the code examples presented in the book.
Each notebook contains numerous empty code cells,
which are an invitation for you to try some commands on your own.
For example, you can copy-paste some of the neighbouring commands
and try modifying them to see what outputs you get.




## Chapter 1: Data

Data is 



### 1.1 Introduction to data
This is a very short notebook that gives some examples of random selection and random assignment.
See the text for the actual definitions of core ideas.

- View notebook: [11_intro_to_data.ipynb](./11_intro_to_data.ipynb)
- Binder link: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F11_intro_to_data.ipynb)
- Colab link:
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/11_intro_to_data.ipynb)


### 1.2 Data in practice
This notebook explains practical aspects of data manipulations using Python library [Pandas](https://pandas.pydata.org/),
including data preprocessing steps like data cleaning and outlier removal.
The notebook also introduces basic plots using the library [Seaborn](https://seaborn.pydata.org/).

- View notebook: [12_data_in_practice.ipynb](./12_data_in_practice.ipynb)
- Binder link: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F12_data_in_practice.ipynb)
- Colab link: TODO



### 1.3 Descriptive statistics
This notebook explains how to compute numerical summaries (mean, standard deviation, quartiles, etc.)
and how to generate data statistical plots like histograms, box plots, bar plots, etc.

- View notebook: [13_descriptive_statistics.ipynb](./13_descriptive_statistics.ipynb)
- Binder link: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F13_descriptive_statistics.ipynb)
- Colab link:
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/13_descriptive_statistics.ipynb)









&nbsp;
<hr>


## Chapter 2: Probability

In this chapter you'll learn about random variables and probability distributions,
which are essential building blocks for statistical models.

### 2.1 Discrete random variables
This notebook contains a complete introduction to probability theory,
including definitions, formulas, and lots of examples of discrete
random variables like coin toss, die roll, and other.

- View notebook: [21_discrete_random_vars.ipynb](./21_discrete_random_vars.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F21_discrete_random_vars.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/21_discrete_random_vars.ipynb)


### 2.2 Multiple random variables
This section will introduce you to the concept of a joint probability distribution.
For example, the pair of random variables $(X,Y)$ can be described by the joint probability distribution function $f_{XY}$.

- View notebook: [22_multiple_random_vars.ipynb](./22_multiple_random_vars.ipynb)  
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F22_multiple_random_vars.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/22_multiple_random_vars.ipynb)


### 2.3 Inventory of discrete distributions
The Python module `scipy.stats` contains pre-defined probability models that you
can use for modeling tasks. These are like LEGOs for the XXIst century.

- View notebook: [23_inventory_discrete_dists.ipynb](./23_inventory_discrete_dists.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F23_inventory_discrete_dists.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/23_inventory_discrete_dists.ipynb)



### 2.4 Continuous random variables
In this notebook we'll revisit all the probability concepts we learned for discrete
random variables, and learn the analogous concepts for continuous random variables.
You can think of Section 2.5 as the result of taking Section 2.1
and replacing every occurrence $\textrm{Pr}(a \leq X \leq b)=\sum_{x=a}^{x=b}f_X(x)$
with $\textrm{Pr}(a \leq X \leq b)=\int_{x=a}^{x=b}f_X(x)dx$.

- View notebook: [24_continuous_random_vars.ipynb](./24_continuous_random_vars.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F24_continuous_random_vars.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/24_continuous_random_vars.ipynb)


### 2.5 Multiple continuous random variables
We study of pairs of continuous random variables and their *joint probability density functions*.
We'll revisit some of the ideas from Section 2.2 but apply them to continuous random variables.

- View notebook: [25_multiple_continuous_random_vars.ipynb](./25_multiple_continuous_random_vars.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F25_multiple_continuous_random_vars.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/25_multiple_continuous_random_vars.ipynb)



### 2.6 Inventory of continuous distributions
In this section we'll complete the inventory of probability distributions by
introducing the "continuous LEGOs" distributions like `uniform`, `norm`, `expon`,
`t`, `f`, `chi2`, `gamma`, `beta`, etc.

- View notebook: [26_inventory_continuous_dists.ipynb](./26_inventory_continuous_dists.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F26_inventory_continuous_dists.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/26_inventory_continuous_dists.ipynb)



### 2.7 Simulations
How can we use computers to generation observations from random variables?
In this notebooks, we'll describe some practical techniques for generating
observations from any probability distribution.
We'll learn how to use lists of observations from random variables for probability calculations.

- View notebook: [27_simulations.ipynb](./27_simulations.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F27_simulations.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/27_simulations.ipynb)



### 2.8 Probability models for random samples
Consider a random variable $X$ with a known probability distribution $f_X$.
What can we say about the characteristics of $n$ copies of the random variable
$\mathbf{X} = (X_1, X_2, X_3, \cdots, X_n) \sim f_{X_1X_2\cdots X_n}$.
Each $X_i$ is independent copy of the random variable $X$.
This is called the *independent, identically distributed* (iid) setting.
Understanding the properties of the *random sample* $\mathbf{X}$ is important for all the 
statistics operations we'll be doing in Chapter 3.

- View notebook: [28_random_samples.ipynb](./28_random_samples.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F28_random_samples.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/28_random_samples.ipynb)








&nbsp;
<hr>


## Chapter 3: Classical (frequentist) statistics
This chapter introduces the main ideas of STAT101.

### 3.1 Estimators
Estimators are the function that we use to compute estimates.
The value of the estimator computed from a random sample $\mathbf{X}$
is called the *sampling distribution* of the estimator.
Understanding estimators and sampling distributions is the key step to understanding the statistics topics in Chapter 3.

- View notebook: [31_estimators.ipynb](./31_estimators.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F31_estimators.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/31_estimators.ipynb)


### 3.2 Confidence intervals
A *confidence interval* describes a lower bound and an upper bound on the possible values for some unknown parameter.
We'll learn how to construct confidence intervals based on what we learned about sampling distributions in Section 3.1.

- View notebook: [32_confidence_intervals.ipynb](./32_confidence_intervals.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F32_confidence_intervals.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/32_confidence_intervals.ipynb)


### 3.3 Introduction to hypothesis testing
Hypothesis testing is a statistical analysis technique we can use to detect "unexpected" patterns in data.
We start our study of hypothesis testing using direct simulation.

- View notebook: [33_intro_to_NHST.ipynb](./33_intro_to_NHST.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F33_intro_to_NHST.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/33_intro_to_NHST.ipynb)

### 3.4 Hypothesis testing using analytical approximations
We revisit the hypothesis testing procedures using analytical formulas.

- View notebook: [34_analytical_approx.ipynb](./34_analytical_approx.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F34_analytical_approx.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/34_analytical_approx.ipynb)

### 3.5 Two-sample hypothesis tests
In this notebook,
we focus on the specific problem of comparing samples from two unknown populations.

- View notebook: [35_two_sample_tests.ipynb](./35_two_sample_tests.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F35_two_sample_tests.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/35_two_sample_tests.ipynb)

### 3.6 Statistical design and error analysis
This section describes the possible mistakes we can make when using the hypothesis testing procedure.
The *statistical design* process includes considerations we take before the study to ensure the hypothesis test we perform has the correct error rates.

- View notebook: [36_design.ipynb](./36_design.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F36_design.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/36_design.ipynb)

### 3.7 Inventory of statistical tests
This notebook contains examples of the statistical testing "recipes" for use in different data analysis scenarios.

- View notebook: [37_inventory_stats_tests.ipynb](./37_inventory_stats_tests.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F37_inventory_stats_tests.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/37_inventory_stats_tests.ipynb)


### 3.8 Statistical practice
No notebook available. See the text or the [preview](https://minireference.com/static/excerpts/noBSstats_part2_preview.pdf#page=92).





&nbsp;
<hr>


## Chapter 4: Linear models

Linear models are a unifying framework for studying the influence of predictor variables $x_1, x_2, \ldots, x_p$ on an outcome variable $y$.

### 4.1 Simple linear regression
The simplest linear model has one predictor $x$ and one outcome variabel $y$.

- View notebook: [41_simple_linear_regression.ipynb](./41_simple_linear_regression.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F41_simple_linear_regression.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/41_simple_linear_regression.ipynb)

### 4.2 Multiple linear regression
We extend the study to a linear model with $p$ predictors $x_1, x_2, \ldots, x_p$.

- View notebook: [42_multiple_linear_regression.ipynb](./42_multiple_linear_regression.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F42_multiple_linear_regression.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/42_multiple_linear_regression.ipynb)

### 4.3 Interpreting linear models
We learn how to interpret the parameter estimates and perform hypothesis tests based on linear models.

- View notebook: [43_interpreting_linear_models.ipynb](./43_interpreting_linear_models.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F43_interpreting_linear_models.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/43_interpreting_linear_models.ipynb)

### 4.4 Regression with categorical predictors
We can encode categorical predictor variables using the "dummy coding" that involves indicator variables.
We'll use dummy coding to revisit some of the statistical analyses we saw in Chapter 3.

- View notebook: [44_regression_with_categorical_predictors.ipynb](./44_regression_with_categorical_predictors.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F44_regression_with_categorical_predictors.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/44_regression_with_categorical_predictors.ipynb)

### 4.5 Model selection for causal inference
We often want to learn about causal links between variables,
which is difficult when all we have is observational data.
In this section,
we'll use linear models as a platform to learn about confounding variables
and other obstacles to causal inference.

- View notebook: [45_causal_inference.ipynb](./45_causal_inference.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F45_causal_inference.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/45_causal_inference.ipynb)

### 4.6 Generalized linear models
In this section,
we'll use the structure of linear models to build models for binary outcome variables (logistic regression)
or count data (Poisson regression).

- View notebook: [46_generalized_linear_models.ipynb](./46_generalized_linear_models.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F46_generalized_linear_models.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/46_generalized_linear_models.ipynb)





&nbsp;
<hr>


## Chapter 5: Bayesian statistics

Bayesian statistics is a whole other approach to statistical inference.


### 5.1 Introduction to Bayesian statistics
This section introduces the basic ideas from first principles,
using simple hands-on calculations based on the *grid approximation*.

- View notebook: [51_intro_to_Bayesian_stats.ipynb](./51_intro_to_Bayesian_stats.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F51_intro_to_Bayesian_stats.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/51_intro_to_Bayesian_stats.ipynb)

### 5.2 Bayesian inference computations
In this section,
we revisit the Bayesian inference calculations we saw in Section 5.1,
but this time use the [Bambi](https://bambinos.github.io/bambi/) library to take care of the calculations for us.
We'll also use the [ArViZ](https://python.arviz.org/) to help with visualization of Bayesian inference results. 

- View notebook: [52_Bayesian_inference_computations.ipynb](./52_Bayesian_inference_computations.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F52_Bayesian_inference_computations.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/52_Bayesian_inference_computations.ipynb)

### 5.3 Bayesian linear models
We use Bambi to build linear models and revisit some of the analyses from Chapter 4 from a Bayesian perspective.

- View notebook: [53_Bayesian_linear_models.ipynb](./53_Bayesian_linear_models.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F53_Bayesian_linear_models.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/53_Bayesian_linear_models.ipynb)

### 5.4 Bayesian difference between means
We build a custom model for comparing two unknown populations based on Bayesian principles.
We use this model to revisit some of the statistical analyses from Section 3.5 from a Bayesian perspective.

- View notebook: [54_difference_between_means.ipynb](./54_difference_between_means.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F54_difference_between_means.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/54_difference_between_means.ipynb)

### 5.5 Hierarchical models
Certain datasets have a group structure.
For example students' data often comes in groups according to the class they are part of.
Hierarchical models (a.k.a. multilevel models) are specially designed to handle data with group structure.
 
- View notebook: [55_hierarchical_models.ipynb](./55_hierarchical_models.ipynb)
- Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstats/main?labpath=notebooks%2F55_hierarchical_models.ipynb)
- Colab link: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minireference/noBSstats/blob/main/notebooks/55_hierarchical_models.ipynb)
```

