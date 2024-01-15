# Statistical analysis examples

This directory contains notebooks with self-contained examples of common statistical analysis techniques.

The purpose is to provide at least one example for each of the test
covered in the [Inventory of statistical test recipes](https://docs.google.com/document/d/1fwep23-95U-w1QMPU31nOvUnUXE2X3s_Dbk5JuLlKAY/edit#heading=h.blivc5m8tn2d).


### List of recipes

- Z-tests:
  - One sample $z$-test: [`one_sample_z-test.ipynb`](./one_sample_z-test.ipynb)

- Proportion tests
  - One-sample $z$-test for proportions
  - Binomial test
  - Two-sample $z$-test for proportions

- T-tests
  - One-sample $t$-test: [`one_sample_t-test.ipynb`](./one_sample_t-test.ipynb)
  - Welch's two-sample $t$-test: 
  - Two-sample $t$-test with pooled variance (not important)
  - Paired $t$-test

- Chi-square tests
  - Chi-square test for goodness of fit
  - Chi-square test of independence
  - Chi-square test for homogeneity
  - Chi-square test for the population variance

- ANOVA tests
  - One-way analysis of variance (ANOVA): [`ANOVA.ipynb`](./ANOVA.ipynb)
  - Two-way ANOVA

- Nonparametric tests
  - Sign test for the population median
  - One-sample Wilcoxon signed-rank test
  - Mann-Whitney U-test
  - Kruskal–Wallis analysis of variance by ranks

- Resampling methods
  - Simulation tests
  - Two-sample permutation test
  - Permutation ANOVA

- Miscellaneous tests
  - Equivalence tests
  - Kolmogorov–Smirnov test
  - Shapiro–Wilk normality test


### Template

For each statistical testing recipe, the notebook follows the same structure:

- Data
- Assumptions
- Hypotheses
- Power calculations 
- Test statistic
- Sampling distribution
- Examples
  - Example 0: synthetic data when H0 is true
  - Example 1: synthetic data when H0 is false
  - Examples 2...n: other examples
- Effect size estimates
- Related tests
- Discussion
- Links



### Why use synthetic data
We use "fake" data for the examples 0 and 1 in order to illustrate the canonical
data type each statistical test is designed to detect.
This is a good "sanity check" to use for any statistical analysis technique:
before trying on your real-world dataset, try it on synthetic data to make sure
it works as expected (is able to detect a difference when a difference exists,
and correctly fails to reject H0 when no difference exists).


