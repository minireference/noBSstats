Software design
===============

In order for this book to be successful the code examples must be solid.
Statistics is a complicated topic, and for readers who have no prior experience programming
the complexity of statistics will be compounded by the complexity of learning Python syntax.

I want the Python code examples to be simple and ideally self-explanatory.
I want the reader to think "yes, this makes sense" when they see each code block,
and support the text narrative, rather than see the code examples as additional stumbling blocks.

Goals:

- Keep it pythonic: follow standard conventions for naming an source formatting.
- Make it general enough, but not too general:
  helper functions should be able to handle all the cases for the book,
  but not generalize beyound what is necessary.
- Make it simple: don't use fancy Python features that beginners might not be familiar with.




## Naming conventions

### General

- Do not reuse variable names in the same notebook (e.g. if you define `rvX` in
  the beginning, don't redefine it to be a different thing later on).



### Datasets

- `eprices` (df): the dataframe loaded from `datasets/eprices.csv`
- `eprices{W}` (df): the subset of the rows from data frame for group `{W}`
  - `pricesW` and `pricesE` (array or series): of the `price` data column 
- `ksample`: kombucha volumes sample, also `ksample02`, `ksample03`, from different batches
- `asample`: apples weights 
- `scoresR` and `scoresU`: the values of the two groups of sleep scores form the `doctors` dataset


Sampling distributions:

- `xbars20`: sampling distribution of the mean for samples of size `n=20` from the random variable `rvX`.
- `xvars20`: sampling distribution of the variance for samples of size `n=20` from  `rvX`.


Sampling and resampling conventions:
- `{x}sample`: generated sample from `rvX` by simulation.  
  Optionally include sample size, e.g. `{x}sample20` for sample of size $n=20$.
- `rsample`: sample generated using resampling (e.g. permutation test) 
- `bsample`: bootstrap sample generated from `sample`


Data frames full of samples or resamples (not currently used in user-facing code):
- `{x}samples_df`: dataframe that contains multiple`{x}samples` (identified by value in `sample` column)
- `rsamples_df`: a DataFrame that contains data from `R` `rsample`s
  and has an extra column `rep=1:n` (to imitate `replicate` col used by `infer`).
- `bsamples_df`: a DataFrame that contains data from `B` `bsample`s





## Code structure

### Sampling distribution generator (known RV)

See function `gen_sampling_dist` in [`ministats/sampling.py`](https://github.com/minireference/ministats/blob/main/ministats/sampling.py).





### Bootstrap sampling distribution

See function `gen_boot_dist` in [`ministats/sampling.py`](https://github.com/minireference/ministats/blob/main/ministats/sampling.py).



### Simulation-based hypothesis test

See functions `simulation_test_mean` in [`ministats/hypothesis_tests.py`](https://github.com/minireference/ministats/blob/main/ministats/hypothesis_tests.py).



### Permutation test

See functions `resample_under_H0` and `permutation_test` in [`ministats/hypothesis_tests.py`](https://github.com/minireference/ministats/blob/main/ministats/hypothesis_tests.py).


Below we save the old implementations (which were less good).


##### Approach 1: from stats overview (Nov 2021)

```Python
def dmeans(sample):
    """
    Compute the difference between groups means.
    """
    xS = sample[sample["group"]=="S"]["ELV"]
    xNS = sample[sample["group"]=="NS"]["ELV"]
    d = np.mean(xS) - np.mean(xNS)
    return d

def resample_under_H0(sample, groupcol="group"):
    """
    Return a copy of the dataframe `sample` with the labels in the column `groupcol`
    modified based on a random permutation of the values in the original sample.
    """
    resample = sample.copy()
    labels = sample[groupcol].values
    newlabels = np.random.permutation(labels)
    resample[groupcol] = newlabels
    return resample

# # resample
# resample = resample_under_H0(data)

# # compute the difference in means for the new labels
# dmeans(resample)

def permutation_test(sample, statfunc, groupcol="group", permutations=10000):
    """
    Compute the p-value of the observed `statfunc(sample)` under the null hypothesis
    where the labels in the `groupcol` are randomized.
    """
    # 1. compute the observed value of the statistic for the sample
    obsstat = statfunc(sample)

    # 2. generate the sampling distr. under H0
    restats = []
    for i in range(0, permutations):
        resample = resample_under_H0(sample, groupcol=groupcol)
        restat = statfunc(resample)
        restats.append(restat)

    # 3. compute p-value: how many `restat`s are equal-or-more-extreme than `obsstat`
    tailstats = [restat for restat in restats \
                 if restat <= -abs(obsstat) or restat >= abs(obsstat)]
    pvalue = len(tailstats) / len(restats)

    return restats, pvalue

# sampling_dist, pvalue = permutation_test(data, statfunc=dmeans)
```

Comments:

- The function `dmeans` is not the same as `dmeans` used earlier in the book.
- The `for`-loop in `permutation_test` could be abstracted into a function.




#### Approach 2: attempted refactoring (Dec 2022)

```Python
def resample_under_H0(data, groupcol="group"):
    """
    Return a copy of the dataframe `data` with
    the labels in the column `groupcol` mixed up.
    """
    redata = data.copy()
    labels = data[groupcol].values
    newlabels = np.random.permutation(labels)
    redata[groupcol] = newlabels    
    return redata

def gen_sampling_dist_under_H0(data, permutations=10000):
    """
    Obtain the sampling distribution of `dmeans` under H0
    by repeatedly permutations of the group labels.
    """
    pstats = []
    for i in range(0, permutations):
        data_perm = resample_under_H0(data, groupcol="end")
        xW_perm = data_perm[data_perm["end"]=="West"]["price"]
        xE_perm = data_perm[data_perm["end"]=="East"]["price"]
        dhat_perm = dmeans(xW_perm, xE_perm)
        pstats.append(dhat_perm)
    return pstats

def permutation_test(data):
    """
    Compute the p-value of the observed `dmeans(xW,xE)` under H0.
    """
    # Obtain the sampling distribution of `dmeans` under H0
    pstats = gen_sampling_dist_under_H0(data)

    # Compute the value of `dmeans` for the observed data
    xW = eprices[eprices["end"]=="West"]["price"]
    xE = eprices[eprices["end"]=="East"]["price"]
    dhat = dmeans(xW, xE)

    # Compute p-value of `dhat` under the distribution `pstats` 
    # (how many of `pstats` are equal-or-more-extreme than `dhat`)
    tailstats = [pstat for pstat in pstats \
                 if pstat <= -abs(dhat) or pstat >= abs(dhat)]
    pvalue = len(tailstats) / len(pstats)
    return pvalue

```

Comments:

- Seems awkward for `gen_sampling_dist_under_H0` and `permutation_test` know
  about the specifics of the problem.




#### Approach 3: `infer`-like verbs

It would be nice to imitate the [infer API](https://infer.tidymodels.org/reference/index.html)
and [process](https://raw.githubusercontent.com/tidymodels/infer/main/figs/ht-diagram.png)
so that we don't have to make design choices:

1. `specify()` or `assume()`
2. `hypothesize()`
3. `generate()`
4. `get_p_value()`
5. `get_confidence_interval()`

I don't really see how to do this yet, but I'm watching the authors of
[UBC-DSCI/introduction-to-datascience-python](https://github.com/UBC-DSCI/introduction-to-datascience-python/blob/main/source/inference.md#bootstrapping-in-python) to see what Python code they will use to imitate infer functionality.



#### Approach 4: numpy arrays (winner!)

1. Extract the data from dataframe when passing to function
2. Combine into an long array; permute; return first `n` and remaining `m`
3. Can use ordinary functions `dmeans` as previously shown.

```Python
def resample_under_H0(sample1, sample2):
    """
    Generate new samples from a random permutation of the values in two samples.
    """
    values = np.concatenate((sample1, sample2))
    shuffled_values = np.random.permutation(values)
    resample1 = shuffled_values[0:len(sample1)]
    resample2 = shuffled_values[len(sample1):]
    return resample1, resample2

# usage:
# >>> resample_under_H0([1,1,1], [2,2,2,2])
# (array([2, 2, 1]), array([2, 1, 1, 2]))
```

```Python
def permutation_test(sample1, sample2, statfunc, permutations=10000):
    """
    Compute the p-value of the observed `statfunc(sample1, sample2)` under
    the null hypothesis where the group membership is randomized.
    """
    # 1. compute the observed value of the statistic for the sample
    obsstat = statfunc(sample1, sample2)

    # 2. Obtain the sampling distribution of `statfunc` under H0
    pstats = []
    for i in range(0, permutations):
        resample1, resample2 = resample_under_H0(sample1, sample2)
        pstat = statfunc(resample1, resample2)
        pstats.append(pstat)

    # 3. compute p-value: how many `pstat`s are equal-or-more-extreme than `obsstat`
    tailstats = [pstat for pstat in pstats \
                 if pstat <= -abs(obsstat) or pstat >= abs(obsstat)]
    pvalue = len(tailstats) / len(pstats)

    return pstats, pvalue

# usage:
# >>> permutation_test(pricesW, pricesE, statfunc=dmeans)[1]
# 0.0003
```

Comments:

- Pretty tight... 
- Only issue is the `[0:len(sample1)]` selection which is weird,
  but we can live with that.
