import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import t as tdist



# ESTIMATORS
################################################################################

def mean(sample):
    return sum(sample) / len(sample)

def var(sample):
    xbar = mean(sample)
    sumsqdevs = sum([(xi-xbar)**2 for xi in sample])
    return sumsqdevs / (len(sample)-1)

def std(sample):
    s2 = var(sample)
    return np.sqrt(s2)

def dmeans(xsample, ysample):
    dhat = mean(xsample) - mean(ysample)
    return dhat


# UTILS
################################################################################

def calcdf(stdX, n, stdY, m):
    """
    Calculate the degrees of freedom parameter used for Welch's t-test.
    """
    vX = stdX**2 / n
    vY = stdY**2 / m
    df = (vX + vY)**2 / (vX**2/(n-1) + vY**2/(m-1))
    return df



# TAIL CALCULATION UTILS
################################################################################

def tailsof(stats, obs, alternative="two-sided"):    # TODO: rename to tailstats
    """
    Select the subset of the values in `stats` that
    equal or more extreme than the observed value `obs`.
    """
    assert alternative in ["greater", "less", "two-sided"]
    stats = np.array(stats)
    if alternative == "greater":
        tails = stats[stats >= obs]
    elif alternative == "less":
        tails = stats[stats <= obs]
    elif alternative == "two-sided":
        statsmean = np.mean(stats)
        dev = abs(statsmean - obs)
        tails = stats[abs(stats-statsmean) >= dev]
    return tails


def tailprobs(rv, obs, alternative="two-sided"):
    """
    Calculate the probability of all outcomes of the random variable `rv` that
    equally or more extreme than the observed value `obs`.
    """
    assert alternative in ["greater", "less", "two-sided"]
    if alternative == "greater":
        pvalue = 1 - rv.cdf(obs)
    elif alternative == "less":
        pvalue = rv.cdf(obs)
    elif alternative == "two-sided":
        absobs = abs(obs)
        pleft = rv.cdf(-absobs)
        pright = 1 - rv.cdf(absobs)
        pvalue = 2*min(pleft, pright)
    return pvalue




# BASIC TESTS (used for kombucha data generation)
################################################################################

def ztest(sample, mu0, sigma0, alternative="two-sided"):
    """
    Z-test to detect mean deviation from known normal population.
    """
    mean = np.mean(sample)
    n = len(sample)
    se = sigma0 / np.sqrt(n)
    obsz = (mean - mu0) / se
    absz = abs(obsz)
    if alternative == "two-sided":
        pval = norm.cdf(-absz) + 1 - norm.cdf(absz)
    else:
        raise ValueError("Not implemented.")
    return obsz, pval


def chi2test(sample, sigma0, alternative="greater"):    # TODO: rename chi2test_var
    """
    Run chi2 test to detect sample variance deviation 
    from the known population variance `sigma0`.
    """
    n = len(sample)
    s2 = np.var(sample, ddof=1)
    obschi2 = (n - 1) * s2 / sigma0**2
    df = n - 1
    rvX2 = chi2(df)
    pvalue = tailprobs(rvX2, obschi2, alternative=alternative)
    return obschi2, pvalue




# SIMULATION TEST (Section 3.3)
################################################################################

def gen_sampling_dist(rv, statfunc, n, N=10000):
    """
    Simulate `N` samples of size `n` from the RV `rv`
    to generate the sampling distribution of `statfunc`.
    """
    stats = []
    for i in range(0, N):
        sample = rv.rvs(n)
        stat = statfunc(sample)
        stats.append(stat)
    return stats


def simulation_test_mean(sample, mu0, sigma0, N=10000):
    """
    Compute the p-value of the observed mean of `sample`
    under H0 of a normal distribution `norm(mu0,sigma0)`.
    """
    # 1. Compute the observed value of the mean
    obsmean = mean(sample)
    n = len(sample)

    # 2. Get sampling distribution of mean under H0
    rvXH0 = norm(mu0, sigma0)
    xbars = gen_sampling_dist(rvXH0, statfunc=mean, n=n)

    # 3. Compute the p-value
    tails = tailsof(xbars, obsmean)
    pvalue = len(tails) / len(xbars)
    return xbars, pvalue


def simulation_test(sample, rvH0, statfunc, N=10000, alternative="two-sided"):
    """
    Compute the p-value of `statfunc(sample)` under H0
    described by the random variable `rvH0`.
    """
    # 1. Compute the observed value of the mean for the sample
    obsstat = statfunc(sample)
    n = len(sample)

    # 2. Obtain the sampling distribution of the mean under H0
    statsH0 = gen_sampling_dist(rvH0, statfunc=statfunc, n=n)

    # 3. Compute the p-value
    tails = tailsof(statsH0, obsstat, alternative=alternative)
    pvalue = len(tails) / len(statsH0)
    return statsH0, pvalue




# BOOTSTRAP TEST FOR THE MEAN
################################################################################

def bootstrap_stat(sample, statfunc, B=10000):
    """
    Compute the sampling distribiton of `statfunc`
    from `B` bootstrap samples generated from `sample`.
    """
    n = len(sample)
    bstats = []
    for i in range(0, B):
        bsample = np.random.choice(sample, n, replace=True)
        bstat = statfunc(bsample)
        bstats.append(bstat)
    return bstats


def bootstrap_test_mean(sample, mu0, B=10000):
    """
    Compute the p-value of the observed `mean(sample)`
    under H0 with mean `mu0`. Model the variability of
    the distribution using bootstrap estimation.
    """
    # 1. Compute the observed value of the mean
    obsmean = mean(sample)

    # 2. Get sampling distribution of the mean under H0
    #    by "shifting" the sample so its mean is `mu0`
    sample_H0 = np.array(sample) - obsmean + mu0
    bmeans = bootstrap_stat(sample_H0, np.mean, B=B)
    
    # 3. Compute the p-value
    tails = tailsof(bmeans, obsmean)
    pvalue = len(tails) / len(bmeans)
    return bmeans, pvalue



# PERMUTATION TEST
################################################################################

def resample_under_H0(sample1, sample2):
    """
    Generate new samples from a random permutation of
    the values in the samples `sample1` and `sample2`.
    """
    values = np.concatenate((sample1, sample2))
    shuffled_values = np.random.permutation(values)
    resample1 = shuffled_values[0:len(sample1)]
    resample2 = shuffled_values[len(sample1):]
    return resample1, resample2


def permutation_test(sample1, sample2, statfunc, P=10000):
    """
    Compute the p-value of the observed `statfunc(sample1, sample2)` under
    the null hypothesis where the group membership is randomized.
    """
    # 1. Compute the observed value of `statfunc`
    obsstat = statfunc(sample1, sample2)

    # 2. Get sampling dist. of `statfunc` under H0
    pstats = []
    for i in range(0, P):
        resample1, resample2 = resample_under_H0(sample1, sample2)
        pstat = statfunc(resample1, resample2)
        pstats.append(pstat)

    # 3. Compute the p-value
    tailstats = tailsof(pstats, obsstat)
    pvalue = len(tailstats) / len(pstats)

    return pstats, pvalue




# STANDARDIZED EFFECT SIZE
################################################################################

def cohend(sample, mu):
    """
    Compute Cohen's d for one group compared to the theoretical mean `mu`.
    """
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)
    cohend = (mean - mu) / std
    return cohend


def cohend2(sample1, sample2):
    """
    Compute Cohen's d measure of effect size for two independent samples.
    """
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    # calculate the pooled variance and standard deviaiton
    pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    cohend = (mean1 - mean2) / pooled_std
    return cohend




# T-TESTS
################################################################################

def ttest_mean(sample, mu0, alternative="two-sided"):
    """
    T-test to detect mean deviation from a population with known mean `mu0`.
    """
    assert alternative in ["greater", "less", "two-sided"]
    obsmean = np.mean(sample)
    n = len(sample)
    std = np.std(sample, ddof=1)
    sehat = std / np.sqrt(n)
    obst = (obsmean - mu0) / sehat
    rvT = tdist(n-1)
    pvalue = tailprobs(rvT, obst, alternative=alternative)
    return obst, pvalue


def ttest_dmeans(sample1, sample2, equal_var=False, alternative="two-sided"):
    """
    T-test to detect difference between two groups based on their means.
    """
    # 1. Calculate the observed mean difference between means
    obsd = np.mean(sample1) - np.mean(sample2)
    # 2. Calculate the sample size and the standard deviation for each group
    n1, n2 = len(sample1), len(sample1)
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)

    if equal_var:
        # 4. Compute the pooled variance and standard error of estimator D
        pooled_var = ((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var)
        seD = pooled_std * np.sqrt(1/n1 + 1/n2)
        # 5. Obtain the degrees of freedom
        df = n1 + n2 - 2

    else:
        # 4'. Compute the standard error of the estimator D (Welch's t-test)
        seD = np.sqrt(std1**2/n1 + std2**2/n2)
        # 5'. Obtain the degrees of freedom from the crazy formula
        df = (std1**2/n1 + std2**2/n2)**2 / \
            ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1) )

    # 6. Compute the value of the t-statistic
    obst = (obsd - 0) / seD

    # Calculate the p-value
    rvT = tdist(df)
    pvalue = tailprobs(rvT, obst, alternative=alternative)
    return obst, pvalue


def ttest_paired(sample1, sample2, alternative="two-sided"):
    """
    Implement `ttest_rel`
    """
    n = len(sample1)
    n2 = len(sample2)
    assert n == n2, "Paired t-test assumes both saamples are of the same size"
    ds = np.array(sample1) - np.array(sample2)
    std = np.std(ds, ddof=1)
    meand  = np.mean(ds)
    se = std / np.sqrt(n)
    df = n - 1
    obst = (meand - 0) / se
    rvT = tdist(df)
    pvalue = tailprobs(rvT, obst, alternative=alternative)
    return obst, pvalue
