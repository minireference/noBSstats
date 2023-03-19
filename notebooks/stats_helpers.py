import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import t as tdist

#######################################################
# max width that fits in the code blocks is 55 chars  #


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



# SAMPLING DISTRIBUTIONS
################################################################################

def gen_sampling_dist(rv, estfunc, n, N=10000):
    """
    Simulate `N` samples of size `n` from the random variable `rv` to
    generate the sampling distribution of the estimator `estfunc`.
    """
    estimates = []
    for i in range(0, N):
        sample = rv.rvs(n)
        estimate = estfunc(sample)
        estimates.append(estimate)
    return estimates



# BOOTSTAP
################################################################################

def gen_boot_dist(sample, estfunc, B=5000):
    """
    Generate estimates from the sampling distribiton of the estimator `estfunc`
    based on `B` bootstrap samples (sampling with replacement) from `sample`.
    """
    n = len(sample)
    bestimates = []
    for i in range(0, B):
        bsample = np.random.choice(sample, n, replace=True)
        bestimate = estfunc(bsample)
        bestimates.append(bestimate)
    return bestimates




# CONFIDENCE INTERVALS
################################################################################

def ci_mean(sample, alpha=0.1, method="a"):
    """
    Compute the confidence interval for the population mean.
    - method "a" will computes analytical approximation based on Student's t-dist
    """
    assert method in ["a", "b"]
    if method == "a":        # analytical approximation
        from scipy.stats import t as tdist
        n = len(sample)
        xbar = np.mean(sample)
        sehat = np.std(sample, ddof=1) / np.sqrt(n)
        t_l = tdist.ppf(alpha/2, df=n-1)
        t_u = tdist.ppf(1-alpha/2, df=n-1)
        return [xbar + t_l*sehat, xbar + t_u*sehat]
    elif method == "b":          # bootstrap estimation
        from stats_helpers import gen_boot_dist
        xbars_boot = gen_boot_dist(sample, estfunc=mean)
        return [np.quantile(xbars_boot, alpha/2),
                np.quantile(xbars_boot, 1-alpha/2)]


def ci_var(sample, alpha=0.1, method="a"):
    """
    Compute the confidence interval for the population variance.
    """
    assert method in ["a", "b"]
    if method == "a":        # analytical approximation
        n = len(sample)
        s2 = np.var(sample, ddof=1)
        x2_l = chi2.ppf(alpha/2, df=n-1)
        x2_u = chi2.ppf(1-alpha/2, df=n-1)
        return [(n-1)*s2/x2_u, (n-1)*s2/x2_l]
    elif method == "b":          # bootstrap estimation
        from stats_helpers import gen_boot_dist
        vars_boot = gen_boot_dist(sample, estfunc=var)
        return [np.quantile(vars_boot, alpha/2),
                np.quantile(vars_boot, 1-alpha/2)]


def ci_dmeans(xsample, ysample, alpha=0.1, method="a"):
    """
    Compute the confidence interval for the difference between population means.
    """
    assert method in ["a", "b"]
    if method == "a":        # analytical approximation
        stdX, n = np.std(xsample, ddof=1), len(xsample)
        stdY, m = np.std(ysample, ddof=1), len(ysample)
        dhat = np.mean(xsample) - np.mean(ysample)
        seD = np.sqrt(stdX**2/n + stdY**2/m)
        df = calcdf(stdX, n, stdY, m)
        t_l = tdist.ppf(alpha/2, df=df)
        t_u = tdist.ppf(1-alpha/2, df=df)
        return [dhat + t_l*seD, dhat + t_u*seD]
    elif method == "b":          # bootstrap estimation
        from stats_helpers import gen_boot_dist
        xbars_boot = gen_boot_dist(xsample, np.mean)
        ybars_boot = gen_boot_dist(ysample, np.mean)
        dmeans_boot = np.subtract(xbars_boot,ybars_boot)
        return [np.quantile(dmeans_boot, alpha/2),
                np.quantile(dmeans_boot, 1-alpha/2)]




# TAIL CALCULATION UTILS
################################################################################

def tailvalues(values, obs, alternative="two-sided"):
    """
    Select the subset of the elements in list `values` that
    are equal or more extreme than the observed value `obs`.
    """
    assert alternative in ["greater", "less", "two-sided"]
    values = np.array(values)
    if alternative == "greater":
        tails = values[values >= obs]
    elif alternative == "less":
        tails = values[values <= obs]
    elif alternative == "two-sided":
        mean = np.mean(values)
        dev = abs(mean - obs)
        tails = values[abs(values-mean) >= dev]
    return tails


def tailprobs(rv, obs, alternative="two-sided"):
    """
    Calculate the probability of all outcomes of the random variable `rv`
    that are equal or more extreme than the observed value `obs`.
    """
    assert alternative in ["greater", "less", "two-sided"]
    if alternative == "greater":
        pvalue = 1 - rv.cdf(obs)
    elif alternative == "less":
        pvalue = rv.cdf(obs)
    elif alternative == "two-sided":
        pleft = rv.cdf(obs)
        pright = 1 - rv.cdf(obs)
        pvalue = 2 * min(pleft, pright)
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


def chi2test_var(sample, sigma0, alternative="greater"):
    """
    Run chi2 test to detect if a sample variance deviation
    from the known population variance `sigma0` exists.
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
    xbars = gen_sampling_dist(rvXH0, estfunc=mean, n=n)

    # 3. Compute the p-value
    tails = tailvalues(xbars, obsmean)
    pvalue = len(tails) / len(xbars)
    return xbars, pvalue


def simulation_test(sample, rvH0, estfunc, N=10000, alternative="two-sided"):
    """
    Compute the p-value of the observed estimate `estfunc(sample)` under H0
    described by the random variable `rvH0`.
    """
    # 1. Compute the observed value of the mean for the sample
    obsest = estfunc(sample)
    n = len(sample)

    # 2. Obtain the sampling distribution of the mean under H0
    sampl_dist_H0 = gen_sampling_dist(rvH0, estfunc=estfunc, n=n)

    # 3. Compute the p-value
    tails = tailvalues(sampl_dist_H0, obsest, alternative=alternative)
    pvalue = len(tails) / len(sampl_dist_H0)
    return sampl_dist_H0, pvalue




# BOOTSTRAP TEST FOR THE MEAN
################################################################################

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
    bmeans = gen_boot_dist(sample_H0, np.mean, B=B)
    
    # 3. Compute the p-value
    tails = tailvalues(bmeans, obsmean)
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


def permutation_test(sample1, sample2, estfunc, P=10000):
    """
    Compute the p-value of the observed estimate `estfunc(sample1,sample2)`
    under the null hypothesis where the group membership is randomized.
    """
    # 1. Compute the observed value of `estfunc`
    obsest = estfunc(sample1, sample2)

    # 2. Get sampling dist. of `estfunc` under H0
    pestimates = []
    for i in range(0, P):
        rs1, rs2 = resample_under_H0(sample1, sample2)
        pestimate = estfunc(rs1, rs2)
        pestimates.append(pestimate)

    # 3. Compute the p-value
    tails = tailvalues(pestimates, obsest)
    pvalue = len(tails) / len(pestimates)

    return pestimates, pvalue




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

    # 3. Calculate the standard error and degrees of f.
    if equal_var:
        # Use pooled variance
        pooled_var = ((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var)
        seD = pooled_std * np.sqrt(1/n1 + 1/n2)
        df = n1 + n2 - 2
    else:
        # Compute the standard error using general formula (Welch's t-test)
        seD = np.sqrt(std1**2/n1 + std2**2/n2)
        # Use Welch's formula for degrees of freedom
        df = calcdf(std1, n1, std2, n2)

    # 4. Compute the value of the t-statistic
    obst = (obsd - 0) / seD

    # 5. Calculate the p-value from the t-distribution
    rvT = tdist(df)
    pvalue = tailprobs(rvT, obst, alternative=alternative)
    return obst, pvalue


def ttest_paired(sample1, sample2, alternative="two-sided"):
    """
    T-test for comparing relative change in a set of pairded measurements.
    """
    n = len(sample1)
    n2 = len(sample2)
    assert n == n2, "Paired t-test assumes both samples are of the same size."
    ds = np.array(sample1) - np.array(sample2)
    std = np.std(ds, ddof=1)
    meand  = np.mean(ds)
    se = std / np.sqrt(n)
    df = n - 1
    obst = (meand - 0) / se
    rvT = tdist(df)
    pvalue = tailprobs(rvT, obst, alternative=alternative)
    return obst, pvalue

