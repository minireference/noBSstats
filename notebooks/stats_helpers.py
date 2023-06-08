from collections import defaultdict
import os

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
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
        q_l = chi2.ppf(alpha/2, df=n-1)
        q_u = chi2.ppf(1-alpha/2, df=n-1)
        return [(n-1)*s2/q_u, (n-1)*s2/q_l]
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


def boot_ci(sample, estfunc, alpha=0.1, method=None, B=5000):
    """
    An adaptor for calling the function `scipy.stats.bootstrap` without the need
    to specify all all the optional keyword arguments.
    """
    res = bootstrap((sample,),
                    statistic=estfunc,
                    confidence_level=1-alpha,
                    n_resamples=B,
                    vectorized=False,
                    method=method)
    return [res.confidence_interval.low,
            res.confidence_interval.high]



# TAIL CALCULATION UTILS
################################################################################

def tailvalues(values, obs, alt="two-sided"):
    """
    Select the subset of the elements in list `values` that
    are equal or more extreme than the observed value `obs`.
    """
    assert alt in ["greater", "less", "two-sided"]
    values = np.array(values)
    if alt == "greater":
        tails = values[values >= obs]
    elif alt == "less":
        tails = values[values <= obs]
    elif alt == "two-sided":
        mean = np.mean(values)
        dev = abs(mean - obs)
        tails = values[abs(values-mean) >= dev]
    return tails


def tailprobs(rv, obs, alt="two-sided"):
    """
    Calculate the probability of all outcomes of the random variable `rv`
    that are equal or more extreme than the observed value `obs`.
    """
    assert alt in ["greater", "less", "two-sided"]
    if alt == "greater":
        pvalue = 1 - rv.cdf(obs)
    elif alt == "less":
        pvalue = rv.cdf(obs)
    elif alt == "two-sided":  # assumes symmetric dist.
        mean = rv.mean()
        dev = abs(mean - obs)
        pleft = rv.cdf(mean - dev)
        pright = 1 - rv.cdf(mean + dev)
        pvalue = 2 * min(pleft, pright)
    return pvalue





# BASIC TESTS (used for kombucha data generation)
################################################################################

def ztest(sample, mu0, sigma0, alt="two-sided"):
    """
    Z-test to detect mean deviation from known normal population.
    """
    mean = np.mean(sample)
    n = len(sample)
    se = sigma0 / np.sqrt(n)
    obsz = (mean - mu0) / se
    rvZ = norm(0,1)
    pval = tailprobs(rvZ, obsz, alt=alt)
    return pval


def chi2test_var(sample, sigma0, alt="greater"):
    """
    Run chi2 test to detect if a sample variance deviation
    from the known population variance `sigma0` exists.
    """
    n = len(sample)
    s2 = np.var(sample, ddof=1)
    obschi2 = (n - 1) * s2 / sigma0**2
    df = n - 1
    rvX2 = chi2(df)
    pvalue = tailprobs(rvX2, obschi2, alt=alt)
    return pvalue




# SIMULATION TESTS (Section 3.3)
################################################################################

def simulation_test_mean(sample, mu0, sigma0, alt="two-sided"):
    """
    Compute the p-value of the observed mean of `sample`
    under H0 of a normal distribution `norm(mu0,sigma0)`.
    """
    # 1. Compute the sample mean
    obsmean = mean(sample)
    n = len(sample)

    # 2. Get sampling distribution of the mean under H0
    rvXH0 = norm(mu0, sigma0)
    xbars = gen_sampling_dist(rvXH0, estfunc=mean, n=n)

    # 3. Compute the p-value
    tails = tailvalues(xbars, obsmean, alt=alt)
    pvalue = len(tails) / len(xbars)
    return pvalue


def simulation_test_var(sample, mu0, sigma0, alt="greater"):
    """
    Compute the p-value of the observed variance of `sample`
    under H0 of a normal distribution `norm(mu0,sigma0)`.
    """
    # 1. Compute the sample variance
    obsvar = var(sample)
    n = len(sample)

    # 2. Get sampling distribution of variance under H0
    rvXH0 = norm(mu0, sigma0)
    xvars = gen_sampling_dist(rvXH0, estfunc=var, n=n)

    # 3. Compute the p-value
    tails = tailvalues(xvars, obsvar, alt=alt)
    pvalue = len(tails) / len(xvars)
    return pvalue


def simulation_test(sample, rvH0, estfunc, alt="two-sided"):
    """
    Compute the p-value of the observed estimate `estfunc(sample)` under H0
    described by the random variable `rvH0`.
    """
    # 1. Compute the observed value of `estfunc`
    obsest = estfunc(sample)
    n = len(sample)

    # 2. Get sampling distribution of `estfunc` under H0
    sampl_dist_H0 = gen_sampling_dist(rvH0, estfunc, n)

    # 3. Compute the p-value
    tails = tailvalues(sampl_dist_H0, obsest, alt=alt)
    pvalue = len(tails) / len(sampl_dist_H0)
    return pvalue




# BOOTSTRAP TEST FOR THE MEAN (cut material)
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
    return pvalue




# PERMUTATION TEST DMEANS
################################################################################

def resample_under_H0(xsample, ysample):
    """
    Generate new samples from a random permutation of
    the values in the samples `xsample` and `ysample`.
    """
    values = np.concatenate((xsample, ysample))
    shuffled_values = np.random.permutation(values)
    xresample = shuffled_values[0:len(xsample)]
    yresample = shuffled_values[len(xsample):]
    return xresample, yresample


def permutation_test_dmeans(xsample, ysample, P=10000):
    """
    Compute the p-value of the observed difference between means
    `dmeans(sample1,sample2)` under the null hypothesis where
    the group membership is randomized.
    """
    # 1. Compute the observed difference between means
    obsdhat = dmeans(xsample, ysample)

    # 2. Get sampling dist. of `dmeans` under H0
    pdhats = []
    for i in range(0, P):
        rsx, rsy = resample_under_H0(xsample, ysample)
        pdhat = dmeans(rsx, rsy)
        pdhats.append(pdhat)

    # 3. Compute the p-value
    tails = tailvalues(pdhats, obsdhat)
    pvalue = len(tails) / len(pdhats)
    return pvalue


def permutation_test(xsample, ysample, estfunc, P=10000):
    """
    Compute the p-value of the observed estimate `estfunc(xsample,ysample)`
    under the null hypothesis where the group membership is randomized.
    """
    # 1. Compute the observed value of `estfunc`
    obsest = estfunc(xsample, ysample)

    # 2. Get sampling dist. of `estfunc` under H0
    pestimates = []
    for i in range(0, P):
        rsx, rsy = resample_under_H0(xsample, ysample)
        pestimate = estfunc(rsx, rsy)
        pestimates.append(pestimate)

    # 3. Compute the p-value
    tails = tailvalues(pestimates, obsest)
    pvalue = len(tails) / len(pestimates)
    return pvalue




# STANDARDIZED EFFECT SIZE MEASURES
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
    varp = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
    stdp = np.sqrt(varp)
    cohend = (mean1 - mean2) / stdp
    return cohend


# T-TESTS
################################################################################

def ttest_mean(sample, mu0, alt="two-sided"):
    """
    T-test to detect mean deviation from a population with known mean `mu0`.
    """
    assert alt in ["greater", "less", "two-sided"]
    obsmean = np.mean(sample)
    n = len(sample)
    std = np.std(sample, ddof=1)
    sehat = std / np.sqrt(n)
    obst = (obsmean - mu0) / sehat
    rvT = tdist(n-1)
    pvalue = tailprobs(rvT, obst, alt=alt)
    return pvalue


def ttest_dmeans(xsample, ysample, equal_var=False, alt="two-sided"):
    """
    T-test to detect difference between two populations means
    based on the difference between sample means.
    """
    # Calculate the observed difference between means
    obsdhat = mean(xsample) - mean(ysample)

    # Calculate the sample sizes and the stds
    n, m = len(xsample), len(ysample)
    sx, sy = std(xsample), std(ysample)

    # Calculate the standard error, the degrees of
    # freedom, the null model, and the t-statistic
    if not equal_var:  # Welch's t-test (default)
        seD = np.sqrt(sx**2/n + sy**2/m)
        obst = (obsdhat - 0) / seD
        df = calcdf(sx, n, sy, m)
        rvT0 = tdist(df)
    else:              # Use pooled variance
        varp = ((n-1)*sx**2 + (m-1)*sy**2) / (n+m-2)
        stdp = np.sqrt(varp)
        seDp = stdp * np.sqrt(1/n + 1/m)
        obst = (obsdhat - 0) / seDp
        dfp = n + m - 2
        rvT0 = tdist(dfp)

    # Calculate the p-value from the t-distribution
    pvalue = tailprobs(rvT0, obst, alt=alt)
    return pvalue


def ttest_paired(sample1, sample2, alt="two-sided"):
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
    obst = (meand - 0) / se
    df = n - 1
    rvT = tdist(df)
    pvalue = tailprobs(rvT, obst, alt=alt)
    return pvalue



# SIMULATION OF CONFIDENCE INTERVAL PROPERTIES
################################################################################

class mixnorms(object):
    """
    Custom class to represent mixture of normals.
    """

    def __init__(self, locs, scales, weights):
        assert len(locs) == len(scales)
        assert len(locs) == len(weights)
        self.locs = locs
        self.scales = scales
        self.weights = weights

    def pdf(self, x):
        rvNs = [norm(loc, scale) for loc, scale in zip(self.locs, self.scales)]
        terms = [w*rvN.pdf(x) for w, rvN in zip(self.weights, rvNs)]
        return sum(terms)
    
    def mean(self):
        return sum([w*loc for w, loc in zip(self.weights, self.locs)])

    def var(self):
        # via https://stats.stackexchange.com/a/604872/62481
        assert len(self.weights) == 2
        wA, wB = self.weights
        muA, muB = self.locs
        sigmaA, sigmaB = self.scales
        return wA*sigmaA**2 + wB*sigmaB**2 + wA*wB*(muA-muB)**2

    def rvs(self, n):
        rvNs = [norm(loc, scale) for loc, scale in zip(self.locs, self.scales)]
        ids = range(0,len(self.weights))
        choices = np.random.choice(ids, n, p=self.weights)
        values = np.zeros(n)
        for i, choice in enumerate(choices):
            rvN = rvNs[choice]
            values[i] = rvN.rvs(1)
        return values



def simulate_ci_props(pops, methods=["a", "percentile", "bca"], ns=[20,40],
                      param="mean", alpha=0.1, N=1000, B=5000, seed=42):
    """
    Runs a simulation of confidence intervals for `param` using `methods`
    for sample sizes `ns` from populations `pops` (dict label: model).
    Simulation parameters:
        - pops          # populations
        - methods       = ["a", "percentile", "bca"]
        - ns = [20,40]  # sample sizes
        - param         # population parameter
        - alpha = 0.1   # target error level
        - N = 1000      # number of simulations
        - B = 5000      # number of bootstrap samples
    """
    assert param in ["mean", "var"]

    # check if cached simulation data exists
    filename = "simulate_ci_props_" + param + "__ns_" + "_".join(map(str,ns)) \
                + "__alpha_" + str(alpha) + "__seed_" + str(seed) + ".csv"
    filepath = os.path.join("simdata", filename)
    if os.path.exists(filepath):  # load cached results
        print("loaded cached results from ", filepath)
        results = pd.read_csv(filepath, header=[0,1], index_col=[0,1])
        return results

    # simulation data structures
    rowsindex = pd.MultiIndex.from_product((pops.keys(),ns), names=["population", "n"])
    colindex = pd.MultiIndex.from_product((["wbar", "cov"], methods), names=["property", "method"])
    widthscolindex = pd.MultiIndex.from_product((methods,ns), names=["method","n"])
    results = pd.DataFrame(index=rowsindex, columns=colindex)

    # run simulation
    np.random.seed(seed)
    print("Starting simulation for confidnece intervals of population {param} :::::::::::::")
    for pop in pops.keys():
        print(f"Evaluating rv{pop} ...")
        rv = pops[pop]
        if param == "mean":
            pop_param = rv.mean()
        elif param == "var":
            pop_param = rv.var()
        counts = defaultdict(int)  # keys are tuples (method,n)
        widths = pd.DataFrame(index=range(0,N), columns=widthscolindex)
        for n in ns:
            print(f"  - running simulation with {n=} ...")
            for j in range(0, N):
                sample = rv.rvs(n)
                for method in methods:
                    if method == "a":
                        if param == "mean":
                            ci = ci_mean(sample, alpha=alpha, method="a")
                        elif param == "var":
                            ci = ci_var(sample, alpha=alpha, method="a")
                    else:
                        if param == "mean":
                            ci = boot_ci(sample, estfunc=np.mean, alpha=alpha, method=method, B=B)
                        elif param == "var":
                            ci = boot_ci(sample, estfunc=var, alpha=alpha, method=method, B=B)
                    # evaluate confidence interval parameters
                    if ci[0] <= pop_param <= ci[1]:
                        counts[(method,n)] += 1  # success
                    # width
                    widths.loc[j,(method,n)] = ci[1] - ci[0]
        for method in methods:
            for n in ns:
                results.loc[(pop,n), ("cov",method)] = counts[(method,n)] / N
                results.loc[(pop,n), ("wbar",method)] = widths.mean()[method,n]

    results.to_csv(filepath)
    print("Saved file to " + filepath)
    return results

