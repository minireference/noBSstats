"""
This file contains helper functions for plotting the probability distributions.
MIT License Minireferece Co.
"""
# TODO: change x to xs (to signal it's a array-like)




import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import seaborn as sns





def calc_prob_and_plot(rv, a, b, xlims=None, ax=None, title=None):
    """
    Calculate the probability random variable `rv` falls between a and b,
    and plot the area-under-the-curve visualization ofr this calculation.
    """

    # 1. calculate Pr(a<X<b) == integral of rv.pdf between x=a and x=b
    p = quad(rv.pdf, a, b)[0]

    # 2. plot the probability density function (pdf)
    if xlims:
        xmin, xmax = xlims
    else:
        xmin, xmax = rv.ppf(0.001), rv.ppf(0.999)
    x = np.linspace(xmin, xmax, 10000)
    pX = rv.pdf(x)
    ax = sns.lineplot(x=x, y=pX, ax=ax)
    if title is None:
        title = "Probability density for the random variable " + rv.dist.name + str(rv.args) \
                 + " between " + str(a) + " and " + str(b)
    ax.set_title(title, y=0, pad=-30)

    # 3. highlight the area under pX between x=a and x=b
    mask = (x > a) & (x < b)
    ax.fill_between(x[mask], y1=pX[mask], alpha=0.2, facecolor="blue")
    ax.vlines([a], ymin=0, ymax=rv.pdf(a), linestyle="-", alpha=0.5, color="blue")
    ax.vlines([b], ymin=0, ymax=rv.pdf(b), linestyle="-", alpha=0.5, color="blue")
    
    # return prob and figure axes
    return p, ax




def calc_prob_and_plot_tails(rv, x_l, x_r, xlims=None, ax=None, title=None):
    """
    Plot the area-under-the-curve visualization for the distribution's tails and
    calculate their combined probability mass: Pr({X < x_l}) + Pr({X > x_r}).
    """
    # 1. compute the probability in the left (-∞,x_l] and right [x_r,∞) tails
    p_l = quad(rv.pdf, rv.ppf(0.0000000000001), x_l)[0]
    p_r = quad(rv.pdf, x_r, rv.ppf(0.9999999999999))[0]
    p_tails = p_l + p_r

    # 2. plot the probability density function (pdf)
    if xlims:
        xmin, xmax = xlims
    else:
        xmin, xmax = rv.ppf(0.001), rv.ppf(0.999)
    x = np.linspace(xmin, xmax, 10000)
    pX = rv.pdf(x)
    ax = sns.lineplot(x=x, y=pX, ax=ax)
    if title is None:
        title = "Tails of the random variable " + rv.dist.name + str(rv.args)
    ax.set_title(title, y=0, pad=-30)

    # 3. highlight the area under pX for the tails
    mask_l = x < x_l   # left tail
    mask_u = x > x_r   # right tail
    ax.fill_between(x[mask_l], y1=pX[mask_l], alpha=0.3, facecolor="red")
    ax.fill_between(x[mask_u], y1=pX[mask_u], alpha=0.3, facecolor="red")
    ax.vlines([x_l], ymin=0, ymax=rv.pdf(x_l), linestyle="-", alpha=0.5, color="red")
    ax.vlines([x_r], ymin=0, ymax=rv.pdf(x_r), linestyle="-", alpha=0.5, color="red")

    # return prob and figure axes
    return p_tails, ax




def plot_pdf_and_cdf(rv, b, a=-np.inf, xlims=None, rv_name="X"):
    """
    Plot side-by-side figure that shows pdf and CDF of random variable `rv`.
    Left plot shows area-under-the-curve visualization until x=b.
    Right plot higlights point at (b, F_X(b)).
    """
    fig, axs = plt.subplots(1, 2)
    ax0, ax1 = axs

    title = "Probability distributions of the random variable " \
        + "$" + rv_name + "$" + " ~ " \
        + rv.dist.name + str(rv.args).replace(" ", "")
    # + " between " + str(a) + " and " + str(b)
    fig.suptitle(title)

    # 1. plot the probability density function (pdf)
    if xlims:
        xmin, xmax = xlims
    else:
        xmin, xmax = rv.ppf(0.001), rv.ppf(0.999)
    x = np.linspace(xmin, xmax, 10000)
    pX = rv.pdf(x)
    sns.lineplot(x=x, y=pX, ax=ax0)
    ax0.set_title("Probability density function")

    # highlight the area under pX between x=a and x=b
    mask = (x > a) & (x < b)
    ax0.fill_between(x[mask], y1=pX[mask], alpha=0.2, facecolor="blue")
    ax0.vlines([b], ymin=0, ymax=rv.pdf(b), linestyle="-", alpha=0.5, color="blue")
    ax0.text(b, 0, "$b$", horizontalalignment="center", verticalalignment="top")
    ax0.text(b, rv.pdf(b)/2.5, r"Pr$(\{" + rv_name + r" \leq b \})$    ",
             horizontalalignment="right", verticalalignment="center")

    # 2. plot the CDF
    FX = rv.cdf(x)
    sns.lineplot(x=x, y=FX, ax=ax1)
    ax1.set_title("Cumulative distribution function")

    # highlight the point x=b
    ax1.vlines([b], ymin=0, ymax=rv.cdf(b), linestyle="-", color="blue")
    ax1.text(b, 0, "$b$", horizontalalignment="center", verticalalignment="top")
    ax1.text(b, rv.cdf(b), "$(b, F_{" + rv_name + "}(b))$",
             horizontalalignment="right", verticalalignment="bottom")

    # return figure and axes
    return fig, axs

