# No Bullshit Stats Notebooks

Use the binder button below to start an ephemeral JupyterLab instance where you
can run the code examples in all the [`notebooks`](./notebooks/README.md) interactively:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main)

## Contents
- [`stats_overview/`](./stats_overview/README.md): a complete worked example to introduce the main concepts of statistics
- [`notebooks/`](./notebooks/README.md): notebooks to accompany each section of the book.
- [`tutorials/`](./tutorials/appendix.md): tutorials that introduce Python basics, and the Pandas and Seaborn Python libraries.


Read on for more info about statistics and why it's worth learning this stuff.


Why learn stats?
----------------
Understanding statistics is of central importance for everyone in the modern days.
We're surrounded by data, including surveys data, analytics, personal data,
location data, sales data, scientific experiments data, etc.
Scientists, researchers, and even business folks are starting to realize they need
to learn the tools of statistics to make sense of all the data that surrounds them.
Indeed, statistical thinking and procedures are at the core of modern research methods,
including data-driven business.


Book pitch
----------
The **No Bullshit Guide to Statistics** presents both theoretical and practical
aspects of statistics as part of a connected whole. The book combines hands-on
matters of data manipulation like exporting data from databases, and also digs
into probability theory prerequisites so readers can become fluent with math models
for random phenomena. With both theory and practical prerequisites in place,
readers can tackle stats topics and truly understand the subject instead of just
skimming the surface by applying stats formulas blindly.

The purpose of this book is to introduce the subject of statistics in a rigorous,
yet accessible manner. Let's call this the "concise and precise" approach.
In order to define statistics concepts precisely, we have to dig deep into
probability theory, which will require a lot of math and equations.
Don't worry: we'll explain the reasoning behind the equations in plain English.



Good news; bad news
-------------------
This book is both good news and bad news for you, depending on your perspective on it.

**The good news is that I'm going to teach you everything I know about statistics.**
This means I'll expose you to all the equations, formula derivations, computations,
including multiple alternative approaches for doing the same calculations.
I'm aiming for a full coverage of all the concepts I leaned in the last couple
of years while researching this book and iterating over the initial drafts.
By the end of this book, you'll not only have all the stats formulas in your toolbox,
but you'll also have practical skills how to do data management,
and do probability and stats calculations using Python code
(think short paragraphs of 3 to 10 lines of instructions that implement some procedure).
I made sure that every math formula in this book is also shown in terms Python code.
The math and the code are both important ingredients for achieving a full understanding of probability theory and statistics.

If you're thinking of using "but I don't know coding" or "but I don't know Python"
as excuse to bail out at this point, it's not allowed, because I have prepared
for you an extensive, step-by-step [Python tutorial](./tutorials/python_tutorial.ipynb),
which will bring you up to speed on the basic Python syntax you need to know
when using Python as a calculator.

**The bad news is that I'm going to teach you everything I know about statistics.**
This means there will be a lot of equations in this book,
and you'll need to concentrate to understand fancy math expression involving things
subscripts like $x_i$, summations like $\sum_{i=1}^n x_i$, integrals like $\int_\cal{X} f_X(x) dx$,
expected values like $\mathbb{E}_X[g(X)]$, and all kinds of other weird-ass math notation.
Yes, the math complexity will escalate quickly, but you have to trust me on this:
it's all necessary complexity.

**More bad news**
You'll also have to endure lots of code examples that illustrate statistical procedures
expressed as paragraphs of Python commands and sometimes define Python functions.
These code examples will help to keep us honest—if answers we obtain using math formulas
and the answers we obtain by running Python code, then we can be sure we're doing things right.
Moreover,
you can play with the code examples by changing the parameters to explore what
happens interactively (generate random variables, plot distributions, run simulations etc.)
We'll do a lot of that in the code examples,
and I encourage you to do more explorations on your own by playing with the notebooks.



How
---
To learn statistics, you need to become familiar with statistical concepts like
estimates, estimators, expected values, sampling distributions, p-values, and confidence intervals.
You'll also need to learn about the various pre-packaged procedures for doing statistical tests,
and understand the assumptions behind each test.
You need to also know about the most common mistakes and pitfalls when
applying statistical testing procedures.
The book covers all these aspects in full detail in Chapter 3.



## Book plug

These notebooks are intended to support the upcoming book titled
**No Bullshit Guide to Statistics** by Ivan Savov  (Minireference Publishing).
Click on the links below to learn more about the book:

- [Detailed book outline](https://docs.google.com/document/d/1fwep23-95U-w1QMPU31nOvUnUXE2X3s_Dbk5JuLlKAY/edit#)
- [Concept map](https://minireference.com/static/excerpts/noBSstats/conceptmaps/BookSubjectsOverview.pdf)
- [Sign up to the mailing list](https://confirmsubscription.com/h/t/A17516BF2FCB41B2)
  to receive preview PDFs of the chapters as they are developed, or follow this
  repository on GitHub so you'll know when new notebooks are added.

The target release date (for pre-orders of a finished book draft) is April 2023.
