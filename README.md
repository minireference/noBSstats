# noBS STATS Notebooks

Hands-on tutorials, exercises, and projects to accompany the **No Bullshit Guide to Statistics**.
Motivation: reading about stats is not enough... you need to experiment with stats,
play with different examples, and do some real-world data analysis tasks.
This is what this repo is about.


## Getting started
Use the binder button below to start an ephemeral JupyterLab instance where you can run the code in each notebook.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main)


## Contents
Once have launched JupyterLab instance, use the file browser (folder icon in the left pane)
to navigate to any of these subfolders and try the notebooks in them:

- [`datasets/`](./datasets/): all the data used for examples, exercises, and problems in the book.
- [`notebooks/`](./notebooks/): notebooks to accompany each section of the book.
- [`stats_overview/`](./stats_overview/): a complete worked example to introduce
  all the statistics concepts from the book: DATA, PROB, STATS, and LINEAR MODELS.
- [`exercises/`](./exercises/): simple, routine exercises to practice new concepts.
- [`problems/`](./problems/): problems requiring some thinking **COMING SOON**
- [`missions/`](./missions/): multi-step procedures and workflows **COMING SOON**
- [`tutorials/`](./tutorials/): tutorials that introduce Python basics, and the Pandas and Seaborn Python libraries.


## Other stuff

This repo also contains the "utility" code and notebooks that were used for the book:

- [`data_generation`](./data_generation): notebooks used to generate datasets used in the book.
- [`figures_generation`](./figures_generation): notebooks to run to generate figures and tables.
- [`notebooks/explorations/`](./notebooks/explorations/): general exploratory notebooks on statistics.
- [`notebooks/drafts/`](./notebooks/drafts/): notebook drafts (not ready for prime time).


### Tech stack
- Python and notebooks runnable in `jupter-lab`
- data manipulation using `pandas` and some `numpy`
- graphing with `seaborn`
- prob. and stats functionality from `scipy`
- advanced stats from `statsmodels`

All of these can be installed by running `pip install -r reqirements.txt` in the
root of this project.



### Pedagogical notes

- The goal is to have lots of hands-on activities, called missions, that ask
  readers to do a complete statistical analyses (data->model->stats->report)
- Each mission will provide some scaffolding so learners are not left completely
  on their own (for beginners 3/4 of the steps are already filled in, then for
  intermediate learners notebook is half-filled, and advanced learners are just
  provided with the goal and otherwise left to work independently)
- Most common task: write a Python function that fulfills a detailed specification,
  including what inputs to accept and what output the function is expected to produce.
- Student's solution (a filled-in function) gets tested using a test function,
  just like unit test used in software projects.
  - Prior work: tutorial ask learners to assign their answer to a variable:
    [assert statements](https://datascienceinpractice.github.io/assignments/D2_Pandas.html)
    We'll be taking this idea to the next level by asking learners to define a callable.



### TODOs

- write CI script to strip out exercise solutions from notebooks

