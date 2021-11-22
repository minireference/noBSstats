# noBS Stats Notebooks

Hands-on tutorials, exercises, and projects to accompany the **No Bullshit Guide to Statistics**.
Motivation: reading about stats is not enough... you need to experiment with stats, play with different examples, and do some real-world data analysis tasks. This is what this repo is about.


## Getting started

Use the binder button below to start an ephemeral JupyterLab instance where you can run the code in each notebook.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/minireference/noBSstatsnotebooks/main)

## Contents

Once have launched JupyterLab instance, use the file browser (folder icon in the left pane) to navigate to any of these subfolders and try the notebooks in them:

- [`stats_overview/`](./stats_overview/]: a complete worked example that introduces main concepts of statistics
- [`notebooks/`](./notebooks/): other explorations (mostly in draft form)


## Project info


### Tech stack
- Python and notebooks runnable in jupterlab
- data manipulation using `pandas` and some `numpy`
- graphing with `seaborn`
- prob. and stats functionality from `scipy`
- advanced stats from `statsmodels`


### Pedagogy
- hands on interactive exercises of complete statistical analyses (data->model->stats->report)
- scaffolding (three-quarters filled, half-filled, and fully independent)
- task: learner has to write a Python function
  with well defined inputs (given...) that produces the expected output
- the output of the function gets tested using a test function either inline or imported
  Prior work: some tutorials ask learners to assign their answer to a variable then tests the answer using
  [assert statements] (https://datascienceinpractice.github.io/assignments/D2_Pandas.html)
  We'll be taking this idea to the next level by asking learners to "assign" a callable to the name.



### TODOs

- Convert repo to jupyter-book
- 

