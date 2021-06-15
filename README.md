# noBSstatsnotebooks
Hands-on tutorials, exercises, and projects to accompany the No Bullshit Guide to Statistics.

Motivation: reading about data stuff is not enough... learners need to play with data. This is what this repo is about.


### Data
- synthetic datasets design for specific teaching reasons
- real data too

### Tech stack
- pandas
- graphing with seabordn and/or plotnine
- numpy + scipy
- statsmodels

### Pedagogy
- hands on interactive exercises of complete statistical analyses (data->model->stats->report)
- scaffolding (three-quarters filled, half-filled, and fully independent)
- task: learner has to write a Python function
  with well defined inputs (given...) that productes
  the appopritate output
- the output of the function gets tested using a test function either inline or imported

Prior work: some tutorials ask learners to assign their answer to a variable then tests the answer using [assert statements](https://datascienceinpractice.github.io/assignments/D2_Pandas.html)

we're taking this idea to the next level by asking learners to "assign" a callable to the name


### Features (planned):

- load ipynb files from google drive https://github.com/minireference/nbexporter
  (for cases where google-drive shared notebook is the source of truth)
- run tests using github actions
- split-off exercise solutions to create tutorial and solutionaire
  see https://github.com/NeuromatchAcademy/course-content/blob/master/ci/process_notebooks.py#L7-L8
- external tests installable using !pip install https://github... 
  (this way we can have simple solutions hidden,
   and also allow more extensive testing than one or two lines-not cluttering the learners nb)







 
