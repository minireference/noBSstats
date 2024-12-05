Datasets
========

This folder contains the datasets used for the examples, exercises, and problem in the **No Bullshit Guide to Statistics**.

The files are stored in `CSV` format, which you can load using the function `pd.read_csv()`.
Each data file has a metadata information in a file of the same name with the extension `.md`.



## Introductory datasets

### Players

- `players.csv`: 12 observations of 7 variables from a computer game
- `players_full.csv`: includes the confounding variable `jobstatus`



### Minimal

- `minimal.csv`: used in Appendix D: Introduction to Pandas
- `raw/minimal.csv`: used as an example of data cleaning of missing values



## Main datasets

These are the datasets used in examples and explanations throughout the book.


### Dataset 1: Apples weights

 - `apples.csv`



### Dataset 2: Electricity prices

- `eprices.csv`
- `epriceswide.csv`

- columns:
  - `end`: location of the charging station
  - `price`: observed price


### Dataset 3: Student scores 

- Dataset: [`students.csv`](./students.csv)
- Description: This dataset consists of student activity obtained from an online learning platform.
  A science teacher has collected data to compare the effectiveness of educational video materials delivered in one of two formats:
  a new model for teaching the material in the form of a `debate` and discussions
  vs. a more traditional `lecture` format where the teacher states facts and provides explanations.
  This [youtube video](https://www.youtube.com/watch?v=eVtCO84MDj8) explains the motivation behind this experiment.
- Format: A CSV file with 15 rows and 5 columns:
  - `student_ID` (numeric): unique identifier for each student.
  - `background` (categorical): the student's academic background.
    This variable can take on one of three possible values: `science`, `business`, and `arts`,
    depending on the student's academic background (which faculty they are enrolled in).
  - `curriculum` (categorical): which of the two different video options did student use.
    Students were randomly assigned to one of the two variants of the course: `lecture` or `debate`.
  - `effort` (numeric): total time in hours spent on the platform.
  - `score` (numeric): combined score out of 100 calculated from all the assessment items (quizzes)
    that the student completed during the course.
- Mode of data collection: Extracted from LMS database.
- Source: Caroline Smith (experimental data)


TODO: move detailed info to `students.md`



### Dataset 4: Kombucha volumes

 - `kombucha.csv`
 - `kombuchapop.csv`


### Dataset 5: Doctors sleep study

 - `doctors.csv`



### Dataset 6: Website A/B test

- `visitors.csv`

- columns:
  - `version` (categorical) which is either B = baseline or A = alternative
  - `bought` (boolean) whether visitor bought something or not




## Datasets used in exercises


### Asthma
- Dataset: [`exercises/asthma.csv`](./exercises/asthma.csv)


### Admissions (binary decision)

- Dataset: [`exercises/binary.csv`](./exercises/binary.csv)

### Honors class

- Dataset: [`exercises/honors.csv`](./exercises/honors.csv)


### Titanic

- Dataset: [`exercises/titanic.csv`](./exercises/titanic.csv)




## Real-world datasets

### Forced expiratory volume (FEV) dataset

- Dataset: [`smokefev.csv`](./smokefev.csv)
- Description: Sample of 654 youths, aged 3 to 19, in the area of East Boston
  during middle to late 1970's. Interest concerns the relationship
  between smoking and FEV. Since the study is necessarily
  observational, statistical adjustment via regression models
  clarifies the relationship.
- Format: A CSV file with 654 observations and 5 variables:
  - `age`: positive integer (years)
  - `fev`: continuous measure (liters)
  - `height`: continuous measure (inches)
  - `sex`: categorical (Female coded `F`, Male coded `M`)
  - `smoke`: categorical (Nonsmoker coded `NS`, Smoker coded `SM`)
- Source: Rosner, B. (1999), Fundamentals of Biostatistics, 5th Ed., Pacific Grove, CA: Duxbury

<!--
# # Original data source
# smokefev_raw = pd.read_fwf("http://jse.amstat.org/datasets/fev.dat.txt",
#                        colspecs=[(0,3),(4,10),(11,15), (18,19),(24,25)],
#                        names=["age", "fev", "height", "sex", "smoke"])
# smokefev_raw["sex"] = smokefev_raw["sex"].replace({0:"F", 1:"M"})
# smokefev_raw["smoke"] = smokefev_raw["smoke"].replace({0:"NS", 1:"SM"})
# smokefev_raw.to_csv("../datasets/smokefev.csv", index=False)
-->


### Lalonde

An experiment whether training increases income.


### Howell

The dataset `howell30.csv` contains `age`, `sex`, `height`, and `weight` measurements
from a sample of 270 individuals of the !Kung San people of Botswana.
The dataset was compiled by Nancy Howell between 1967 and 1969.
See [here](https://tspace.library.utoronto.ca/handle/1807/10395) for more info about the dataset.
The specific data file I used was [`All_17996.xls`](https://tspace.library.utoronto.ca/handle/1807/17996).
I have filtered the data to select only individuals of age 30 or less.


### Radon

- Dataset: [`radon.csv`](./radon.csv)
- Description: Radon measurements of 919 homes in 85 Minnesota counties.
- Format: A CSV file with 919 observations and 6 variables:
  - `idnum`: unique identifier from each house.
  - `state`: `MN` constant for all observations.
  - `county`: county name (this is the grouping variable).
  - `floor`: where the radon measurement was taken: `basement` or `ground` floor
  - `log_radon`: Radon measurement (in log pCi/L, i.e., log picoCurie per liter)
  - `log_uranium`: Average county-level soil uranium content.
- Source:
  - Gelman, A. and Hill, J. (2007) Data analysis using regression and multilevel/hierarchical models. Cambridge University Press. 
  - Price, P. N., Nero, A. V. and Gelman, A. (1996) Bayesian prediction of mean indoor radon concentrations for Minnesota counties. Health Physics. 71(6), 922–936.




## Other datasets

Additional datasets you'll need for the exercises can be found in the directory `exercises/`
while data sets used for problems are in `problems/`.

The directory `formats/` contains examples of clean data in various file formats like.

The directory `datasets/raw/` contains "raw" data files that need data cleanup and transformations,
and are used as part of the exercises in the Pandas tutorial (Appendix D).
