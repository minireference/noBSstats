Datasets
========

This folder contains the datasets used for the examples,
exercises, and problem in the **No Bullshit Guide to Statistics**.

The files are stored in `CSV` format,
which you can load using the pandas function `pd.read_csv()`.



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
- `raw/epriceswide.csv`

- columns:
  - `loc`: location of the charging station
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





## Exercises datasets


## Problems datasets


### Test marks

- Dataset: [`markswide.csv`](./markswide.csv)
- Description: A toy dataset in wide format.
- Format: CSV file in wide format that has 7 rows and observations from 3 variables:
  - `student_ID`: unique identifier for the student
  - `name`: student name
  - `test1`: the marks from the first test
  - `test2`: the marks from the second test
  - `test3`: the marks from the third test
- Source: Synthetic data created by the author.
- Use case: Practice using the `.melt()` method to convert to long form (tidy data).




### Grades

TODO

- Datasets: [`grades.csv`](./grades.csv)
- Description: 
- Format: CSV file that contain n observations and k variables:
  - `hours` = number of hours student spent studying
  - `grade` = 
  - `tutor` = 0 or 1  
  - `sup` = amount of support (hours) -- computed from tutor
  - `mem` = second confounder
  - `M` = mediator
  - `C` = collider



### Medical trial

TODO

- Datasets: [`medtrial.csv`](./medtrial.csv)
- Description: TODO
- Format: CSV file that contain n observations and k variables:
  - `id`
  - `age`
  - `sex`
  - `pre` = pretest 
  - `treat`: =1 hypertension medication; treat=0
  - `sev`: = severity (SBP) 
- Source: Synthetic dataset
- Use cases:
  - computed columns https://chatgpt.com/c/69b1b782-3600-8327-a844-6e92c94e2214 
  - classify into stages https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/in-depth/blood-pressure/art-20050982 




### Blood pressure datasets

- Datasets: [`bpwide.csv`](./bpwide.csv) and [`bplong.csv`](./bplong.csv)
- Description: Fictional blood-pressure data provided in "wide format" and "long format"
- Format: CSV file that contain 240 observations and 5 variables:
  - `patient`: patient id
  - `sex`: patient sex
  - `agegrp`: age group
  - `when`: when the blood was taken
  - `bp`: blood pressure measurement
- Source:
  - Data files [`bpwide.dta`](http://www.stata-press.com/data/r9/bpwide.dta)
    and [`bplong.dta`](http://www.stata-press.com/data/r9/bplong.dta)
    are provided as part of the [Stata User's Guide](https://www.stata-press.com/data/r9/u.html).
	



## Raw datasets

The subdirectory `raw/` contains "raw" data files that need data cleanup and transformations,
and are used as part of the exercises in the Pandas tutorial (Appendix D).

- `raw/minimal.csv`


## Data file formats datasets

The subdirectory `formats/` contains examples of clean data in various file formats like.

- `formats/minimal.tsv`: TSV
- `formats/minimal.xlsx`: Microsoft Excel file
- `formats/minimal.ods`: OpenDocument spreadsheet
- `formats/minimal.json`: JSON
- `formats/minimal.html`: HTML table
- `formats/minimal.xml`: XML
- `formats/minimal.sqlite`: SQLite3 database
- `formats/students_meta.csv`: the students dataset with extra metadata rows at the top




### Old Faithful

Old Faithful is a geyser in the Yellowstone National Park in Wyoming, USA.
It erupts every 35 to 120 minutes for 1 to 5 minutes.
This dataset records duration of the eruption and the waiting time between eruptions.

- Datasets: [`faithful.csv`](./faithful.csv)
- Description: 
- Format: CSV file that contain 272 observations and 2 variables:
  - `duration`: how long the eruption lasts in minutes
  - `waiting`: time until the next eruption in minutes
- Source:
  - https://plotnine.org/reference/faithful.html
  - Härdle, W. (1991) Smoothing Techniques with Implementation in S. New York: Springer.
  - https://yellowstone.net/geysers/old-faithful/





### NAME

- Datasets: [``](./)
- Description: 
- Format: CSV file that contain n observations and k variables:
  - `id`

- Source:

