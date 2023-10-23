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




## Other datasets

Additional datasets you'll need for the exercises can be found in the directory `exercises/`
while data sets used for problems are in `problems/`.

The directory `formats/` contains examples of clean data in various file formats like.

The directory `datasets/raw/` contains "raw" data files that need data cleanup and transformations,
and are used as part of the exercises in the Pandas tutorial (Appendix D).
