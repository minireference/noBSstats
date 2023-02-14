Datasets
========

This folder contains the datasets used for the examples, exercises, and problem in the **No Bullshit Guide to Statistics**.


## Dataset 1: Website A/B test

(coming soon)


## Dataset 2: Electricity prices

(coming soon)


## Dataset 3: Student scores 

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
