Design of exercise processing steps
===================================

cf. [scripts/process_tutorials.py](../scripts/process_tutorials.py)


Folder structure
----------------

    exercises/                                  (B1)
        exercises_13_descr_stats.ipynb          (ex:exrc)
        src/                                    (SRC)
            exercises_13_descr_stats_src.ipynb  (ex:src)
        solutions/                              (SOL)
            exercise_find_mean.py               (ex:sol)
    notebooks/                                  (B2) 
        13_descriptive_statistics.ipynb         (ex:nb)



where:
- `(SRC)` main location for exercises notebooks with solutions
- `(B1)` build output for standalone exercise notebooks
- `(B2)` build output for section notebooks (append to the `## Exercises` block in the notebook)
- `(SOL)` main location for solution snippets

The full notebook with solutions lives in `exercises/src/`.
This is where you edit and create the exercises.
The build script parses these notebooks and outputs to `(B1)`, `(B2)`, and `(SOL)`.


The setup is similar for problem files

    problems/                                   (B1)
        14_data_problems.ipynb                  (ex:probnb1)
        src/                                    (SRC)
            ch1_data_problems_src.ipynb         (ex:src)
        solutions/                              (SOL)
            problem_prove_mean.py               (ex:sol)
    notebooks/                                  (B2) 
        14_data_problems.ipynb                  (ex:probnb2)




Processing steps:
- replace attachment/ image includes with URLs pointing to raw.github..

