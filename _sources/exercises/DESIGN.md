Design of exercise processing steps
===================================

The script [scripts/process_exercises.py](../scripts/process_exercises.py)
takes the source notebooks from the `exercises/src/` folder and produces
the student version in `exercises/` and the solutions version in `exercises/solutions/`.


Folder structure
----------------

    exercises/                                  (B1)
        src/                                    (SRC)
            sec13_descr_stats_src.ipynb         (src_nb)
        sec13_descr_stats.ipynb                 (student_nb)
        solutions/                              (SOL)
            sec13_descr_stats_solutions.ipynb   (solutions_nb)
    notebooks/                                  (B2) 
        13_descriptive_statistics.ipynb         (ex:nb)

The full notebook with solutions lives in `exercises/src/`.
This is where you edit and create the exercises.

where:
- `(SRC)` location for exercises notebooks with solutions
- `(B1)` build output for standalone exercise notebooks
- `(B2)` build output for section notebooks (append to the `## Exercises` block in the notebook)
- `(SOL)` folder for solution notebooks


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

