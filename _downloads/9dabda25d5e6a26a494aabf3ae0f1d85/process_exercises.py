#!/usr/bin/env python
"""
Process exercises notebooks in the src/ folder to produce the student and solutions notebooks.

Folder structure:
    exercises/
        src/
            sec11_name_src.ipynb            source notebook (combines student prompts and solutions) 
            attachments/
                my_figure.png               image served via raw.github 
        sec11_name.ipynb                    student notebook
        solutions/
            sec11_name_solutions.ipynb      solutions notebook
"""
import nbformat
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
EXERCISES_DIR = "exercises/"
EXERCISES_SRC_DIR = "exercises/src/"
EXERCISES_SOLUTIONS_DIR = "exercises/solutions/"

GITHUB_RAW_TREE_URL = "https://raw.githubusercontent.com/minireference/noBSstats/main"
SRC_ATT_BASE = GITHUB_RAW_TREE_URL + "/exercises/src/attachments/"


def has_studentprompt(cell):
    """
    Return True if cell is marked as containing a student prompt.
    """
    cell_text = cell["source"].replace(" ", "").lower()
    return cell_text.startswith("#@studentprompt")


def has_solution(cell):
    """
    Return True if cell's first line indicates it contains an exercise solution.
    """
    if cell.cell_type == 'code':
        cell_text = cell["source"].replace(" ", "").lower()
        return cell_text.startswith("#@titlesolution") or cell_text.startswith("#@solution")
    elif cell.cell_type == 'markdown':
        # Check for <!-- @solution --> on first line of markdown source
        lines = cell["source"].splitlines()
        first_line = lines[0] if lines else ""
        solution_pat = re.compile(r"<!--\s*@solution\b.*?-->", re.IGNORECASE)
        return bool(solution_pat.search(first_line))
    else:
        return False

def has_solution_tag(cell):
    tags = cell.get("metadata", {}).get("tags", [])
    return "solution" in tags or "@solution" in tags


def rewrite_attachments_links(cell):
    if cell.cell_type == 'markdown':
        cell_source = cell["source"]
        attachments_pat = r'(?:\./)?attachments/'
        if re.search(attachments_pat, cell_source):
            updated_source = re.sub(attachments_pat, SRC_ATT_BASE, cell_source)
            cell["source"] = updated_source


def process_exercises_notebook(src_filepath: str, dest_filepath: str, version: str):
    """
    Load the source notebook `src_filepath` and process the content:
    - rewrite attachment/ images to images URLs hosted on github
    - when version == "student":
      - leave `@studentprompt` cells and remove solutions cells
    - when version == "solutions":
      - leave solutions cells and remove `@studentprompt` cells
    """
    if version not in ["student", "solutions"]:
        raise ValueError(f"Unknown version: {version}; use `student` or `solutions`")
    with open(src_filepath, 'r', encoding='utf-8') as inf:
        nb = nbformat.read(inf, as_version=4)

    new_cells = []    
    previous_cell = None

    # PRODUCE STUDENT NOTEBOOK
    #################################################################################
    if version == "student":

        for cell in nb.cells:    
            # rewrite ./attachment/ images links as absolute URLs
            rewrite_attachments_links(cell)  

            # cleanup @studentprompt comment from first line
            if has_studentprompt(cell):
                cell_source = cell["source"]
                lines_after_first = cell_source.splitlines()[1:]
                cell['source'] = "\n".join(lines_after_first)
                tags = cell.metadata.setdefault("tags", [])
                tags.append("studentprompt")

            # remove solutions
            if has_solution(cell) or has_solution_tag(cell):

                # clear cell source
                cell['source'] = ""

                # clear cell outputs
                if cell.cell_type == 'code':    
                    if "outputId" in cell["metadata"]:
                        del cell["metadata"]["outputId"]
                    if "outputs" in cell:
                        cell["outputs"] = []
                    if "execution_count" in cell:
                        cell["execution_count"] = None

                # minimize empty cells
                if previous_cell:
                    # avoid multiple empty cell
                    if previous_cell["source"] == "":
                        continue
                    # avoid empty cell after student prompt
                    previous_cell_tags = previous_cell.get("metadata", {}).get("tags", [])
                    if "studentprompt" in previous_cell_tags:
                        continue

            new_cells.append(cell)
            previous_cell = cell
    

    # PRODUCE SOLUTIONS NOTEBOOK
    #################################################################################
    elif version == "solutions":

        for cell in nb.cells:    
            # rewrite ./attachment/ images links as absolute URLs
            rewrite_attachments_links(cell)  

            # skip studentprompt cells
            if has_studentprompt(cell):
                continue

            # cleanup @solution comment from first line
            if has_solution(cell):
                cell_source = cell["source"]
                lines_after_first = cell_source.splitlines()[1:]
                cell['source'] = "\n".join(lines_after_first)

            new_cells.append(cell)

    nb.cells = new_cells

    # Write the processed output to `dest_filepath`
    os.makedirs(os.path.dirname(dest_filepath), exist_ok=True)
    with open(dest_filepath, 'w', encoding='utf-8') as outf:
        nbformat.write(nb, outf)


def find_exercises_notebooks(src_dir: str):
    all_files = os.listdir(src_dir)
    src_nbs = [name for name in all_files if name.endswith("_src.ipynb")]
    exercies_notebooks = {}
    for src_nb in src_nbs:
        sec_name = src_nb.split("_", 1)[0]
        student_nb = src_nb.replace("_src.ipynb", ".ipynb")
        solutions_nb = src_nb.replace("_src.ipynb", "_solutions.ipynb")
        sec_nbs = dict(src_nb=src_nb,
                       student_nb=student_nb,
                       solutions_nb=solutions_nb)
        exercies_notebooks[sec_name] = sec_nbs
    return exercies_notebooks


if __name__ == "__main__":
    print("Processing exercises solutions files...")
    src_dir = os.path.join(PROJECT_DIR, EXERCISES_SRC_DIR)
    exercises_notebooks = find_exercises_notebooks(src_dir)
    for sec_name, sec_filenames in exercises_notebooks.items():
        print("Processing the", sec_name, "exercises source file", sec_filenames["src_nb"])
        print("  Generating the student version", sec_filenames["student_nb"])
        src_filepath = os.path.join(PROJECT_DIR, EXERCISES_SRC_DIR, sec_filenames["src_nb"])
        student_filepath = os.path.join(PROJECT_DIR, EXERCISES_DIR, sec_filenames["student_nb"])
        process_exercises_notebook(src_filepath, student_filepath, version="student")

        print("  Generating the solutions version", sec_filenames["solutions_nb"])
        solutions_filepath = os.path.join(PROJECT_DIR, EXERCISES_SOLUTIONS_DIR, sec_filenames["solutions_nb"])
        process_exercises_notebook(src_filepath, solutions_filepath, version="solutions")

    print("DONE")

     
