#!/usr/bin/env python
"""
Process exercise notebooks

Folder structure:

    exercises/
        stc/
            sec11_name_src.ipynb            source notebook (combines student prompts and solutions) 
            attachments/
                my_figure.png               image served via raw.gihub 
        sec11_name.ipynb                    student notebook
        solutions/
            sec11_name_solutions.ipynb      soutions notebook

"""
import nbformat
import os
import re


PROJECT_DIR = "/Users/ivan/Projects/Minireference/STATSbook/noBSstats/"
EXERCISES_DIR = "exercises/"
EXERCISES_SRC_DIR = "exercises/src/"
EXERCISES_SOLUTIONS_DIR = "exercises/solutions/"

GITHUB_RAW_TREE_URL = "https://raw.githubusercontent.com/minireference/noBSstats/main"



def has_exercise_start(cell):
    """
    Return True if cell contains a heading of the form ### E?.??
    TODO: extend to handle P?.?? as well
    """
    cell_text = cell["source"].replace(" ", "").lower()
    return cell_text.startswith("#@studentprompt")


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
        # Check one: <!-- @solution --> in markdown source
        solution_pat = re.compile(r"<!--\s*@solution\b[\s\S]*?-->", re.MULTILINE)
        if solution_pat.search(cell["source"].lower()):
            return True
        else:
            return False        
    else:
        return False

def has_solution_tag(cell):
    tags = cell.get("metadata", {}).get("tags", [])
    return "solution" in tags or "@solution" in tags



def rewrite_attachments_links(cell):
    tut_src_att_url = GITHUB_RAW_TREE_URL + "/exercises/src/attachments/"
    if cell.cell_type == 'markdown':
        cell_source = cell["source"]
        attachments_pat = r'(?:\./)?attachments/'
        if re.search(attachments_pat, cell_source):
            updated_source = re.sub(attachments_pat, tut_src_att_url, cell_source)
            cell["source"] = updated_source


def process_exercises_notebook(src_filepath: str, dest_filepath: str, version: str):
    """
    Load the source notebook `src_filepath` and process the content:
    - rewrite attachment/ images to images URLs hosted on github
    - when version == "student":
      - leave `@studentprompt` cells and remove solutions cells
    - when version == "student":
      - leave solutions cells and remove `@studentprompt` cells
    """
    assert version in ["student", "solutions"]
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
                    if "outputID" in cell["metadata"]:
                        del cell["metadata"]["outputId"]
                    if "outputs" in cell:
                        cell["outputs"] = []
                    if "execution_count" in cell:
                        del cell["execution_count"]
                
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
    # src_dir = os.path.join(PROJECT_DIR, EXERCISES_SRC_DIR)
    # exercises_notebooks = find_exercises_notebooks(src_dir)
    # for sec_name, sec_filenames in exercises_notebooks.items():
    sec_name = "sec12"
    sec_filenames = {'src_nb': 'sec12_data_in_practice_src.ipynb',
                     'student_nb': 'sec12_data_in_practice.ipynb',
                     'solutions_nb': 'sec12_data_in_practice_solutions.ipynb'}

    print("Processing the", sec_name, "exercises source file", sec_filenames["src_nb"])

    print("  Generating the student version", sec_filenames["student_nb"])
    src_filepath = os.path.join(PROJECT_DIR, EXERCISES_SRC_DIR, sec_filenames["src_nb"])
    student_filepath = os.path.join(PROJECT_DIR, EXERCISES_DIR, sec_filenames["student_nb"])
    process_exercises_notebook(src_filepath, student_filepath, version="student")

    print("  Generating the solutions version", sec_filenames["solutions_nb"])
    solutions_filepath = os.path.join(PROJECT_DIR, EXERCISES_SOLUTIONS_DIR, sec_filenames["solutions_nb"])
    process_exercises_notebook(src_filepath, solutions_filepath, version="solutions")
     
