#!/usr/bin/env python
"""
Process problems notebooks in the src/ folder to produce the student and solutions notebooks.

Folder structure:

    problems/
        src/
            ch1_name_src.ipynb            source notebook (combines student prompts and solutions) 
            attachments/
                my_figure.png             image served via raw.github 
        ch1_name.ipynb                    student notebook
        solutions/
            ch1_name_solutions.ipynb      solutions notebook
"""
import nbformat
import os
import re


PROJECT_DIR = "/Users/ivan/Projects/Minireference/STATSbook/noBSstats/"
PROBLEMS_DIR = "problems/"
PROBLEMS_SRC_DIR = "problems/src/"
PROBLEMS_SOLUTIONS_DIR = "problems/solutions/"

GITHUB_RAW_TREE_URL = "https://raw.githubusercontent.com/minireference/noBSstats/main"
SRC_ATT_BASE = GITHUB_RAW_TREE_URL + "/problems/src/attachments/"


def has_problem_start(cell):
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
    Return True if cell's first line indicates it contains an problem solution.
    """
    if cell.cell_type == 'code':
        cell_text = cell["source"].replace(" ", "").lower()
        return cell_text.startswith("#@titlesolution") or cell_text.startswith("#@solution")
    elif cell.cell_type == 'markdown':
        # Check for <!-- @solution --> in markdown source
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
    if cell.cell_type == 'markdown':
        cell_source = cell["source"]
        attachments_pat = r'(?:\./)?attachments/'
        if re.search(attachments_pat, cell_source):
            updated_source = re.sub(attachments_pat, SRC_ATT_BASE, cell_source)
            cell["source"] = updated_source


def process_problems_notebook(src_filepath: str, dest_filepath: str, version: str):
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
    with open(dest_filepath, 'w', encoding='utf-8') as outf:
        nbformat.write(nb, outf)


def find_problems_notebooks(src_dir: str):
    all_files = os.listdir(src_dir)
    src_nbs = [name for name in all_files if name.endswith("_src.ipynb")]
    exercies_notebooks = {}
    for src_nb in src_nbs:
        ch_name = src_nb.split("_", 1)[0]
        student_nb = src_nb.replace("_src.ipynb", ".ipynb")
        solutions_nb = src_nb.replace("_src.ipynb", "_solutions.ipynb")
        ch_nbs = dict(src_nb=src_nb,
                      student_nb=student_nb,
                      solutions_nb=solutions_nb)
        exercies_notebooks[ch_name] = ch_nbs
    return exercies_notebooks



if __name__ == "__main__":
    print("Processing problems solutions files...")
    # src_dir = os.path.join(PROJECT_DIR, PROBLEMS_SRC_DIR)
    # problems_notebooks = find_problems_notebooks(src_dir)
    # for ch_name, ch_filenames in problems_notebooks.items():
    ch_name = "ch1"
    ch_filenames = {'src_nb': 'ch1_data_problems_src.ipynb',
                     'student_nb': 'ch1_data_problems.ipynb',
                     'solutions_nb': 'ch1_data_problems_solutions.ipynb'}

    print("Processing the", ch_name, "problems source file", ch_filenames["src_nb"])

    print("  Generating the student version", ch_filenames["student_nb"])
    src_filepath = os.path.join(PROJECT_DIR, PROBLEMS_SRC_DIR, ch_filenames["src_nb"])
    student_filepath = os.path.join(PROJECT_DIR, PROBLEMS_DIR, ch_filenames["student_nb"])
    process_problems_notebook(src_filepath, student_filepath, version="student")

    print("  Generating the solutions version", ch_filenames["solutions_nb"])
    solutions_filepath = os.path.join(PROJECT_DIR, PROBLEMS_SOLUTIONS_DIR, ch_filenames["solutions_nb"])
    process_problems_notebook(src_filepath, solutions_filepath, version="solutions")

