#!/usr/bin/env python
"""
Process tutorials

Folder structure:
    tutorials/
        src/
            python_tutorial_src.ipynb
            pandas_tutorial_src.ipynb
            seaborn_tutorial_src.ipynb
            attachments/
                my_figrue.png               image served via raw.gihub 
        python_tutorial.ipynb
        pandas_tutorial.ipynb
        seaborn_tutorial.ipynb
        solutions/
            python/
                Exercise_1_something.py
        static/
            94u93u0293u4i0329i409i.png      images served via raw.github ...


Processing steps:
- replace attachment/ image includes with URLs pointing to raw.github..
- output TEX
    - replace URLs with local copy
    - build using IEEEtrans format
- output HTML static
    - clean formatting to use as standalone webapge

# TODO: import 
# def extract_solutions(nb, nb_dir, nb_name):
# from process_notebooks.py
"""
from copy import deepcopy
import glob
import nbformat
import os
import re


PROJECT_DIR = "/Users/ivan/Projects/Minireference/STATSbook/noBSstats/"
SOLUTIONS_DIR = "tutorials/solutions/"
GITHUB_TREE_URL = "https://github.com/minireference/noBSstats/blob/main"
GITHUB_RAW_TREE_URL = "https://raw.githubusercontent.com/minireference/noBSstats/main"


TUTORIALS = {
    "python": { "src": "python_tutorial_src.ipynb",
               "dest": "python_tutorial.ipynb"},
    "pandas": { "src": "pandas_tutorial_src.ipynb",
               "dest": "pandas_tutorial.ipynb"}
    # "seaborn": {}; TODO
}

def has_solution(cell):
    """Return True if cell is marked as containing an exercise solution."""
    cell_text = cell["source"].replace(" ", "").lower()
    first_line = cell_text.split("\n")[0]
    return (
        cell_text.startswith("#@titlesolution")
        or "to_remove" in first_line
    )


def clear_solutions_dir(tutorial):
    solutions_subdir = os.path.join(PROJECT_DIR, SOLUTIONS_DIR, tutorial)
    solutions_py_pat = os.path.join(solutions_subdir, '*.py')
    solutions_py_files = glob.glob(solutions_py_pat)
    for solutions_py_file in solutions_py_files:
        os.remove(solutions_py_file)


def rewrite_attachment_links(cell):
    tut_src_att_url = GITHUB_RAW_TREE_URL + "/tutorials/src/attachments/"
    if cell.cell_type == 'markdown':
        cell_source = cell["source"]
        attachments_pat = r'(?:\./)?attachments/'
        if re.search(attachments_pat, cell_source):
            updated_source = re.sub(attachments_pat, tut_src_att_url, cell_source)
            cell["source"] = updated_source


def process_tutorial_notebook(tutorial: str, src_filepath: str, dest_filepath: str):
    """
    Load the source notebook `src_filepath` and process the content:
    - rewrite attachment/ images to URLs from github images (to make the tutorial independent from )
    - split off solutions
    """
    assert tutorial in TUTORIALS.keys()
    with open(src_filepath, 'r', encoding='utf-8') as inf:
        nb = nbformat.read(inf, as_version=4)

    for cell in nb.cells:
        # rewrite ./attachment/ relative images links as absolute URLs
        rewrite_attachment_links(cell)

        # split off solutions
        if cell.cell_type == 'code' and has_solution(cell):
            cell_source = cell["source"]
            cell_source = cell_source.replace("@titlesolution", "")
            title = cell_source.splitlines()[0].strip()
            filename = title.replace("# ","").replace(" ","_") + ".py"
            solutions_subdir = os.path.join(PROJECT_DIR, SOLUTIONS_DIR, tutorial)
            with open(os.path.join(solutions_subdir, filename), "w") as solf:
                solf.write(cell_source)

            # clear cell outputs
            if "outputID" in cell["metadata"]:
                del cell["metadata"]["outputId"]
            if "outputs" in cell:
                del cell["outputs"]
            if "execution_count" in cell:
                del cell["execution_count"]

            # set cell content to link to the solution .py
            cell["cell_type"] = "markdown"
            py_url = f"{GITHUB_TREE_URL}/tutorials/solutions/{tutorial}/{filename}"
            new_source = f'<a href=\"{py_url}\" target=\"_blank\">Click for solution.</a>\n'
            cell['source'] = new_source

    with open(dest_filepath, 'w', encoding='utf-8') as outf:
        nbformat.write(nb, outf)



if __name__ == "__main__":
    print("Processing tutorials source files...")
    for tutorial, tutorial_filenames in TUTORIALS.items():
        print('Processing the', tutorial, "tutorial...")
        src_filename = tutorial_filenames["src"]
        dest_filename = tutorial_filenames["dest"]        
        src_filepath = os.path.join(PROJECT_DIR, "tutorials/src/", src_filename)
        dest_filepath = os.path.join(PROJECT_DIR, "tutorials", dest_filename)
        clear_solutions_dir(tutorial)
        process_tutorial_notebook(tutorial, src_filepath, dest_filepath)
