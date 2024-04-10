#!/usr/bin/env python
"""
Process tutorials

Folder structure:
    tutorials/
        src/
            python_tutorial.ipynb
            pandas_tutorial.ipynb
            seaborn_tutorial.ipynb
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


PROJECT_DIR = "/Users/ivan/Projects/Minireference/STATSbook/noBSstatsnotebooks/"
SOLUTIONS_DIR = "tutorials/solutions/python/"
SOLUTIONS_PATH = os.path.join(PROJECT_DIR, SOLUTIONS_DIR)
GITHUB_TREE_URL = "https://github.com/minireference/noBSstats/blob/main"
GITHUB_RAW_TREE_URL = "https://raw.githubusercontent.com/minireference/noBSstats/main"

def has_solution(cell):
    """Return True if cell is marked as containing an exercise solution."""
    cell_text = cell["source"].replace(" ", "").lower()
    first_line = cell_text.split("\n")[0]
    return (
        cell_text.startswith("#@titlesolution")
        or "to_remove" in first_line
    )


def clear_solutions_dir():
    solutions_dir = SOLUTIONS_PATH
    solutions_py_pat = os.path.join(solutions_dir, '*.py')
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


def split_off_solutions(srcfilename: str, destfilename: str):
    """
    Load the notebook, adjust headings in markdown cells, and write back.
    """
    with open(srcfilename, 'r', encoding='utf-8') as inf:
        nb = nbformat.read(inf, as_version=4)

    for cell in nb.cells:
        # rewrite ./attachment/ relative images links as absolute URLs
        rewrite_attachment_links(cell)

        # split_off_solutions
        if cell.cell_type == 'code' and has_solution(cell):
            cell_source = cell["source"]
            cell_source = cell_source.replace("@titlesolution", "")
            title = cell_source.splitlines()[0].strip()
            filename = title.replace("# ","").replace(" ","_") + ".py"
            with open(os.path.join(SOLUTIONS_PATH, filename), "w") as solf:
                solf.write(cell_source)

            # clear cell outputs
            if "outputID" in cell["metadata"]:
                del cell["metadata"]["outputId"]
            if "outputs" in cell:
                cell["outputs"] = []
            if "execution_count" in cell:
                del cell["execution_count"]

            # set cell content to markdown link to the solution .py
            cell["cell_type"] = "markdown"
            py_url = f"{GITHUB_TREE_URL}/tutorials/solutions/python/{filename}"
            new_source = f"[*Click for solution*]({py_url})\n\n"
            cell['source'] = new_source

    with open(destfilename, 'w', encoding='utf-8') as outf:
        nbformat.write(nb, outf)


if __name__ == "__main__":
    srcfilename = os.path.join(PROJECT_DIR, "tutorials/src/python_tutorial.ipynb")
    destfilename = os.path.join(PROJECT_DIR, "tutorials/python_tutorial.ipynb")
    print('Processing', srcfilename)
    clear_solutions_dir()
    split_off_solutions(srcfilename, destfilename)
