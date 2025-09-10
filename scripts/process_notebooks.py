"""Process tutorials for Neuromatch Academy

- Filter input file list for .ipynb files
- Check that the cells have been executed sequentially on a fresh kernel
- Strip trailing whitespace from all code lines
- Either:
  - Execute the notebook and fail if errors are encountered (apart from the `NotImplementedError`)
  - Check that all code cells have been executed without error
- Extract solution code and write a .py file with the solution
- Create the student version by replacing solution cells with a "hint" image and a link to the solution code
- Create the instructor version by replacing cells with code exercises with text cells with code in markdown form.
- Redirect Colab-inserted badges to the main branch
- Set the Colab notebook name field based on file path
- Standardize some Colab settings (always have ToC, always hide form cells)
- Clean the notebooks (remove outputs and noisy metadata)
- Write the executed version of the input notebook to its original path
- Write the post-processed notebook to a student/ subdirectory
- Write solution images to a static/ subdirectory
- Write solution code to a solutions/ subdirectory


via https://github.com/NeuromatchAcademy/nmaci/blob/main/scripts/process_notebooks.py
LICENSE: BSD 3-Clause License, Copyright (c) 2021, Neuromatch Academy.
"""
import os
import re
import sys
import argparse
import hashlib
from io import BytesIO
from binascii import a2b_base64
from copy import deepcopy

from PIL import Image
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

REPO = os.environ.get("NMA_REPO", "course-content")
MAIN_BRANCH = os.environ.get("NMA_MAIN_BRANCH", "main")

GITHUB_RAW_URL = (
    f"https://raw.githubusercontent.com/NeuromatchAcademy/{REPO}/{MAIN_BRANCH}"
)
GITHUB_TREE_URL = (
    f"https://github.com/NeuromatchAcademy/{REPO}/tree/{MAIN_BRANCH}/"
)


def main(arglist):
    """Process IPython notebooks from a list of files."""
    args = parse_args(arglist)

    # Filter paths from the git manifest
    # - Only process .ipynb
    # - Don't process student notebooks
    # - Don't process deleted notebooks (which are paths in the git manifest)
    def should_process(path):
        return all([
            path.endswith(".ipynb"),
            "student/" not in path,
            "instructor/" not in path,
            os.path.isfile(path),
        ])

    nb_paths = [arg for arg in args.files if should_process(arg)]
    if not nb_paths:
        print("No notebook files found")
        sys.exit(0)

    # Set execution parameters. We allow NotImplementedError as that is raised
    # by incomplete exercises and is unlikely to be otherwise encountered.
    exec_kws = {"timeout": 14400, "allow_error_names": ["NotImplementedError"]}

    # Allow environment to override stored kernel name
    if "NB_KERNEL" in os.environ:
        exec_kws["kernel_name"] = os.environ["NB_KERNEL"]

    # Defer failures until after processing all notebooks
    notebooks = {}
    errors = {}

    for nb_path in nb_paths:

        # Load the notebook structure
        with open(nb_path) as f:
            nb = nbformat.read(f, nbformat.NO_CONVERT)

        if not sequentially_executed(nb):
            if args.require_sequential:
                err = (
                    "Notebook is not sequentially executed on a fresh kernel."
                    "\n"
                    "Please do 'Restart and run all' before pushing to Github."
                )
                errors[nb_path] = err
                continue

        # Clean whitespace from all code cells
        clean_whitespace(nb)

        # Ensure that we have an executed notebook, in one of two ways
        executor = ExecutePreprocessor(**exec_kws)
        if args.execute:
            # Check dynamically by executing and reporting errors
            print(f"Executing {nb_path}")
            error = execute_notebook(executor, nb, args.raise_fast)
        elif args.check_execution:
            # Check statically by examining the cell outputs
            print(f"Checking {nb_path} execution")
            error = check_execution(executor, nb, args.raise_fast)
        else:
            error = None

        if error is None:
            notebooks[nb_path] = nb
        else:
            errors[nb_path] = error

    if errors or args.check_only:
        exit(errors)

    # Post-process notebooks
    for nb_path, nb in notebooks.items():

        # Extract components of the notebook path
        nb_dir, nb_fname = os.path.split(nb_path)
        nb_name, _ = os.path.splitext(nb_fname)

        # Loop through the cells and fix any Colab badges we encounter
        for cell in nb.get("cells", []):
            if has_colab_badge(cell):
                redirect_colab_badge_to_main_branch(cell)
                # add kaggle badge
                add_kaggle_badge(cell, nb_path)

        # Ensure that Colab metadata dict exists and enforce some settings
        add_colab_metadata(nb, nb_name)

        # Write the original notebook back to disk, clearing outputs only for tutorials
        print(f"Writing complete notebook to {nb_path}")
        with open(nb_path, "w") as f:
            nb_clean = clean_notebook(nb, clear_outputs=nb_path.startswith("tutorials"))
            nbformat.write(nb_clean, f)

        # if the notebook is not in tutorials, skip the creation/update of the student, static, solutions directories
        if not nb_path.startswith("tutorials"):
          continue

        # Create subdirectories, if they don't exist
        student_dir = make_sub_dir(nb_dir, "student")
        static_dir = make_sub_dir(nb_dir, "static")
        solutions_dir = make_sub_dir(nb_dir, "solutions")
        instructor_dir = make_sub_dir(nb_dir, "instructor")

        # Generate the student version and save it to a subdirectory
        print(f"Extracting solutions from {nb_path}")
        processed = extract_solutions(nb, nb_dir, nb_name)
        student_nb, static_images, solution_snippets = processed
        
        # Generate the instructor version and save it to a subdirectory
        print(f"Create instructor notebook from {nb_path}")
        instructor_nb = instructor_version(nb, nb_dir, nb_name)

        # Loop through cells and point the colab badge at the student version
        for cell in student_nb.get("cells", []):
            if has_colab_badge(cell):
                redirect_colab_badge_to_student_version(cell)
                # add kaggle badge
                add_kaggle_badge(cell, nb_path)

        # Loop through cells and point the colab badge at the instructor version
        for cell in instructor_nb.get("cells", []):
            if has_colab_badge(cell):
                redirect_colab_badge_to_instructor_version(cell)
                # add kaggle badge
                add_kaggle_badge(cell, nb_path)

        # Write the student version of the notebook
        student_nb_path = os.path.join(student_dir, nb_fname)
        print(f"Writing student notebook to {student_nb_path}")
        with open(student_nb_path, "w") as f:
            clean_student_nb = clean_notebook(student_nb)
            nbformat.write(clean_student_nb, f)

        # Write the images extracted from the solution cells
        print(f"Writing solution images to {static_dir}")
        for fname, image in static_images.items():
            fname = fname.replace("static", static_dir)
            image.save(fname)

        # Write the solution snippets
        print(f"Writing solution snippets to {solutions_dir}")
        for fname, snippet in solution_snippets.items():
            fname = fname.replace("solutions", solutions_dir)
            with open(fname, "w") as f:
                f.write(snippet)

        # Write the instructor version of the notebook
        instructor_nb_path = os.path.join(instructor_dir, nb_fname)
        print(f"Writing instructor notebook to {instructor_nb_path}")
        with open(instructor_nb_path, "w") as f:
            clean_instructor_nb = clean_notebook(instructor_nb)
            nbformat.write(clean_instructor_nb, f)

    exit(errors)


# ------------------------------------------------------------------------------------ #

def execute_notebook(executor, nb, raise_fast):
    """Execute the notebook, returning errors to be handled."""
    try:
        executor.preprocess(nb)
    except Exception as error:
        if raise_fast:
            # Exit here (useful for debugging)
            raise error
        else:
            # Raise the error to be handled by the caller
            return error


def check_execution(executor, nb, raise_fast):
    """Check that all code cells with source have been executed without error."""
    error = None
    for cell in nb.get("cells", []):

        # Only check code cells
        if cell["cell_type"] != "code":
            continue

        if cell["source"] and cell["execution_count"] is None:
            error = "Notebook has unexecuted code cell(s)."
            if raise_fast:
                raise RuntimeError(error)
            break
        else:
            for output in cell["outputs"]:
                if output["output_type"] == "error":
                    if output["ename"] in executor.allow_error_names:
                        continue
                    error = "\n".join(output["traceback"])
                    if raise_fast:
                        raise RuntimeError("\n" + error)
                    break

    return error


def extract_solutions(nb, nb_dir, nb_name):
    """Convert solution cells to markdown; embed images from Python output."""
    nb = deepcopy(nb)
    _, tutorial_dir = os.path.split(nb_dir)

    static_images = {}
    solution_snippets = {}

    nb_cells = nb.get("cells", [])
    for i, cell in enumerate(nb_cells):

        if has_solution(cell):

            # Get the cell source
            cell_source = cell["source"]

            # Hash the source to get a unique identifier
            cell_id = hashlib.sha1(cell_source.encode("utf-8")).hexdigest()[:8]

            # Extract image data from the cell outputs
            cell_images = {}
            for j, output in enumerate(cell.get("outputs", [])):

                fname = f"static/{nb_name}_Solution_{cell_id}_{j}.png"
                try:
                    image_data = a2b_base64(output["data"]["image/png"])
                except KeyError:
                    continue
                cell_images[fname] = Image.open(BytesIO(image_data))
            static_images.update(cell_images)

            # Clean up the cell source and assign a filename
            snippet = "\n".join(cell_source.split("\n")[1:])
            py_fname = f"solutions/{nb_name}_Solution_{cell_id}.py"
            solution_snippets[py_fname] = snippet

            # Convert the solution cell to markdown,
            # Insert a link to the solution snippet script on github,
            # and embed the image as a link to static file (also on github)
            py_url = f"{GITHUB_TREE_URL}/tutorials/{tutorial_dir}/{py_fname}"
            new_source = f"[*Click for solution*]({py_url})\n\n"

            if cell_images:
                new_source += "*Example output:*\n\n"
                for f, img in cell_images.items():

                    url = f"{GITHUB_RAW_URL}/tutorials/{tutorial_dir}/{f}"

                    # Handle matplotlib retina mode
                    dpi_w, dpi_h = img.info["dpi"]
                    w = img.width // (dpi_w // 72)
                    h = img.height // (dpi_h // 72)

                    tag_args = " ".join([
                        "alt='Solution hint'",
                        "align='left'",
                        f"width={w}",
                        f"height={h}",
                        f"src={url}",
                    ])
                    new_source += f"<img {tag_args}>\n\n"

            cell["source"] = new_source
            cell["cell_type"] = "markdown"
            cell["metadata"]["colab_type"] = "text"
            if "outputID" in cell["metadata"]:
                del cell["metadata"]["outputId"]
            if "outputs" in cell:
                del cell["outputs"]
            if "execution_count" in cell:
                del cell["execution_count"]

    return nb, static_images, solution_snippets


def instructor_version(nb, nb_dir, nb_name):
    """Convert notebook to instructor notebook."""
    nb = deepcopy(nb)
    _, tutorial_dir = os.path.split(nb_dir)

    nb_cells = nb.get("cells", [])
    for i, cell in enumerate(nb_cells):

        if has_code_exercise(cell):
            if nb_cells[i-1]["cell_type"] == "markdown":
                cell_id = i-2
            else:
                cell_id = i-1
            nb_cells[cell_id]["cell_type"] = "markdown"
            nb_cells[cell_id]["metadata"]["colab_type"] = "text"
            if "outputID" in nb_cells[cell_id]["metadata"]:
                del nb_cells[cell_id]["metadata"]["outputId"]
            if "outputs" in nb_cells[cell_id]:
                del nb_cells[cell_id]["outputs"]
            if "execution_count" in nb_cells[cell_id]:
                del nb_cells[cell_id]["execution_count"]

            nb_cells[cell_id]['source'] = '```python\n' + nb_cells[cell_id]['source']+'\n\n```'

    return nb


def clean_notebook(nb, clear_outputs=True):
    """Remove cell outputs and most unimportant metadata."""
    # Always operate on a copy of the input notebook
    nb = deepcopy(nb)

    # Remove some noisy metadata
    nb.metadata.pop("widgets", None)

    # Set kernel to default Python3
    nb.metadata["kernel"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }

    # Iterate through the cells and clean up each one
    for cell in nb.get("cells", []):

        # Remove blank cells
        if not cell["source"]:
            nb.cells.remove(cell)
            continue

        # Reset cell-level Jupyter metadata
        for key in ["prompt_number", "execution_count"]:
            if key in cell:
                cell[key] = None

        if "metadata" in cell:
            cell.metadata["execution"] = {}
            for field in ["colab", "collapsed", "scrolled", "ExecuteTime", "outputId"]:
                cell.metadata.pop(field, None)

        # Reset cell-level Colab metadata
        if "id" in cell["metadata"]:
            if not cell["metadata"]["id"].startswith("view-in"):
                cell["metadata"].pop("id")

        if cell["cell_type"] == "code":
            # Remove code cell outputs if requested
            if clear_outputs:
                cell["outputs"] = []

            # Ensure that form cells are hidden by default
            first_line, *_ = cell["source"].splitlines()
            if "@title" in first_line or "@markdown" in first_line:
                cell["metadata"]["cellView"] = "form"

    return nb


def add_colab_metadata(nb, nb_name):
    """Ensure that notebook has Colab metadata and enforce some settings."""
    if "colab" not in nb["metadata"]:
        nb["metadata"]["colab"] = {}

    # Always overwrite the name and show the ToC/Colab button
    nb["metadata"]["colab"].update({
        "name": nb_name,
        "toc_visible": True,
        "include_colab_link": True,
    })

    # Allow collapsed sections, but default to not having any
    nb["metadata"]["colab"].setdefault("collapsed_sections", [])


def clean_whitespace(nb):
    """Remove trailing whitespace from all code cell lines."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type", "") == "code":
            source_lines = cell["source"].splitlines()
            clean_lines = [line.rstrip() for line in source_lines]
            cell["source"] = "\n".join(clean_lines)


def test_clean_whitespace():

    nb = {
        "cells": [
            {"cell_type": "code", "source": "import numpy  \nimport matplotlib   "},
            {"cell_type": "markdown", "source": "# Test notebook  "},
        ]
    }
    clean_whitespace(nb)
    assert nb["cells"][0]["source"] == "import numpy\nimport matplotlib"
    assert nb["cells"][1]["source"] == "# Test notebook  "


def has_solution(cell):
    """Return True if cell is marked as containing an exercise solution."""
    cell_text = cell["source"].replace(" ", "").lower()
    first_line = cell_text.split("\n")[0]
    return (
        cell_text.startswith("#@titlesolution")
        or "to_remove" in first_line
    )


def has_code_exercise(cell):
    """Return True if cell is marked as containing an exercise solution."""
    cell_text = cell["source"].replace(" ", "").lower()
    first_line = cell_text.split("\n")[0]
    return (
        cell_text.startswith("#@titlesolution")
        or "to_removesolution" in first_line
    )


def test_has_solution():

    cell = {"source": "# solution"}
    assert not has_solution(cell)

    cell = {"source": "def exercise():\n    pass\n# to_remove"}
    assert not has_solution(cell)

    cell = {"source": "# to_remove_solution\ndef exercise():\n    pass"}
    assert has_solution(cell)


def has_colab_badge(cell):
    """Return True if cell has a Colab badge as an HTML element."""
    return "colab-badge.svg" in cell["source"]


def test_has_colab_badge():

    cell = {
        "source": "import numpy as np"
    }
    assert not has_colab_badge(cell)

    cell = {
        "source":
        "<img src=\"https://colab.research.google.com/assets/colab-badge.svg\" "
    }
    assert has_colab_badge(cell)


def redirect_colab_badge_to_main_branch(cell):
    """Modify the Colab badge to point at the main branch on Github."""
    cell_text = cell["source"]
    p = re.compile(r"^(.+/NeuromatchAcademy/" + REPO + r"/blob/)[\w-]+(/.+$)")
    cell["source"] = p.sub(r"\1" + MAIN_BRANCH + r"\2", cell_text)


def test_redirect_colab_badge_to_main_branch():

    original = (
        "\"https://colab.research.google.com/github/NeuromatchAcademy/"
        "course-content/blob/W1D1-updates/tutorials/W1D1_ModelTypes/"
        "W1D1_Tutorial1.ipynb\""
    )
    cell = {"source": original}
    redirect_colab_badge_to_main_branch(cell)

    expected = (
        "\"https://colab.research.google.com/github/NeuromatchAcademy/"
        "course-content/blob/main/tutorials/W1D1_ModelTypes/"
        "W1D1_Tutorial1.ipynb\""
    )

    assert cell["source"] == expected


def redirect_colab_badge_to_student_version(cell):
    """Modify the Colab badge to point at student version of the notebook."""
    cell_text = cell["source"]
    # redirect the colab badge
    p = re.compile(r"(^.+blob/" + MAIN_BRANCH + r"/tutorials/W\dD\d\w+)/(\w+\.ipynb.+)")
    cell_text = p.sub(r"\1/student/\2", cell_text)
    # redirect the kaggle badge
    p = re.compile(r"(^.+/tutorials/W\dD\d\w+)/(\w+\.ipynb.+)")
    cell["source"] = p.sub(r"\1/student/\2", cell_text)


def redirect_colab_badge_to_instructor_version(cell):
    """Modify the Colab badge to point at instructor version of the notebook."""
    cell_text = cell["source"]
    # redirect the colab badge
    p = re.compile(r"(^.+blob/" + MAIN_BRANCH + r"/tutorials/W\dD\d\w+)/(\w+\.ipynb.+)")
    cell_text = p.sub(r"\1/instructor/\2", cell_text)
    # redirect the kaggle badge
    p = re.compile(r"(^.+/tutorials/W\dD\d\w+)/(\w+\.ipynb.+)")
    cell["source"] = p.sub(r"\1/instructor/\2", cell_text)


def test_redirect_colab_badge_to_student_version():

    original = (
        "\"https://colab.research.google.com/github/NeuromatchAcademy/"
        "course-content/blob/main/tutorials/W1D1_ModelTypes/"
        "W1D1_Tutorial1.ipynb\""
    )

    cell = {"source": original}
    redirect_colab_badge_to_student_version(cell)

    expected = (
        "\"https://colab.research.google.com/github/NeuromatchAcademy/"
        "course-content/blob/main/tutorials/W1D1_ModelTypes/student/"
        "W1D1_Tutorial1.ipynb\""
    )

    assert cell["source"] == expected

def add_kaggle_badge(cell, nb_path):
    """Add a kaggle badge if not exists."""
    cell_text = cell["source"]
    if "kaggle" not in cell_text:
        badge_link = "https://kaggle.com/static/images/open-in-kaggle.svg"
        service = "https://kaggle.com/kernels/welcome?src="
        alter = "Open in Kaggle"
        basic_url = "https://raw.githubusercontent.com/NeuromatchAcademy"
        a = f'<a href=\"{service}{basic_url}/{REPO}/{MAIN_BRANCH}/{nb_path}\" target=\"_parent\"><img src=\"{badge_link}\" alt=\"{alter}\"/></a>'
        cell["source"] += f' &nbsp; {a}'

def sequentially_executed(nb):
    """Return True if notebook appears freshly executed from top-to-bottom."""
    exec_counts = [
        cell["execution_count"]
        for cell in nb.get("cells", [])
        if (
            cell["source"]
            and cell.get("execution_count", None) is not None
        )
    ]
    sequential_counts = list(range(1, 1 + len(exec_counts)))
    # Returns True if there are no executed code cells, which is fine?
    return exec_counts == sequential_counts


def make_sub_dir(nb_dir, name):
    """Create nb_dir/name if it does not exist."""
    sub_dir = os.path.join(nb_dir, name)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    return sub_dir


def exit(errors):
    """Exit with message and status dependent on contents of errors dict."""
    for failed_file, error in errors.items():
        print(f"{failed_file} failed quality control.", file=sys.stderr)
        print(error, file=sys.stderr)

    status = bool(errors)
    report = "Failure" if status else "Success"
    print("=" * 30, report, "=" * 30)
    sys.exit(status)


def parse_args(arglist):
    """Handle the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process neuromatch tutorial notebooks",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="File name(s) to process. Will filter for .ipynb extension."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the notebook and fail if errors are encountered."
    )
    parser.add_argument(
        "--check-execution",
        action="store_true",
        dest="check_execution",
        help="Check that each code cell has been executed and did not error."
    )
    parser.add_argument(
        "--allow-non-sequential",
        action="store_false",
        dest="require_sequential",
        help="Don't fail if the notebook is not sequentially executed."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        dest="check_only",
        help="Only run QC checks; don't do post-processing."
    )
    parser.add_argument(
        "--raise-fast",
        action="store_true",
        dest="raise_fast",
        help="Raise errors immediately rather than collecting and reporting."
    )
    return parser.parse_args(arglist)


if __name__ == "__main__":

    main(sys.argv[1:])
