# ALI Assignment 1

This project contains the code to recreate the plots I put in my ALI assignment 1 titled $40 K 4^2 S_{1/2}$.

## Installation

The dependencies for this project are only NumPy, SciPy, and Matplotlib, so you likely already have an environment on your machine that can run this code. If not, you can use the pyproject.toml in this repo with UV to run this code. Once you have UV installed on your machine simply run:

```
uv sync
```

and UV will create a venv for you.

**! Note** that the plotting in this project uses TeX to format annotations and labels. If you want to reproduce the plots in the report you *must* have LaTeX installed on your local machine, then change the variable `USE_TEX` on line 17 of S2.py to `True`.

## Running

The main code lives in `s2.py`, if you have your own managed environment, you just need to do your environment's equivalent of:

```
python s2.py
```

If you installed the UV environent, just run:

```
uv run python s2.py
```