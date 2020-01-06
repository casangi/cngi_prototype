# CASA Next Generation Infrastructure

### Documentation
API and User Manual: https://cngi-prototype.readthedocs.io

### Organization
CNGI is organized in to modules as described below. Each module is
responsible for a different functional area, such as file conversion,
input / output, MS operations, and Image operations.  

- conversion
- dio
- ms
- images
- direct
- gridding

### Installation Requirements
Installing the CNGI prototype currently only works on Linux systems with Python 3.6 (due to the CASA 6 requirement for the measurement set conversion functions). 
This dependency on CASA 6 and will be removed in the future. It suggested that a Python virtual environment is created using either Conda or pip.

### Installation

```sh
conda create -n cngi python=3.6
conda activate cngi
#(maybe) sudo apt-get install libgfortran3
pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-group/simple casatools
pip install cngi-prototype
```

### Installation from Source

```sh
git clone https://github.com/casangi/cngi_prototype.git
cd cngi-prototype
conda create -n cngi python=3.6
conda activate cngi
pip install -e .
```

###  Building Documentation from Source
Navigate to the cngi_prototype directory.

```sh
conda activate cngi
pip install sphinx-automodapi
pip install sphinx_rtd_theme
sphinx-build -b html ./docs/ ./_build/
```

### Running CNGI in Parallel using Dask.distributed
To avoid thread collisions, when using the Dask.distributed Client, set the following environment variables.

```sh
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 
```

### Usage
You can import things several different ways.  For example:
```python
>>> from cngi import dio
>>> df = dio.read_pq(...)
```
or
```python
>>> from cngi.dio import read_pq
>>> df = read_pq(...)
```
or
```python
>>> import cngi.dio as cdio
>>> df = cdio.read_pq(...)
```

### Run Tests

Dowload the unit test data from https://astrocloud.nrao.edu/s/Hacr42aZmJ3eb7i and place the files in `cngi_prototype/cngi/data/`.
Unit tests can be found in `cngi_prototype/tests/`. 

### Coding Standards

Documentation is generated using Sphinx, with the autodoc and napoleon extensions enabled. Function docstrings should be written in [NumPy style](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#google-vs-numpy). For compatibility with Sphinx, import statements should generally be underneath function definitions, not at the top of the file.

A complete set of formal and enforced coding standards have not yet been formally adopted. Some alternatives under consideration are:

* Google's [style guide](https://google.github.io/styleguide/pyguide.html)
* Python Software Foundation's [style guide](https://www.python.org/dev/peps/pep-008/)
* Following conventions established by PyData projects (examples [one](https://docs.dask.org/en/latest/develop/html) and [two](https://xarray.pydata.org/en/stable/contributing.html#code-standards))

We are evaluating the adoption of [PEP 484](https://www.python.org/dev/peps/pep-0484/) convention, [myp](http://mypy-lang.org/), or  [param](https://param.holoviz.org) for type-checking, and [flake8](http://flake8.pycqa.org/en/latest/index.html) or [pylint](https://www.pylint.org/) for enforcement.

