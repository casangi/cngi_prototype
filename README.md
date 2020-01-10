# README
Installation, setup and development rules for the CASA Next Generation Infrastructure

### Documentation
API and User Manual: https://cngi-prototype.readthedocs.io  
  
Visibility Processing Overview: https://colab.research.google.com/github/casangi/examples/blob/master/MeasurementSet_overview.ipynb  
  
Image Processing Overview: https://colab.research.google.com/github/casangi/examples/blob/master/Image_overview.ipynb  
  

### Organization
CNGI is organized in to modules as described below. Each module is
responsible for a different functional area, such as file conversion,
input / output, Visibility data operations, and Image data operations.  

- conversion
- dio
- vis
- image
- direct
- gridding

### Installation Requirements
Conversion functions depend on casatools from [CASA 6](https://open-bitbucket.nrao.edu/projects/CASA/repos/casa6/browse) which is currently compatible with Python 3.6 only.  CASA 6 also requires FORTRAN libraries be installed in the OS.
These are not included in the dependency list so that the rest of CNGI functionality may be used without these constraints.

In the future, the conversion module may be moved outside of the CNGI package and distributed separately.

### Pip Installation

```sh
python3 -m venv cngi
source cngi/bin/activate
#(maybe) sudo apt-get install libgfortran3
pip install --index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple casatools==6.0.0.27
pip install cngi-prototype
```

### Conda Installation

```sh
conda create -n cngi python=3.6
conda activate cngi
#(maybe) sudo apt-get install libgfortran3
pip install --index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple casatools==6.0.0.27
pip install cngi-prototype
```


### Installation from Source

```sh
git clone https://github.com/casangi/cngi_prototype.git
cd cngi_prototype
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py install --root=.
```


###  Building Documentation from Source
Follow steps to install cngi from source. Then navigate to the docs folder and execute the following:

```sh
sphinx-build -b html . ./build
```
View the documentation in your browser by navigating to:

 ```sh
file:///path/to/cngi/cngi_prototype/docs/build/index.html
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
>>> df = dio.read_image(...)
```
or
```python
>>> from cngi.dio import read_image
>>> df = read_image(...)
```
or
```python
>>> import cngi.dio as cdio
>>> df = cdio.read_image(...)
```

### Run Tests

Download the test data from https://astrocloud.nrao.edu/s/Hacr42aZmJ3eb7i (or [AWS S3](https://cngi-prototype-test-data.s3.amazonaws.com/sis14_twhya_field5_mstrans_lsrk.zarr.zip)) and place the files in `cngi_prototype/cngi/data/`.
The test scripts can be found in `cngi_prototype/tests/`. 

### Design
READ THIS BEFORE YOU CONTRIBUTE CODE!!!  
  
The CNGI code base is not object oriented, and instead follows a more functional paradigm. Objects are indeed used to hold Visibility
and Image data, but they come directly from the underlying xarray/dask framework and are not extended in any way. The API
consists of stateless Python functions only.  They take in an Visibility or Image object and return a new Visibility or Image object with no
global variables.  

The cngi_prototype _repository_ contains the cngi _package_ along with supporting folders docs and tests. 
Within the cngi package there are a number of modules.  Within each module there are one or more python files. 
CNGI adheres to a strict design philosophy with the following **RULES**:  
1. Each file in a module must have exactly one function exposed to the external API (by docstring and \_\_init\_\_.py).
The exposed function name should match the file name.  This must be a stateless function, not a class. 
2. Files in a module cannot import each other.  
3. Files in separate modules cannot import each other.
4. A single special _helper module exists for internal functions meant to be shared across modules/files. But each
module file should be as self contained as possible.
5. Nothing in _helper may be exposed to the external API.  

```sh
cngi_prototype  
|-- cngi
|    |-- module1
|    |     |-- __init__.py  
|    |     |-- file1.py    
|    |     |-- file2.py  
|    |     | ...  
|    |-- module2  
|    |     |-- __init__.py
|    |     |-- file3.py    
|    |     |-- file4.py  
|    |     | ...  
|    |-- _helper
|    |     |-- __init__.py
|    |     |-- file5.py    
|    |     |-- file6.py  
|    |     | ...  
|-- docs  
|    | ...  
|-- tests  
|    | ...  
|-- requirements.txt  
|-- setup.py  
```
File1, file2, file3 and file4 MUST be documented in the API exactly as they appear. They must NOT import each other.  
  
File5 and file6 must NOT be documented in the API. They may be imported by file1 - 4.
    
\_\_init\_\_.py dictates what is
seen by the API and importable by other functions.
  
  
### Coding Standards

Documentation is generated using Sphinx, with the autodoc and napoleon extensions enabled. Function docstrings should be written in [NumPy style](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#google-vs-numpy). For compatibility with Sphinx, import statements should generally be underneath function definitions, not at the top of the file.

A complete set of formal and enforced coding standards have not yet been formally adopted. Some alternatives under consideration are:

* Google's [style guide](https://google.github.io/styleguide/pyguide.html)
* Python Software Foundation's [style guide](https://www.python.org/dev/peps/pep-008/)
* Following conventions established by PyData projects (examples [one](https://docs.dask.org/en/latest/develop/html) and [two](https://xarray.pydata.org/en/stable/contributing.html#code-standards))

We are evaluating the adoption of [PEP 484](https://www.python.org/dev/peps/pep-0484/) convention, [mypy](http://mypy-lang.org/), or  [param](https://param.holoviz.org) for type-checking, and [flake8](http://flake8.pycqa.org/en/latest/index.html) or [pylint](https://www.pylint.org/) for enforcement.

