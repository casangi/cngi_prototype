# CASA Next Generation Infrastructure

### Documentation
API and User Manual: https://cngi-prototype.readthedocs.io

### Installation Requirements
Installing the CNGI prototype currently only works on Linux systems with Python 3.6 (due to the CASA 6 requirement for the measurement set conversion functions). 
This dependency on CASA 6 and will be removed in the future. It suggested that a Python virtual environment is created using either Conda or pip.

### Installation

- conda create -n cngi python=3.6
- conda activate cngi
- (maybe) sudo apt-get install libgfortran3
- pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-group/simple casatools
- pip install cngi-prototype

### Installation from Source

- git clone https://github.com/casangi/cngi_prototype.git
- cd cngi-prototype
- conda create -n cngi python=3.6
- conda activate cngi
- pip install -e .

###  Building Documentation from Source
Navigate to the cngi_prototype directory.

- conda activate cngi
- pip install sphinx-automodapi
- pip install sphinx_rtd_theme
- sphinx-build -b html ./docs/ ./build/

### Running CNGI in Parallel using Dask.distributed
To avoid thread collisions, when using the Dask.distributed Client, set the following environment variables.

- export OMP_NUM_THREADS=1 
- export MKL_NUM_THREADS=1
- export OPENBLAS_NUM_THREADS=1 


### Organization
CNGI is organized in to modules as described below. Each module is
responsible for a different functional area, such as file conversion,
input / output, MS operations, and Image operations.  

- conversion
- dio
- ms
- images
- direct
- unit_tests

### Usage
You can import things several different ways.  For example:
```sh
>>> from cngi import dio
>>> df = dio.read_pq(...)
```
or
```sh
>>> from cngi.dio import read_pq
>>> df = read_pq(...)
```
or
```sh
>>> import cngi.dio as cdio
>>> df = cdio.read_pq(...)
```
