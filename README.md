Installation, setup and development rules for the CASA Next Generation Infrastructure  

# Installation
Conversion functions depend on casatools from [CASA 6](https://open-bitbucket.nrao.edu/projects/CASA/repos/casa6/browse) which is currently compatible with Python 3.6 only.  CASA 6 also requires FORTRAN libraries be installed in the OS.
These are not included in the dependency list so that the rest of CNGI functionality may be used without these constraints.

In the future, the conversion module may be moved outside of the CNGI package and distributed separately.

## Pip Installation

```sh
python3 -m venv cngi
source cngi/bin/activate
#(maybe) sudo apt-get install libgfortran3
pip install --index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple casatools==6.0.0.27
pip install cngi-prototype
```

## Conda Installation

```sh
conda create -n cngi python=3.6
conda activate cngi
#(maybe) sudo apt-get install libgfortran3
pip install --index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple casatools==6.0.0.27
pip install cngi-prototype
```


## Installation from Source

```sh
git clone https://github.com/casangi/cngi_prototype.git
cd cngi_prototype
python3 -m venv venv
source venv/bin/activate
pip install wheel
pip install -r requirements.txt
python setup.py install --root=.
```

## Configure dask.distributed
To avoid thread collisions, when using the Dask.distributed Client, set the following environment variables.

```sh
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 
```

##  Building Documentation from Source
Follow steps to install cngi from source. Then navigate to the docs folder and execute the following:

```sh
sphinx-build -b html . ./build
```
View the documentation in your browser by navigating to:

 ```sh
file:///path/to/cngi/cngi_prototype/docs/build/index.html
```


# Usage
You can import things several different ways.  For example:
```python
>>> from cngi import dio
>>> xds = dio.read_image(...)
```
or
```python
>>> from cngi.dio import read_image
>>> xds = read_image(...)
```
or
```python
>>> import cngi.dio as cdio
>>> xds = cdio.read_image(...)
```

Throughout the documentation we use the variable name `xds` to refer to Xarray DataSets.  
`xda` may be used to refer to Xarray DataArrays.  This is a "loose" convention only. 
