CASA Next Generation Infrastructure
===============================================================      

Installation Requirements
^^^^^^^^^^^^^^^^^^^^^
Installing the CNGI prototype currently only works on Linux systems with Python 3.6 (due to the CASA 6 requirement for the measurement set conversion functions). 
This dependency on CASA 6 and will be removed in the future. It suggested that a Python virtual environment is created using either Conda or pip.

Installation
^^^^^^^^^^^^^^^^^^^^^

- conda create -n cngi python=3.6
- conda activate cngi
- (maybe) sudo apt-get install libgfortran3
- pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-group/simple casatools
- pip install cngi-prototype

Installation from Source
^^^^^^^^^^^^^^^^^^^^^

- git clone https://github.com/casangi/cngi_prototype.git
- cd cngi-prototype
- conda create -n cngi python=3.6
- conda activate cngi
- pip install -e .

Building Documentation from Source
^^^^^^^^^^^^^^^^^^^^^
Navigate to the cngi_prototype directory.

- conda activate cngi
- pip install sphinx-automodapi
- pip install sphinx_rtd_theme
- sphinx-build -b html ./docs/ ./build/

Running CNGI in Parallel using Dask.distributed
^^^^^^^^^^^^^^^^^^^^^
To avoid thread collisions, when using the Dask.distributed Client, set the following environment variables.

- export OMP_NUM_THREADS=1 
- export MKL_NUM_THREADS=1
- export OPENBLAS_NUM_THREADS=1 


Organization
^^^^^^^^^^^^^^^^^^^^^

CNGI is organized in to modules as described below. Each module is
responsible for a different functional area, such as file conversion,
input / output, MS operations, and Image operations.  

.. toctree::
   :maxdepth: 1

   conversion
   dio
   ms
   images
   direct
   synthesis
   unit_tests

Usage
^^^^^^^^^^^^^^^^^^^^^

You can import things several different ways.  For example:

``from cngi import dio``

``df = dio.read_pq(...)``

or

``from cngi.dio import read_pq``

``df = read_pq(...)``

or

``import cngi.dio as cdio``

``df = cdio.read_pq(...)``

Coding Standards
^^^^^^^^^^^^^^^^^^^^^

Documentation is generated using Sphinx, with the autodoc and napoleon extensions enabled. Function docstrings should be written in `NumPy style <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#google-vs-numpy>`_. For compatibility with Sphinx, import statements should generally be underneath function definitions, not at the top of the file.

A complete set of formal and enforced coding standards have not yet been formally adopted. Some alternatives under consideration are:

* Google's `style guide <https://google.github.io/styleguide/pyguide.html>`_
* Python Software Foundation's `style guide <https://www.python.org/dev/peps/pep-008/>`_
* Following conventions established by PyData projects (examples `one <https://docs.dask.org/en/latest/develop/html>`_ and `two <https://xarray.pydata.org/en/stable/contributing.html#code-standards>`_)

We are evaluating the adoption of `PEP 484 <https://www.python.org/dev/peps/pep-0484/>`_ convention, `mypy <http://mypy-lang.org/>`_, or  `param <https://param.holoviz.org/>`_ for type-checking, and `flake8 <http://flake8.pycqa.org/en/latest/index.html>`_ or `pylint <https://www.pylint.org/>`_ for enforcement.

