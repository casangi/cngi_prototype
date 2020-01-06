CASA Next Generation Infrastructure
===================================    

Organization
^^^^^^^^^^^^

CNGI is organized into modules as described below. Each module is
responsible for a different functional area, such as file conversion,
input / output, MS operations, and Image operations.  

.. toctree::
   :maxdepth: 1

   conversion
   dio
   ms
   images
   direct
   gridding

Installation Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^
Installing the CNGI prototype currently only works on Linux systems with Python 3.6 (due to the CASA 6 requirement for the measurement set conversion functions). 
This dependency on CASA 6 and will be removed in the future. It suggested that a Python virtual environment is created using either Conda or pip.

Installation
^^^^^^^^^^^^

.. code-block:: none

   conda create -n cngi python=3.6
   conda activate cngi
   sudo apt-get install libgfortran3 # (maybe) 
   pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-group/simple casatools
   pip install cngi-prototype

Installation from Source
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   git clone https://github.com/casangi/cngi_prototype.git
   cd cngi-prototype
   conda create -n cngi python=3.6
   conda activate cngi
   pip install -e .

Building Documentation from Source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Navigate to the cngi_prototype directory.

.. code-block:: none

   conda activate cngi
   pip install sphinx-automodapi
   pip install sphinx_rtd_theme
   sphinx-build -b html ./docs/ ./docs/_build/

Running CNGI in Parallel using Dask.distributed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To avoid thread collisions, when using the Dask.distributed Client, set the following environment variables.

.. code-block:: none

   export OMP_NUM_THREADS=1 
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1 

Usage
^^^^^

You can import things several different ways.  For example:

.. code-block:: python

   from cngi import dio
   df = dio.read_pq(...)

or

.. code-block:: python

   from cngi.dio import read_pq
   df = read_pq(...)

or

.. code-block:: python

   import cngi.dio as cdio
   df = cdio.read_pq(...)
   
Run Tests
^^^^^^^^^

Download the test data from https://astrocloud.nrao.edu/s/Hacr42aZmJ3eb7i and place the files in cngi_prototype/data/.
The test scripts can be found in cngi_prototype/cngi/tests/. 

Coding Standards
^^^^^^^^^^^^^^^^

Documentation is generated using Sphinx, with the autodoc and napoleon extensions enabled. Function docstrings should be written in `NumPy style <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#google-vs-numpy>`_. For compatibility with Sphinx, import statements should generally be underneath function definitions, not at the top of the file.

A complete set of formal and enforced coding standards have not yet been formally adopted. Some alternatives under consideration are:

* Google's `style guide <https://google.github.io/styleguide/pyguide.html>`_
* Python Software Foundation's `style guide <https://www.python.org/dev/peps/pep-008/>`_
* Following conventions established by PyData projects (examples `one <https://docs.dask.org/en/latest/develop/html>`_ and `two <https://xarray.pydata.org/en/stable/contributing.html#code-standards>`_)

We are evaluating the adoption of `PEP 484 <https://www.python.org/dev/peps/pep-0484/>`_ convention, `mypy <http://mypy-lang.org/>`_, or  `param <https://param.holoviz.org/>`_ for type-checking, and `flake8 <http://flake8.pycqa.org/en/latest/index.html>`_ or `pylint <https://www.pylint.org/>`_ for enforcement.

