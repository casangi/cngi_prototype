CASA Next Generation Infrastructure
===============================================================      

Installation
^^^^^^^^^^^^^^^^^^^^^
- (maybe) apt-get install libgfortran3
- pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-group/simple casatools
- pip install cngi-prototype


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

The CNGI project has not yet formally adopted a set of enforceable coding standards.

Some alternatives under consideration are:

* Google's `style guide <https://google.github.io/styleguide/pyguide.html>`_
* Python Software Foundation's `style guide <https://www.python.org/dev/peps/pep-008/>`_
* Something like a PyData project (`two <https://docs.dask.org/en/latest/develop/html>`_ `examples <xarray.pydata.org/en/stable/contributing.html#code-standards>`_)

We're thinking about using flake8, mypy, pysort, lint, etc. for checking and enforcement.
