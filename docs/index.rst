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
   gridding

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

