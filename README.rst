CASA Next Generation Infrastructure
===============================================================      

Documentation
^^^^^^^^^^^^^^^^^^^^^

API and User Manual: https://cngi-prototype.readthedocs.io


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

- conversion
- dio
- ms
- images
- direct


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
