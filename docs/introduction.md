# Introduction

CASA software development faces several competing challenges that are expected to only worsen with time.  As both a scientific 
research and development platform as well as a necessary component in multi-million dollar telescope operations, CASA must overcome 
the contradictory goals of being a flexible experimentation platform with mature, documented, tested, and stable performance.  
Furthermore, it must accomplish this with a relatively modest development staff that is diverse, remote, and matrixed across many 
competing projects.

Historically, much of CASA traces its roots to algorithms developed and refined over many decades and implemented in a custom code base 
utilized and maintained by a global, yet small developer base.  In years past, it was necessary to develop a custom infrastructure to 
support the unique needs of radio astronomy science.  Yet as data sizes, performance demands, and conflicting needs of multiple telescopes 
continue to rise, the overhead of maintaining this infrastructure grows exponentially.

The CASA team has begun exploring options for a new generation of software to meet the growing demands of current and future instruments. 
This **cngi_prototype** package is a demonstration of the current state of our research efforts.  Its primary purpose is to showcase
new data structures for MeasurementSet and Image contents built entirely in Python atop the popular technology stack of numpy, dask, 
and xarray. A selection of core mathematics, manipulation, middleware and analysis functions are shown to demonstrate the simplicity and
scalability of the technology choices. Finally, the most compute intensive areas of CASA imaging are implemented and benchmarked to
demonstrate the parallel scalability and raw performance now possible from a pure-Python software stack.


## Project Background
The Common Astronomy Support Applications (CASA) package supports a variety of radio astronomy needs by providing mechanisms for data 
manipulation, analysis, calibration and imaging.  Derivatives of these broad categories include items such as visualization, simulation, 
conversion, and experimentation.  The primary data products of CASA are the MeasurementSet (MS) and Image file formats.  Additional 
products are not discussed here but include calibration tables and imaging cache.

CASA is a layered collection of libraries, classes and functions spanning both C++ and Python languages as well as Object Oriented and 
Functional paradigms.  The base layer of CASA includes a standalone project, known as casacore, developed and maintained here: 
[https://github.com/casacore/casacore](https://github.com/casacore/casacore), as well as various external libraries to support mathematics, 
parallelization, visualization, etc. The base layer is wrapped under a middleware layer of CASA that abstracts some of the details of data 
access and allows for simpler expression of science algorithms.  The top layers are user-facing tools and tasks that directly expose the 
science of radio interferometer data reduction.


### Goals
A great deal of the engineering complexity and processing time in CASA stems from the way in which underlying data is accessed and 
manipulated in the infrastructure layers.  By overhauling these base infrastructure layers, significant savings in complexity and 
processing time along with dramatic improvements to science development flexibility and potential can be realized.  Here we highlight 
the main goals and opportunities for improvement within the CASA codebase:

**Maximize Performance**
- Fast experimentation
- (Near) linear scalability allowing users to reduce processing time by adding cheap commodity hardware resources 
- Extreme scalability to thousands of cores and petascale data sizes needed by next generation instruments such as ngVLA
- Fully utilize all cores, memory, and disk I/O simultaneously
- Support everything from single user laptops to high-end clusters in enterprise datacenters as well as fully distributed grid 
  and cloud processing environments

**Minimize engineering overhead**
- Simplify scientific implementations without distraction of software engineering complexity
- Reduce custom code base by an order of magnitude
- Reduce development time of new features and maintenance overhead of existing features
- Support a variety of use cases and interoperability with other packages by leveraging off-the-shelf standards, tools, and formats


### Scope
A complete top-down overhaul of the entirety of CASA is not feasible as a first (and only) step, as there are no complete and 
detailed requirements specifications, design documents, architecture or definition of correct output. Instead, the scope of this 
project is to first condense and replace the CASA data processing software infrastructure and casacore code base only with a new 
and functionally equivalent package (CNGI) in a bottom-up approach.  As a consequence of this infrastructure replacement, a second 
effort to migrate and/or replace significant portions of CASA science functionality is planned as an additional follow-on project (ngCASA).  

Although roughly analogous to the current separation of CASA and casacore, the new ngCASA / CNGI separation will include more capabilities 
in CNGI than existed in casacore.  An additional distinction is being made between ngCASA as a library of Python functions, and the user 
applications built around those functions (i.e. PlotMS).

The following diagram illustrates the scope of CNGI and ngCASA, and the ramifications for subsequent development.

![singlemachinearchitecture](https://raw.githubusercontent.com/casangi/cngi_prototype/master/docs/_media/scope.png)


### CASA Next Generation Infrastructure (CNGI)
The CASA Next Generation Infrastructure (CNGI) project will provide a replacement to the infrastructure and middleware layers of CASA 
using largely off-the-shelf products that are dramatically simpler to use and scalable to a variety of computing environments.  This 
will take the form of a processing package that includes a programming language, API, and hardware interface drivers.  This package 
will be functionally equivalent to the MeasurementSet and Image data manipulation capabilities of the base and middleware layers it 
replaces, but realized in an entirely new paradigm.


### Next Generation CASA (ngCASA) - Future
The future Next Generation CASA project will use visibility and image manipulation methods from CNGI to implement a data analysis package 
that replaces the scientific application layer of CASA. This package will provide a set of functions that may be used either as stand-alone 
building blocks or strung together in a series (internally forming a DAG) to implement operations such as synthesis calibration and image
reconstruction.  A user of ngCASA may choose to implement their own analysis task, use a pre-packaged task similar to one in current CASA, 
or embed ngCASA methods in a custom pipeline DAG.


### High Level Separation
The main purpose behind separating layers of the code base into individual CNGI and ngCASA packages (and then further separating ngCASA 
with an additional Application layer) is to better support a diverse and wide ranging user base.  Global partnerships and collaborative 
agreements may be formed over the basic data types and processing engine in CNGI, with more observatory specific tailoring appearing the 
further up you go in the preceding diagram.  An important but often overlooked distinction exists between the needs of interactive human 
users, automated pipelines, and other applications wishing to build in CASA capabilities.  The addition of an application layer allows 
for better tailoring of the user experience to each different category of user.


### CNGI_Prototype Demonstration Package
This CNGI_Prototype demonstration package is a pilot effort to assess the technology choices made to date for the CNGI Project. It is
primarily focused on prototyping CNGI-layer functionality to see how well it can work and perform against the goals of the project. A
second objective is to show how technology choices are likely to satisfy future scientific and engineering needs of the ngCASA project. 
As such, some preliminary ngCASA layer components are included to build confidence that the infrastructure framework can handle the 
performance and complexity demands, and to illustrate how such functionality may look in the future.

A bottom-up strategy that begins with these prototype building blocks while emphasizing scalability, simplicity and flexibility ensures
meaningful work can proceed in the absence of a detailed requirements specification of future ngCASA needs. This package is likely to 
form the starting point for full production development of CNGI and ngCASA at a later date. As such a high degree of code reuse is 
anticipated.


## Installation
The CNGI_Prototype demonstration package is available as a pypi package for user installation and assessment. Note this is a demonstration
prototype only and **not intended for science**.  

This is a source distribution package, and should work on most recent OS and Python versions >= 3.6.  However the functions to convert
MS and Image data structures from current CASA format to the new format requires the CASA 6.2 casatools module. CASA 6 also requires 
FORTRAN libraries be installed in the OS. These are not included in the dependency list so that the rest of CNGI functionality may be used 
without these constraints.

In the future, the conversion module may be moved outside of the CNGI package and distributed separately.

### Pip Installation

```sh
python3 -m venv cngi
source cngi/bin/activate
#(maybe) sudo apt-get install libgfortran3
pip install --index-url https://casa-pip.nrao.edu:443/repository/pypi-group/simple casatasks==6.2.0.106
pip install --index-url https://casa-pip.nrao.edu:443/repository/pypi-group/simple casadata
pip install cngi-prototype
```

### Conda Installation

```sh
conda create -n cngi python=3.6
conda activate cngi
#(maybe) sudo apt-get install libgfortran3
pip install --index-url https://casa-pip.nrao.edu:443/repository/pypi-group/simple casatasks==6.2.0.106
pip install --index-url https://casa-pip.nrao.edu:443/repository/pypi-group/simple casadata
pip install cngi-prototype
```


### Installation from Source

```sh
git clone https://github.com/casangi/cngi_prototype.git
cd cngi_prototype
python3 -m venv venv
source venv/bin/activate
pip install wheel
pip install -r requirements.txt
python setup.py install --root=.
```

### Configure dask.distributed
To avoid thread collisions, when using the Dask.distributed Client, set the following environment variables.

```sh
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 
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


## Usage
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
