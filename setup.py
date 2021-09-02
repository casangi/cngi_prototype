from setuptools import setup, find_packages

with open('README.md', "r") as fid:   #encoding='utf-8'
    long_description = fid.read()

setup(
    name='cngi_prototype',
    version='0.0.91',
    description='CASA Next Generation Infrastructure Prototype',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='National Radio Astronomy Observatory',
    author_email='casa-feedback@nrao.edu',
    url='https://github.com/casangi/cngi_prototype',
    license='Apache-2.0',
    packages=find_packages(),
    install_requires=['bokeh>=2.2.3',
                      'dask>=2.13.0',
                      'distributed>=2.9.3',
                      'graphviz>=0.13.2',
                      'matplotlib>=3.1.2',
                      'numba>=0.51.0',
                      'numcodecs>=0.6.3',
                      'numpy>=1.18.1',
                      'pandas>=0.25.2',
                      'scipy>=1.4.1',
                      'scikit-learn>=0.22.2',
                      'toolz>=0.10.0',
                      'xarray>=0.16.1',
                      'zarr>=2.3.2',
                      'fsspec>=0.6.2',
                      'gdown>=3.12.2',
                      'ipympl>=0.7.0',
                      'python-casacore>=3.4.0',
                      'casadata>=2020.11.16'],
    extras_require={
        'dev': [
            'pytest>=5.3.5',
            'black>=19.10.b0',
            'flake8>=3.7.9',
            'isort>=4.3.21',
            's3fs>=0.4.0',
            'pylint>=2.4.4',
            #'pytest-pep8',
            #'pytest-cov'
        ]
    }

)
