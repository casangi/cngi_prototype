"""
These functions examine or manipulate Image data in the xarray Dataset (xds) format.  They
take an xds as input and return a new xds or some other structure as
output.  Some may operate directly on the zarr data store on
disk.

The input xarray Dataset is never modified.

To access these functions, use your favorite variation of:
``import cngi.image``
"""
from .contsub import contsub
from .corr_to_stokes import corr_to_stokes
from .cube_to_mfs import cube_to_mfs
from .fit_gaussian import fit_gaussian
from .fit_gaussian_rl import fit_gaussian_rl
from .fourier_transform import fourier_transform
from .gaussianbeam import gaussianbeam
from .implot import implot
from .mask import mask
from .mfs_to_cube import mfs_to_cube
from .moments import moments
from .posvel import posvel
from .rebin import rebin
from .reframe import reframe
from .region import region
from .regrid import regrid
from .rmfit import rmfit
from .sdfixscan import sdfixscan
from .smooth import smooth
from .specfit import specfit
from .specflux import specflux
from .spxfit import spxfit
from .statistics import statistics
from .stokes_to_corr import stokes_to_corr
from .make_empty_sky_image import make_empty_sky_image


