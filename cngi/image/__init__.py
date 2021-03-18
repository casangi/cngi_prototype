"""
These functions examine or manipulate Image data in the xarray Dataset (xds) format.  They
take an xds as input and return a new xds or some other structure as
output.  Some may operate directly on the zarr data store on
disk.

The input xarray Dataset is never modified.

To access these functions, use your favorite variation of:
``import cngi.image``
"""
from .cont_sub import cont_sub
from .fit_gaussian import fit_gaussian
from .fit_gaussian_rl import fit_gaussian_rl
from .gaussian_beam import gaussian_beam
from .implot import implot
from .mask import mask
from .moments import moments
from .rebin import rebin
from .reframe import reframe
from .region import region
from .smooth import smooth
from .spec_fit import spec_fit
from .statistics import statistics
from .stokes_to_corr import stokes_to_corr
from .make_empty_sky_image import make_empty_sky_image
