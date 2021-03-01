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
from .corr_to_stokes import corr_to_stokes
from .cube_to_mfs import cube_to_mfs
from .fit_gaussian import fit_gaussian
from .fit_gaussian_rl import fit_gaussian_rl
from .fourier_transform import fourier_transform
from .gaussian_beam import gaussian_beam
from .implot import implot
from .mask import mask
from .mfs_to_cube import mfs_to_cube
from .moments import moments
from .pos_vel import pos_vel
from .rebin import rebin
from .reframe import reframe
from .region import region
from .regrid import regrid
from .rm_fit import rm_fit
from .sd_fix_scan import sd_fix_scan
from .smooth import smooth
from .spec_fit import spec_fit
from .spec_flux import spec_flux
from .spx_fit import spx_fit
from .statistics import statistics
from .stokes_to_corr import stokes_to_corr
from .make_empty_sky_image import make_empty_sky_image


