"""
These functions examine or manipulate Visibility data in the xarray Dataset (xds) format.  They
take an xds as input and return a new xds or some other structure as
output.  Some may operate directly on the zarr data store on
disk.

The input xarray Dataset is never modified.

To access these functions, use your favorite variation of:
``import cngi.vis``
"""
from .apply_flags import apply_flags
from .chan_average import chan_average
from .chan_smooth import chan_smooth
from .join_vis import join_vis
from .join_dataset import join_dataset
from .phase_shift import phase_shift
from .reframe import reframe
from .time_average import time_average
from .uv_cont_fit import uv_cont_fit
from .uv_model_fit import uv_model_fit
from .visplot import visplot
from .sd_fit import sd_fit
from .sd_polaverage import sd_polaverage
