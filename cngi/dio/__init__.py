"""
Most CNGI functions operate on xarray Datasets while the
data is stored on disk in Zarr format.  These functions
allow the transition back and forth between the two.

To access these functions, use your favorite variation of:
``import cngi.dio``
"""
from .write_vis import *
from .write_image import *
from .read_vis import *
from .describe_vis import *
from .read_image import *
from .append_xds import *
