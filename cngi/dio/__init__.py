"""
Most CNGI functions operate on xarray Datasets while the
data is stored on disk in Zarr format.  These functions
allow the transition back and forth between the two.

To access these functions, use your favorite variation of:
``import cngi.dio``
"""
from .read_vis import *
from .write_vis import *
from .describe_vis import *
from .read_image import *
from .write_image import *
from .write_zarr import *
from .append_zarr import *
