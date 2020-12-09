"""
Legacy CASA uses a custom MS format while CNGI uses the standard
Zarr format.  These functions allow conversion between
the two as well as directly from the telescope archival science data
model (ASDM) (future growth).  Note that both the MS and Zarr formats
are directories, not single files.

This package has a dependency on legacy CASA / casacore and will be
separated in the future to its own distribution apart from the rest of
the CNGI package.

To access these functions, use your favorite variation of:
``import cngi.conversion``
"""
from .convert_ms import *
from .convert_image import *
from .convert_asdm import *
from .convert_table import *
from .describe_ms import *
from .save_ms import *
from .save_image import *
from .save_asdm import *

