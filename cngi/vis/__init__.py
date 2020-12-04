"""
These functions examine or manipulate Visibility data in the xarray Dataset (xds) format.  They
take an xds as input and return a new xds or some other structure as
output.  Some may operate directly on the zarr data store on
disk.

The input xarray Dataset is never modified.

To access these functions, use your favorite variation of:
``import cngi.vis``
"""
from .applyflags import *
from .chanaverage import *
from .chansmooth import *
from .ddijoin import *
from .visjoin import *
from .phaseshift import *
from .ddiregrid import *
from .timeaverage import *
from .uvcontfit import *
from .uvmodelfit import *
from .visplot import *
from .sdfit import *
from .sdpolaverage import *
