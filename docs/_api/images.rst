Images
====================

These functions examine or manipulate Image data in the xarray Dataset (xds) format.  They
take an xds as input and return a new xds or some other structure as
output.  Some may operate directly on the zarr data store on
disk.

The input xarray Dataset is never modified.

To access these functions, use your favorite variation of:
``import cngi.image``

.. automodsumm:: cngi.image
   :toctree: api
   :nosignatures:
   :functions-only:
