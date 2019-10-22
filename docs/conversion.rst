Conversion
====================

Legacy CASA uses a custom MS format while CNGI uses the standard
Apache Parquet (PQ) format.  These functions allow conversion between
the two as well as directly from the telescope archival science data
model (ASDM) (future growth).  Note that both the MS and PQ formats
are directories, not single files.

This package has a dependency on legacy CASA / casacore and will be
separated in the future to its own distribution apart from the rest of
the CNGI package.

.. automodsumm:: conversion
   :toctree: api
   :nosignatures:
