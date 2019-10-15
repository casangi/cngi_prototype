Conversion
====================

Legacy CASA uses a custom MS format while CNGI uses the standard Apache Parquet (PQ) format.  These functions allow conversion between the two as well as directly from the telescope archival science data model (ASDM) (future growth).  Note that both the MS and PQ formats are directories, not single files.


.. autofunction:: conversion.ms_to_pq
.. autofunction:: conversion.asdm_to_pq
.. autofunction:: conversion.fits_to_pq
.. autofunction:: conversion.pq_to_ms
.. autofunction:: conversion.pq_to_asdm
.. autofunction:: conversion.pq_to_fits
