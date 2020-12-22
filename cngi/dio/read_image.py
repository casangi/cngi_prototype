#   Copyright 2019 AUI, Inc. Washington DC, USA
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
this module will be included in the api
"""


#############################################
def read_image(infile, chunks=None, consolidated=True, overwrite_encoded_chunks=True):
    """
    Read xarray zarr format image from disk

    Parameters
    ----------
    infile : str
        input zarr image filename
    chunks : dict
        sets specified chunk size per dimension. Dict is in the form of 'dim':chunk_size, for example {'d0':100, 'd1':100, 'chan':32, 'pol':1}.
        Default None uses the original zarr chunking.
    consolidated : bool
        use zarr consolidated metadata capability. Only works for stores that have already been consolidated. Default True works with datasets
        produced by convert_image which automatically consolidates metadata.
    overwrite_encoded_chunks : bool
        drop the zarr chunks encoded for each variable when a dataset is loaded with specified chunk sizes.  Default True, only applies when chunks
        is not None.
  
    Returns
    -------
    xarray.core.dataset.Dataset
        New xarray Dataset of image contents
    """
    import os
    from xarray import open_zarr
 
    if chunks is None:
        chunks = 'auto'
        overwrite_encoded_chunks = False
 
    infile = os.path.expanduser(infile)
    xds = open_zarr(infile, chunks=chunks, consolidated=consolidated, overwrite_encoded_chunks=overwrite_encoded_chunks)
    return xds

