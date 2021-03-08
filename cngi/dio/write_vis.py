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
def write_vis(mxds, outfile='vis.zarr', partition=None, compressor=None, append=False):
    """
    Write xarray Visibility Dataset to zarr format on disk
    
    Note the convention that global subtables are all CAPITALIZED in the mxds while visibility
    partitions are all lowercase
  
    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        input multi-xarray Dataset with global data
    outfile : str
        output filename, generally ends in .zarr
    partition : str or list
        Name of partition xds to write into outfile (from the mxds attributes section). Overwrites existing partition of same name.
        Default None writes entire mxds
    compressor : numcodecs.blosc.Blosc
        The blosc compressor to use when saving the converted data to disk using zarr.
        If None the zstd compression algorithm used with compression level 2.
    append : bool
        Append this partition in to an existing zarr directory. False will erase old zarr directory. Default=True
    
    Returns
    -------
    """
    import os
    from numcodecs import Blosc
    from itertools import cycle
    import numpy as np
    
    outfile = os.path.expanduser(outfile)
    
    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
        
    # need to manually remove existing parquet file (if any)
    if not append:
      tmp = os.system("rm -fr " + outfile)
      tmp = os.system("mkdir " + outfile)
    
    if partition is None:
        partition = list(mxds.attrs.keys())
    partition = list(np.atleast_1d(partition))
    
    for part in partition:
        assert part in mxds.attrs, 'invalid partition parameter, not found in mxds'
        encoding = dict(zip(list(mxds.attrs[partition].data_vars), cycle([{'compressor': compressor}])))
        subdir = 'global' if partition == partition.upper() else ''
        mxds.attrs[partition].to_zarr(os.path.join(outfile, subdir), mode='w', encoding=encoding)
        
