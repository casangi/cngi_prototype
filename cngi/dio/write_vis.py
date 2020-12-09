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
def write_vis(xds, outfile='vis.zarr', partition='part0', compressor=None, append=True):
    """
    Write xarray Visibility Dataset to zarr format on disk
  
    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        Visibility Dataset to write to disk
    outfile : str
        output filename, generally ends in .zarr
    partition : str
        Name of partition to write into outfile. Overwrites existing partition of same name. Default is 'part0'
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
    
    outfile = os.path.expanduser(outfile)
    
    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
        
    # need to manually remove existing parquet file (if any)
    if not append:
      tmp = os.system("rm -fr " + outfile)
      tmp = os.system("mkdir " + outfile)
    
    encoding = dict(zip(list(xds.data_vars), cycle([{'compressor': compressor}])))
    xds.to_zarr(os.path.join(outfile, partition), mode='w', encoding=encoding)
