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
def write_image(xds, outfile='image.zarr'):
    """
    Write image dataset to xarray zarr format on disk
    
    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        image Dataset to write to disk
    outfile : str
        output filename, generally ends in .zarr
    
    Returns
    -------
    """
    import os
    from numcodecs import Blosc
    from itertools import cycle
    
    outfile = os.path.expanduser(outfile)
    compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
    encoding = dict(zip(list(xds.data_vars), 
                        cycle([{'compressor': compressor}])))
    
    xds.to_zarr(outfile, mode='w', encoding=encoding)


