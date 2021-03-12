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


'''
To Do:
1. zarr.consolidate_metadata(outfile) is very slow for a zarr group (datatset) with many chunks (there is a python for loop that checks each file). We might have to implement our own version. This is also important for cngi.dio.append_zarr
'''

def write_image(xds, outfile, chunks_return={}, chunks_on_disk={}, consolidated=True, compressor=None, graph_name='write_zarr'):
    """
    Write image xarray dataset to zarr format on disk. When chunks_on_disk is not specified the chunking in the input dataset is used.
    When chunks_on_disk is specified that dataset is saved using that chunking.
    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        Dataset to write to disk
    outfile : str
        outfile filename, generally ends in .zarr
    chunks_on_disk : dict of int
        A dictionary with the chunk size that will be used when writing to disk. For example {'time': 20, 'chan': 6}.
        If chunks_on_disk is not specified the chunking of dataset will be used.
    compressor : numcodecs.blosc.Blosc
        The blosc compressor to use when saving the converted data to disk using zarr.
        If None the zstd compression algorithm used with compression level 2.
    graph_name : string
        The time taken to execute the graph and save the dataset is measured and saved as an attribute in the zarr file.
        The graph_name is the label for this timing information.
    Returns
    -------
    """
    import xarray as xr
    import zarr
    import time
    from numcodecs import Blosc
    from itertools import cycle
    
    #Check if disk chunking is specified
    if bool(chunks_on_disk):
        xds_for_disk = dataset.chunk(chunks=chunks_on_disk)
    else:
        xds_for_disk = xds
        
    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
    
    #Create compression encoding for each datavariable
    encoding = dict(zip(list(xds_for_disk.data_vars), cycle([{'compressor': compressor}])))
    start = time.time()
    #Consolidated is set to False so that the timing information is included in the consolidate metadata.
    xr.Dataset.to_zarr(xds_for_disk, store=outfile, mode='w', encoding=encoding,consolidated=False)
    time_to_calc_and_store = time.time() - start
    print('Time to store and execute graph ', graph_name, time_to_calc_and_store)
    
    #Add timing information
    xds_group = zarr.open_group(outfile,mode='a')
    xds_group.attrs[graph_name+'_time'] = time_to_calc_and_store
    
    if consolidated:
        #Consolidate metadata
        zarr.consolidate_metadata(outfile)
