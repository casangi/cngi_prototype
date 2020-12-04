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

def write_zarr(dataset, outfile, chunks_return={}, chunks_on_disk={}, compressor=None, graph_name='write_zarr'):
    """
    Write xarray dataset to zarr format on disk. When chunks_on_disk is not specified the chunking in the input dataset is used.
    When chunks_on_disk is specified that dataset is saved using that chunking. The dataset on disk is then opened and rechunked using chunks_return or the chunking of dataset.
    Parameters
    ----------
    dataset : xarray.core.dataset.Dataset
        Dataset to write to disk
    outfile : str
        outfile filename, generally ends in .zarr
    chunks_return : dict of int
        A dictionary with the chunk size that will be returned. For example {'time': 20, 'chan': 6}.
        If chunks_return is not specified the chunking of dataset will be used.
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
    from zarr.meta import json_dumps, json_loads
    from zarr.creation import normalize_store_arg, open_array
    
    #Check if disk chunking is specified
    if bool(chunks_on_disk):
        dataset_for_disk = dataset.chunk(chunks=chunks_on_disk)
    else:
        dataset_for_disk = dataset
        
    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
    
    #Create compression encoding for each datavariable
    encoding = dict(zip(list(dataset_for_disk.data_vars), cycle([{'compressor': compressor}])))
    start = time.time()
    #Consolidated is set to False so that the timing information is included in the consolidate metadata.
    xr.Dataset.to_zarr(dataset_for_disk, store=outfile, mode='w', encoding=encoding,consolidated=False)
    time_to_calc_and_store = time.time() - start
    print('Time to store and execute graph ', graph_name, time_to_calc_and_store)
    
    #Add timing information
    dataset_group = zarr.open_group(outfile,mode='a')
    dataset_group.attrs[graph_name+'_time'] = time_to_calc_and_store
    
    #Consolidate metadata
    zarr.consolidate_metadata(outfile)
    
    if bool(chunks_return):
        return xr.open_zarr(outfile,consolidated=True,overwrite_encoded_chunks=True)
    else:
        #Get input dataset chunking
        for dim_key in dataset.chunks:
            chunks_return[dim_key] = dataset.chunks[dim_key][0]
        return xr.open_zarr(outfile,chunks=chunks_return,consolidated=True,overwrite_encoded_chunks=True)
    
