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


#############################################
'''
To Do:
1. Allow users to specify the compressor used. Is this even allowed with Dask, Xarray and Zarr (having a zarr group (datatset) where the data variables are compressed differently).
2. zarr.consolidate_metadata(outfile) is very slow for a zarr group (datatset) with many chunks (there is a python for loop that checks each file). We might have to implement our own version. We could also look at just appending to the json file the information of the new data variables. This is also important for cngi.dio.write_zarr
3. Appending data variables with a dim that does not yet exist. Common dimentions must still be the same size and have the same chunking. This would have to be done carefully.
'''

def append_zarr(list_xarray_data_variables,outfile,chunks_return={},compressor=None,graph_name='append_zarr'):
    """
    Append a list of dask arrays to a zarr file on disk. If a data variable with the same name is found it will be overwritten.
    Data will probably be corrupted if append_zarr overwrites the data variable from which the dask array gets its data.
    The xarray data variables in list_xarray_data_variables must have dimensions that are a subset of the dimensions that are on disk.
    Any graphs that are not part of the list_dask_array will not be executed.
    
    Parameters
    ----------
    list_xarray_data_variables: list of dask arrays
        List of xarray datavariables to append.
    outfile : str
        The file name of the dataset on disk, generally ends in .zarr
    chunks_return : dict of int
        A dictionary with the chunk size that will be returned. For example {'time': 20, 'chan': 6}.
        If chunks_return is not specified the chunking on disk will be used.
    compressor : #Not Yet Implemented# numcodecs.blosc.Blosc
        The blosc compressor to use when saving the converted data to disk using zarr.
        If None the zstd compression algorithm used with compression level 2.
    graph_name : string
        The time taken to execute the graph and save the dataset is measured and saved as an attribute in the zarr file.
        The graph_name is the label for this timing information.
    Returns
    -------
    """
    #Why this function is needed https://stackoverflow.com/questions/58042559/adding-new-xarray-dataarray-to-an-existing-zarr-store-without-re-writing-the-who
    #To understand this function go over dask/array/core.py and xarray/backends/common.py.
    
    from fsspec import get_mapper
    import xarray as xr
    import zarr
    import dask
    import dask.array as da
    import time
    
    start = time.time()
    n_arrays = len(list_xarray_data_variables)
    try:
        disk_dataset = xr.open_zarr(outfile)
    except ValueError:
        print("######### ERROR: Could not open " + outfile)
    
    #Create a list of delayed zarr.create commands for each dask array in ist_dask_array.
    list_target_zarr = []
    list_dask_array = []
    for i in range(n_arrays):
        #Create list of dimension chunk sizes on disk
        chunksize_on_disk =[]
        for array_dim in list_xarray_data_variables[i].dims:
            chunksize_on_disk.append(disk_dataset.chunks[array_dim][0])
        
        
        #Rechunk the dask arrays to match the chunking on disk
        #dask_array = list_dask_array[i].rechunk(chunksize_on_disk[i])
        dask_array = list_xarray_data_variables[i].data.rechunk(chunksize_on_disk[i])
        list_dask_array.append(dask_array)
        
        #Create list of delayed objects
        mapper = get_mapper(outfile+'/'+ list_xarray_data_variables[i].name)
        list_target_zarr.append(dask.delayed(zarr.create)(
             shape=dask_array.shape,
             chunks=chunksize_on_disk,
             dtype=dask_array.dtype,
             store=mapper,
             overwrite=True)) #Can not specify the zarr file attributes at creation , attrs={'_ARRAY_DIMENSIONS':array_dimensions[i]} (last checked on May 2020 )
        
    
    #Trigger compute of delayed zarr.create funmctions in list_target_zarr.
    da.store(list_dask_array,list_target_zarr,compute=True,flush=True)
    
    # Open zarr to add array dimension labels so that xarray.open_zarr works. This is the magic that allows xarray to understand zarr.
    dataset_group = zarr.open_group(outfile,mode='a')
    #This should one day be done during zarr.create. See https://github.com/zarr-developers/zarr-python/issues/538.
    for i in range(n_arrays):
        dataset_group[list_xarray_data_variables[i].name].attrs['_ARRAY_DIMENSIONS'] = list_xarray_data_variables[i].dims
    
    time_to_calc_and_store = time.time() - start
    print('Time to append and execute graph ', graph_name, time_to_calc_and_store)
    dataset_group.attrs[graph_name+'_time'] = time_to_calc_and_store
    
    #Consolidate metadata #Can be improved by only adding appended metadata
    zarr.consolidate_metadata(outfile)
    
    if bool(chunks_return):
        return xr.open_zarr(outfile,chunks=chunks_return,overwrite_encoded_chunks=True,consolidated=True)
    else:
        return xr.open_zarr(outfile,overwrite_encoded_chunks=True,consolidated=True)
