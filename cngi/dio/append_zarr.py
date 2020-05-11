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
def append_zarr(dataset,outfile,list_dask_array,list_data_variable_name,list_array_dimensions,graph_name='append_zarr'):
    """
    Append a list of dask arrays to a zarr file on disk. If a data variable with the same name is found it will be overwritten.
    Data will probably be corrupted if append_zarr overwrites the data variable from which the dask array gets its data.
    The xarray data variables must have dimensions that are a subset of the dimensions that are on disk. The number of elements in each list (list_dask_array,list_data_variable_name,list_array_dimensions) must be the same.
    An example of parameters for saving two dask arrays:
    list_data_variable_name = [psf_nat, sum_weight]
    list_data_variable_name = ['PSF_NAT', 'SUM_WEIGHT']
    list_array_dimensions = [['d0', 'd1', 'chan', 'pol'],['chan', 'pol']]
    Note that all the dimensions in list_array_dimensions must already be in the dataset on disk. Any graphs that are not part of the list_dask_array will not be executed.
    
    Parameters
    ----------
    dataset : xarray.core.dataset.Dataset
        Dataset that the list_dask_array will be appended to.
    outfile : str
        The file name of the dataset on disk, generally ends in .zarr
    list_dask_array: list of dask arrays
        List of dask arrays to append.
    list_data_variable_name: list of str
        List of names that will be used to identify the list_dask_array in the dataset.
    list_array_dimensions: list of (list of str)
        A list where each element is a list of the dimensions for each array in list_dask_array.
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
    n_arrays = len(list_dask_array)
    disk_dataset = xr.open_zarr(outfile)
    
    #Create a list of delayed zarr.create commands for each dask array in ist_dask_array.
    list_target_zarr = []
    for i in range(n_arrays):
        #Create list of dimension chunk sizes on disk
        chunksize_on_disk =[]
        for array_dim in list_array_dimensions[i]:
            chunksize_on_disk.append(disk_dataset.chunks[array_dim][0])
        
        #Rechunk the dask arrays to match the chunking on disk
        dask_array = list_dask_array[i].rechunk(chunksize_on_disk[i])
        
        #Create list of delayed objects
        mapper = get_mapper(outfile+'/'+list_data_variable_name[i])
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
        dataset_group[list_data_variable_name[i]].attrs['_ARRAY_DIMENSIONS'] = list_array_dimensions[i]
    
    time_to_calc_and_store = time.time() - start
    print('Time to append and execute graph ', graph_name, time_to_calc_and_store)
    dataset_group.attrs[graph_name+'_time'] = time_to_calc_and_store
    
    # Open dataset from zarr array preserving the chunking in dataset
    # Must be a beter way to get dict from SortedKeyDict
    current_dataset_chunk_size = {}
    for dim_key in dataset.chunks:
        current_dataset_chunk_size[dim_key] = dataset.chunks[dim_key][0]
    
    return xr.open_zarr(outfile,chunks=current_dataset_chunk_size,overwrite_encoded_chunks=True)
