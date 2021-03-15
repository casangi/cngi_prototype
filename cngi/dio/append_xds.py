#  CASA Next Generation Infrastructure
#  Copyright (C) 2021 AUI, Inc. Washington DC, USA
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
this module will be included in the api
"""

#############################################
'''
To Do:
1. zarr.consolidate_metadata(outfile) is very slow for a zarr group (datatset) with many chunks (there is a python for loop that checks each file). We might have to implement our own version. We could also look at just appending to the json file the information of the new data variables. This is also important for cngi.dio.write_zarr
'''

def append_xds(list_xarray_data_variables,outfile,chunks_return={},compressor=None,graph_name='append_zarr'):
    """
    Append a list of dask arrays to a zarr file on disk. If a data variable with the same name is found it will be overwritten.
    Data will probably be corrupted if append_zarr overwrites the data variable from which the dask array gets its data.
    All data variables that share dimensions and coordinates with data variables already on disk must have the same values (chunking can be different).
    
    Parameters
    ----------
    list_xarray_data_variables: list of dask arrays
        List of xarray datavariables to append.
    outfile : str
        The file name of the dataset on disk, generally ends in .zarr
    chunks_return : dict of int
        A dictionary with the chunk size that will be returned. For example {'time': 20, 'chan': 6}.
        If chunks_return is not specified the chunking on disk will be used.
    compressor : numcodecs.blosc.Blosc
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
    from numcodecs import Blosc
    
    start = time.time()
    n_arrays = len(list_xarray_data_variables)
    try:
        disk_dataset = xr.open_zarr(outfile)
    except ValueError:
        print("######### ERROR: Could not open " + outfile)
    
    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
    ######################################################################################
    #Create a list of delayed zarr.create commands (list_target_zarr) for each dask array in list_dask_array.
    ######################################################################################
    
    list_target_zarr = []
    list_dask_array = []
    
    for i in range(n_arrays):
        list_new_dim_name = []
        list_new_coord_name = []
        list_new_coord_dim_names = []
    
        #Create list of dimension chunk sizes on disk
        chunksize_on_disk =[]
        
        #Get array chunksize on disk and add new dimentions
        #The order of the for loop is important, since chunksize_on_disk must have the correct dimention ordering.
        for dim_name in list_xarray_data_variables[i].dims:
            if dim_name in disk_dataset.dims:
                chunksize_on_disk.append(disk_dataset.chunks[dim_name][0])
            else:
                #Since the dimention does not exist on disk use chunking in list_xarray_data_variables[i]
                chunksize_on_disk.append(list_xarray_data_variables[i].to_dataset().chunks[dim_name][0])
                
                #Add dim to be stored
                dask_dim_array = da.from_array(list_xarray_data_variables[i][dim_name].data,chunks=(1,))
                mapper = get_mapper(outfile+'/'+ dim_name)
                list_target_zarr.append(dask.delayed(zarr.create)(
                         shape=dask_dim_array.shape,
                         compressor=compressor,
                         chunks=(1,),
                         dtype=dask_dim_array.dtype,
                         store=mapper,
                         overwrite=True))
                
                list_dask_array.append(dask_dim_array)
                list_new_dim_name.append(dim_name)
                
        #Add all other non dimentional coordinates. Order does nor matter
        for coord_name in list_xarray_data_variables[i].coords._names:
            if coord_name not in list_xarray_data_variables[i].dims:
                if coord_name not in  disk_dataset.coords._names:
                    coord_dim_names = list_xarray_data_variables[i][coord_name].dims
                    
                    coord_chunksize_on_disk = []
                    for coord_dim_name in coord_dim_names:
                        if coord_dim_name in disk_dataset.dims:
                            coord_chunksize_on_disk.append(disk_dataset.chunks[coord_dim_name][0])
                        else:
                            #Since the dimention does not exist on disk use chunking in list_xarray_data_variables[i]
                            coord_chunksize_on_disk.append(list_xarray_data_variables[i].to_dataset().chunks[coord_dim_name][0])
                    
                    #Add coord to be stored
                    dask_coord_array = list_xarray_data_variables[i][coord_name].data.rechunk(chunks=coord_chunksize_on_disk)
                    mapper = get_mapper(outfile+'/'+ coord_name)
                    list_target_zarr.append(dask.delayed(zarr.create)(
                             shape=dask_coord_array.shape,
                             compressor=compressor,
                             chunks=coord_chunksize_on_disk,
                             dtype=dask_coord_array.dtype,
                             store=mapper,
                             overwrite=True))
                    
                    list_dask_array.append(dask_coord_array)
                    
                    list_new_coord_dim_names.append(coord_dim_names)
                    list_new_coord_name.append(coord_name)

        #Rechunk the dask arrays to match the chunking on disk
        dask_array = list_xarray_data_variables[i].data.rechunk(chunksize_on_disk)
        list_dask_array.append(dask_array)
        
        #Create list of delayed objects
        mapper = get_mapper(outfile+'/'+ list_xarray_data_variables[i].name)
        list_target_zarr.append(dask.delayed(zarr.create)(
             shape=dask_array.shape,
             compressor=compressor,
             chunks=chunksize_on_disk,
             dtype=dask_array.dtype,
             store=mapper,
             overwrite=True)) #Can not specify the zarr file attributes at creation , attrs={'_ARRAY_DIMENSIONS':array_dimensions[i]} (last checked on May 2020 )
        
    
    #Trigger compute of delayed zarr.create functions in list_target_zarr.
    da.store(list_dask_array,list_target_zarr,compute=True,flush=True,lock=False)
    
    # Open zarr to add array dimension labels so that xarray.open_zarr works. This is the magic that allows xarray to understand zarr.
    dataset_group = zarr.open_group(outfile,mode='a')
    #This should one day be done during zarr.create. See https://github.com/zarr-developers/zarr-python/issues/538.
    #Data variables labels
    for i in range(n_arrays):
        dataset_group[list_xarray_data_variables[i].name].attrs['_ARRAY_DIMENSIONS'] = list_xarray_data_variables[i].dims
        
    #Dimention labels
    for new_dim_name in list_new_dim_name:
        dataset_group[new_dim_name].attrs['_ARRAY_DIMENSIONS'] = new_dim_name
        
    #Coord labels
    for new_coord_name, new_coord_dim_names in zip(list_new_coord_name, list_new_coord_dim_names):
        dataset_group[new_coord_name].attrs['_ARRAY_DIMENSIONS'] = new_coord_dim_names
    
    time_to_calc_and_store = time.time() - start
    print('Time to append and execute graph ', graph_name, time_to_calc_and_store)
    dataset_group.attrs[graph_name+'_time'] = time_to_calc_and_store
    
    #Consolidate metadata #Can be improved by only adding appended metadata
    zarr.consolidate_metadata(outfile)
    
    if bool(chunks_return):
        return xr.open_zarr(outfile,chunks=chunks_return,overwrite_encoded_chunks=True,consolidated=True)
    else:
        return xr.open_zarr(outfile,overwrite_encoded_chunks=True,consolidated=True)
