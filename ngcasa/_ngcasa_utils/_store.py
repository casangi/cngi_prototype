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

def _store(dataset,list_xarray_data_variables,storage_parms):
    """
    Parameters
    ----------
    Returns
    -------
    """
    from cngi.dio import write_zarr, append_zarr
    
    if  storage_parms['to_disk']:
        
        #If no chunks_return is specified use the dask chunking (not chunking on disk)
        #Must be a beter way to convert a sortedDict to dict
        if not bool(storage_parms['chunks_return']):
            chunks_return = {}
            for dim_key in dataset.chunks:
                chunks_return[dim_key] = dataset.chunks[dim_key][0]
            storage_parms['chunks_return'] = chunks_return
    
        if storage_parms['append']:
            data_variables_name = ""
            for data_variable in list_xarray_data_variables:
                data_variables_name = data_variables_name + ", " + data_variable.name
            print('Atempting to add ', data_variables_name, ' to ', storage_parms['outfile'])
            
            #try:
            if True:
                stored_dataset = append_zarr(list_xarray_data_variables,outfile=storage_parms['outfile'],chunks_return=storage_parms['chunks_return'],compressor=storage_parms['compressor'],graph_name=storage_parms['graph_name'])
                print('##################### Finished appending ',storage_parms['graph_name'],' #####################')
                return stored_dataset
            #except Exception:
            #    print('ERROR : Could not append ', data_variables_name , 'to', storage_parms['outfile'])
        else:
            print('Saving dataset to ', storage_parms['outfile'])
            
            stored_dataset = write_zarr(dataset, outfile=storage_parms['outfile'], chunks_return=storage_parms['chunks_return'], chunks_on_disk=storage_parms['chunks_on_disk'], compressor=storage_parms['compressor'], graph_name=storage_parms['graph_name'])
            
            print('##################### Created new dataset with',storage_parms['graph_name'],'#####################')
            return stored_dataset
    
    print('##################### Created graph for',storage_parms['graph_name'],'#####################')
    return dataset
