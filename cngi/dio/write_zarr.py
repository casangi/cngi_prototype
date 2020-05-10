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


def write_zarr(dataset, outfile=None, compressor=None, graph_name='save_zarr'):
    """
    Write xarray dataset to zarr format on disk.

    Parameters
    ----------
    dataset : xarray.core.dataset.Dataset
        Dataset to write to disk
    outfile : str
        output filename, generally ends in .zarr
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
    import zarr as zr
    import time
    from numcodecs import Blosc
    from itertools import cycle
    
    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
    
    encoding = dict(zip(list(dataset.data_vars), cycle([{'compressor': compressor}])))
    start = time.time()
    xr.Dataset.to_zarr(dataset, store=outfile, mode='w', encoding=encoding,consolidated=True)
    time_to_calc_and_store = time.time() - start
    print('Time to store and execute graph ', graph_name, time_to_calc_and_store)
    
    dataset_group = zr.open_group(outfile,mode='a')
    dataset_group.attrs[graph_name+'_time'] = time_to_calc_and_store
    
    dataset = xr.open_zarr(outfile,consolidated=True,overwrite_encoded_chunks=True)
    return dataset
