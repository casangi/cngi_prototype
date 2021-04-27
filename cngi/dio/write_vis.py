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


'''
To Do:
1. zarr.consolidate_metadata(outfile) is very slow for a zarr group (datatset) with many chunks (there is a python for loop that checks each file). We might have to implement our own version. This is also important for cngi.dio.append_zarr
'''

def write_vis(mxds, outfile, chunks_on_disk=None, partition=None, consolidated=True, compressor=None, graph_name='write_zarr'):
    """
    Write xarray dataset to zarr format on disk. When chunks_on_disk is not specified the chunking in the input dataset is used.
    When chunks_on_disk is specified that dataset is saved using that chunking.

    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        Dataset of dataset to write to disk
    outfile : str
        outfile filename, generally ends in .zarr
    chunks_on_disk : dict of int
        A dictionary with the chunk size that will be used when writing to disk. For example {'time': 20, 'chan': 6}.
        If chunks_on_disk is not specified the chunking of dataset will be used.
    partition : str or list
        Name of partition xds to write into outfile (from the mxds attributes section). Overwrites existing partition of same name.
        Default None writes entire mxds
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
    import os
    import numpy as np

    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)

    if partition is None:
        partition = list(mxds.attrs.keys())
    partition = list(np.atleast_1d(partition))
        
    os.system("rm -fr " + outfile)
    os.system("mkdir " + outfile)
        
    for xds_name in partition:
        if "xds" in xds_name:
            xds_outfile = outfile + '/' + xds_name
            xds_for_disk = mxds.attrs[xds_name]
            if chunks_on_disk is not None:
                xds_for_disk = xds_for_disk.chunk(chunks=chunks_on_disk)
        else:
            xds_outfile = outfile + '/global/' + xds_name
            xds_for_disk = mxds.attrs[xds_name]
            
        # Create compression encoding for each datavariable
        encoding = dict(zip(list(xds_for_disk.data_vars), cycle([{'compressor': compressor}])))
        start = time.time()

        # Consolidated is set to False so that the timing information is included in the consolidate metadata.
        xr.Dataset.to_zarr(xds_for_disk, store=xds_outfile, mode='w', encoding=encoding,consolidated=False)
        time_to_calc_and_store = time.time() - start
        print('Time to store and execute graph for ', xds_name, graph_name, time_to_calc_and_store)

        #Add timing information
        dataset_group = zarr.open_group(xds_outfile, mode='a')
        dataset_group.attrs[graph_name+'_time'] = time_to_calc_and_store
            
        if consolidated == True:
            zarr.consolidate_metadata(xds_outfile)

