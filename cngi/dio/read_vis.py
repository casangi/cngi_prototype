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
def read_vis(infile, partition=None, chunks=None, consolidated=True, overwrite_encoded_chunks=True):
    """
    Read zarr format Visibility data from disk to xarray Dataset

    Parameters
    ----------
    infile : str
        input Visibility filename
    partition : string or list
        name of partition(s) to read as returned by describe_vis. Multiple partitions in list form will return a master dataset of datasets.
        Use 'global' for global metadata. Default None returns everything
    chunks : dict
        sets specified chunk size per dimension. Dict is in the form of 'dim':chunk_size, for example {'time':100, 'baseline':400, 'chan':32, 'pol':1}.
        Default None uses the original zarr chunking.
    consolidated : bool
        use zarr consolidated metadata capability. Only works for stores that have already been consolidated. Default True works with datasets
        produced by convert_ms which automatically consolidates metadata.
    overwrite_encoded_chunks : bool
        drop the zarr chunks encoded for each variable when a dataset is loaded with specified chunk sizes.  Default True, only applies when chunks
        is not None.

    Returns
    -------
    xarray.core.dataset.Dataset
        New xarray Dataset of Visibility data contents
    """
    import os
    import numpy as np
    import cngi._utils._io as xdsio
    from xarray import open_zarr

    infile = os.path.expanduser(infile)
    if partition is None: partition = os.listdir(infile)
    partition = np.atleast_1d(partition)

    if chunks is None:
        chunks = 'auto'
        overwrite_encoded_chunks = False

    if ('global' in partition) and (os.path.isdir(os.path.join(infile,'global'))):
        global_dirs = sorted(['global/'+tt for tt in os.listdir(os.path.join(infile,'global'))])
        partition = np.hstack((np.delete(partition, np.where(partition == 'global')), global_dirs))
  
    if len(partition) == 1:
        xds = open_zarr(os.path.join(infile, str(partition[0])), chunks=chunks, consolidated=consolidated,
                        overwrite_encoded_chunks=overwrite_encoded_chunks)
    else:
        xds_list = []
        for part in partition:
            if os.path.isdir(os.path.join(infile, str(part))):
                xds_list += [(part.replace('global/',''), open_zarr(os.path.join(infile, str(part)), chunks=chunks, consolidated=consolidated,
                                                                    overwrite_encoded_chunks=overwrite_encoded_chunks))]
        # build the master xds to return
        xds = xdsio.vis_xds_packager(xds_list)

    return xds

