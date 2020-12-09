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
def read_vis(infile, partition='part0'):
    """
    Read zarr format Visibility data from disk to xarray Dataset

    Parameters
    ----------
    infile : str
        input Visibility filename
    partition : string or list
        name of partition(s) to read as returned by describe_vis. Multiple partitions in list form will return a dataset of datasets.
        Use 'global' for global metadata. None returns everything. Default is 'part0'

    Returns
    -------
    xarray.core.dataset.Dataset
        New xarray Dataset of Visibility data contents
    """
    import os
    import numpy as np
    import cngi._helper.io as xdsio
    from xarray import open_zarr

    infile = os.path.expanduser(infile)
    if partition is None: partition = os.listdir(infile)
    partition = np.atleast_1d(partition)

    if 'global' in partition:
        globals = sorted([tt for tt in os.listdir(infile) if os.path.isdir(os.path.join(infile, tt)) and tt.startswith('global_')])
        partition = np.hstack((np.delete(partition, np.where(partition == 'global')), globals))
  
    if len(partition) == 1:
        xds = open_zarr(os.path.join(infile, str(partition[0])))
    else:
        xds_list = []
        for part in partition:
            if os.path.isdir(os.path.join(infile, str(part))):
                xds_list += [(part.replace('global_',''), open_zarr(os.path.join(infile, str(part))))]
        # build the master xds to return
        xds = xdsio.vis_xds_packager(xds_list)

    return xds

