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
def describe_vis(infile):
    """
    Summarize the contents of a zarr format Visibility directory on disk

    Parameters
    ----------
    infile : str
        input filename of zarr Visibility data

    Returns
    -------
    pandas.core.frame.DataFrame
        Summary information
    """
    import os
    import numpy as np
    import pandas as pd
    from xarray import open_zarr
    
    infile = os.path.expanduser(infile)  # does nothing if $HOME is unknown
    ddis = list(np.array(os.listdir(infile), dtype=int))
    
    summary = pd.DataFrame([])
    for ii, ddi in enumerate(ddis):
        dpath = os.path.join(infile, str(ddi))
        xds = open_zarr(dpath)
        sdf = {'ddi': ddi, 'size_GB': xds.nbytes / 1024 ** 3}
        sdf.update(dict(xds.dims))
        summary = pd.concat([summary, pd.DataFrame(sdf, index=[ii])], axis=0, sort=False)
    
    return summary
