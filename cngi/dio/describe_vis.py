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
    import pandas as pd
    from xarray import open_zarr
    
    infile = os.path.expanduser(infile)  # does nothing if $HOME is unknown
    summary = pd.DataFrame([])
    for ii, ddi in enumerate(os.listdir(infile)):
        if ddi == 'global': continue
        dpath = os.path.join(infile, str(ddi))
        xds = open_zarr(dpath)
        sdf = {'ddi': ddi, 'spw_id':xds.spw.values[0], 'size_GB': xds.nbytes / 1024 ** 3, 'channels':len(xds.chan), 'times': len(xds.time),
               'baselines':len(xds.baseline), 'fields':len(xds.field)}
        summary = pd.concat([summary, pd.DataFrame(sdf, index=[ii])], axis=0, sort=False)
    
    return summary.set_index('ddi')
