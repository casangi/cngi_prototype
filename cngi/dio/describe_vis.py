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
    summary = pd.DataFrame([])
    parts = [dd for dd in os.listdir(infile) if os.path.isdir(os.path.join(infile, str(dd)))]
    for ii, part in enumerate(parts):
        if part.startswith('global'): continue
        print('processing partition %i of %i' % (ii+1, len(parts)), end='\r')
        xds = open_zarr(os.path.join(infile, str(part)))
        sdf = {'xds': part, 'spw_id':xds.spw_id.values[0], 'pol_id':xds.pol_id.values[0],
               'times': len(xds.time),
               'baselines': len(xds.baseline),
               'chans': len(xds.chan),
               'pols': len(xds.pol),
               'size_MB': np.ceil(xds.nbytes / 1024 ** 2).astype(int)}
        summary = pd.concat([summary, pd.DataFrame(sdf, index=[ii])], axis=0, sort=False)
    
    print(' '*50, end='\r')
    return summary.set_index('xds').sort_index()
