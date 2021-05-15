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
def describe_ms(infile):
    """
    Summarize the contents of an MS directory in casacore table format

    Parameters
    ----------
    infile : str
        input filename of MS

    Returns
    -------
    pandas.core.frame.DataFrame
        Summary information
    """
    import os
    import pandas as pd
    import numpy as np
    import cngi._utils._table_conversion2 as tblconv
    from casacore import tables
    
    infile = os.path.expanduser(infile)  # does nothing if $HOME is unknown
    if not infile.endswith('/'): infile = infile + '/'

    # as part of MSv3 conversion, these columns in the main table are no longer needed
    ignorecols = ['FLAG_CATEGORY', 'FLAG_ROW', 'SIGMA', 'WEIGHT_SPECTRUM', 'DATA_DESC_ID']

    # figure out characteristics of main table from select subtables (must all be present)
    spw_xds = tblconv.read_simple_table(infile, subtable='SPECTRAL_WINDOW', ignore=ignorecols, add_row_id=True)
    pol_xds = tblconv.read_simple_table(infile, subtable='POLARIZATION', ignore=ignorecols)
    ddi_xds = tblconv.read_simple_table(infile, subtable='DATA_DESCRIPTION', ignore=ignorecols)
    ddis = list(ddi_xds['d0'].values)

    summary = pd.DataFrame([])
    spw_ids = ddi_xds.spectral_window_id.values
    pol_ids = ddi_xds.polarization_id.values
    chans = spw_xds.NUM_CHAN.values
    pols = pol_xds.NUM_CORR.values

    for ddi in ddis:
        print('processing ddi %i of %i' % (ddi+1, len(ddis)), end='\r')
        sorted_table = tables.taql('select * from %s where DATA_DESC_ID = %i' % (infile, ddi))
        sdf = {'ddi': ddi, 'spw_id': spw_ids[ddi], 'pol_id': pol_ids[ddi], 'rows': sorted_table.nrows(),
               'times': len(np.unique(sorted_table.getcol('TIME'))),
               'baselines': len(np.unique(np.hstack([sorted_table.getcol(rr)[:,None] for rr in ['ANTENNA1', 'ANTENNA2']]), axis=0)),
               'chans': chans[spw_ids[ddi]],
               'pols': pols[pol_ids[ddi]]}
        sdf['size_MB'] = np.ceil((sdf['times']*sdf['baselines']*sdf['chans']*sdf['pols']*9) / 1024**2).astype(int)
        summary = pd.concat([summary, pd.DataFrame(sdf, index=[str(ddi)])], axis=0, sort=False)
        sorted_table.close()
    print(' '*50, end='\r')

    return summary.set_index('ddi').sort_index()
