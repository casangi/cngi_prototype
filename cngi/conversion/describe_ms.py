#   Copyright 2020 AUI, Inc. Washington DC, USA
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
    import cngi._helper.table_conversion as tblconv
    from casatools import table as tb
    
    infile = os.path.expanduser(infile)  # does nothing if $HOME is unknown
    if not infile.endswith('/'): infile = infile + '/'

    # as part of MSv3 conversion, these columns in the main table are no longer needed
    ignorecols = ['FLAG_CATEGORY', 'FLAG_ROW', 'SIGMA', 'WEIGHT_SPECTRUM', 'DATA_DESC_ID']

    # figure out characteristics of main table from select subtables (must all be present)
    spw_xds = tblconv.convert_simple_table(infile, outfile='', subtable='SPECTRAL_WINDOW', ignore=ignorecols, nofile=True)
    pol_xds = tblconv.convert_simple_table(infile, outfile='', subtable='POLARIZATION', ignore=ignorecols, nofile=True)
    ddi_xds = tblconv.convert_simple_table(infile, outfile='', subtable='DATA_DESCRIPTION', ignore=ignorecols, nofile=True)
    ddis = list(ddi_xds['d0'].values)

    summary = pd.DataFrame([])
    spw_ids = ddi_xds.spectral_window_id.values
    pol_ids = ddi_xds.polarization_id.values
    chans = spw_xds.NUM_CHAN.values
    pols = pol_xds.NUM_CORR.values
    tb_tool = tb()
    tb_tool.open(infile, nomodify=True, lockoptions={'option': 'usernoread'})  # allow concurrent reads
    for ddi in ddis:
        print('processing ddi %i of %i' % (ddi+1, len(ddis)), end='\r')
        sorted_table = tb_tool.taql('select * from %s where DATA_DESC_ID = %i' % (infile, ddi))
        sdf = {'ddi': ddi, 'spw_id': spw_ids[ddi], 'pol_id': pol_ids[ddi], 'rows': sorted_table.nrows(),
               'times': len(np.unique(sorted_table.getcol('TIME'))),
               'baselines': len(np.unique(np.hstack([sorted_table.getcol(rr)[:,None] for rr in ['ANTENNA1', 'ANTENNA2']]), axis=0)),
               'chans': chans[spw_ids[ddi]],
               'pols': pols[pol_ids[ddi]]}
        sdf['size_MB'] = np.ceil((sdf['times']*sdf['baselines']*sdf['chans']*sdf['pols']*17) / 1024**2).astype(int)
        summary = pd.concat([summary, pd.DataFrame(sdf, index=[str(ddi)])], axis=0, sort=False)
        sorted_table.close()
    print(' '*50, end='\r')
    tb_tool.close()
    
    return summary.set_index('ddi').sort_index()
