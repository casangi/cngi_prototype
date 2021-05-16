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


def read_ms(infile, ddis=None, ignore=None, chunks=(400, 400, 64, 2)):
    """
    Read legacy format MS to xarray Visibility Dataset

    The CASA MSv2 format is converted to the MSv3 schema per the
    specified definition here: https://drive.google.com/file/d/10TZ4dsFw9CconBc-GFxSeb2caT6wkmza/view?usp=sharing

    The MS is partitioned by DDI, which guarantees a fixed data shape per partition. This results in separate xarray
    dataset (xds) partitions contained within a main xds (mxds).  There is no DDI in MSv3, so this simply serves as
    a partition id for each xds.

    Parameters
    ----------
    infile : str
        Input MS filename
    ddis : list
        List of specific DDIs to read. DDI's are integer values, or use 'global' string for subtables. Leave as None to read entire MS
    ignore : list
        List of subtables to ignore (case sensitive and generally all uppercase). This is useful if a particular subtable is causing errors
        or is very large and slowing down reads. Default is None
    chunks: 4-D tuple of ints
        Shape of desired chunking in the form of (time, baseline, channel, polarization). Larger values reduce the number of chunks and
        speed up the reads at the cost of more memory. Chunk size is the product of the four numbers. Default is (400, 400, 64, 2)

    Returns
    -------
    xarray.core.dataset.Dataset
      Main xarray dataset of datasets for this visibility set
    """
    import os
    import xarray
    import dask.array as da
    import numpy as np
    import cngi._utils._table_conversion2 as tblconv
    import cngi._utils._io as xdsio
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    # parse filename to use
    infile = os.path.expanduser(infile)

    # as part of MSv3 conversion, these columns in the main table are no longer needed
    ignorecols = ['FLAG_CATEGORY', 'FLAG_ROW', 'DATA_DESC_ID']
    if ignore is None: ignore = []

    # we need to assume an explicit ordering of dims
    dimorder = ['time', 'baseline', 'chan', 'pol']

    # we need the spectral window, polarization, and data description tables for processing the main table
    spw_xds = tblconv.read_simple_table(infile, subtable='SPECTRAL_WINDOW', ignore=ignorecols, add_row_id=True)
    pol_xds = tblconv.read_simple_table(infile, subtable='POLARIZATION', ignore=ignorecols)
    ddi_xds = tblconv.read_simple_table(infile, subtable='DATA_DESCRIPTION', ignore=ignorecols)

    # let's assume that each DATA_DESC_ID (ddi) is a fixed shape that may differ from others
    # form a list of ddis to process, each will be placed it in its own xarray dataset and partition
    if ddis is None:
        ddis = list(ddi_xds['d0'].values) + ['global']
    else:
        ddis = np.atleast_1d(ddis)
    xds_list = []

    ####################################################################
    # process each selected DDI from the input MS, assume a fixed shape within the ddi (should always be true)
    # each DDI is written to its own subdirectory under the parent folder
    for ddi in ddis:
        if ddi == 'global': continue  # handled afterwards
        ddi = int(ddi)

        # convert columns that are common to MSv2 and MSv3
        xds = tblconv.read_main_table(infile, subsel=ddi, ignore=ignorecols, chunks=chunks)
        if len(xds.dims) == 0: continue

        # convert and append the ANTENNA1 and ANTENNA2 columns separately so we can squash the unnecessary time dimension
        xds = xds.assign({'ANTENNA1': xds.ANTENNA1.max(axis=0), 'ANTENNA2': xds.ANTENNA2.max(axis=0)})

        # MSv3 changes to weight/sigma column handling
        # 1. DATA_WEIGHT = 1/sqrt(SIGMA)
        # 2. CORRECTED_DATA_WEIGHT = WEIGHT
        # 3. if SIGMA_SPECTRUM or WEIGHT_SPECTRUM present, use them instead of SIGMA and WEIGHT
        # 4. discard SIGMA, WEIGHT, SIGMA_SPECTRUM and WEIGHT_SPECTRUM from converted ms
        # 5. set shape of DATA_WEIGHT / CORRECTED_DATA_WEIGHT to (time, baseline, chan, pol) padding as necessary
        if 'DATA' in xds.data_vars:
            if 'SIGMA_SPECTRUM' in xds.data_vars:
                xds = xds.assign({'DATA_WEIGHT': 1 / xds.SIGMA_SPECTRUM**2}).drop('SIGMA_SPECTRUM')
            elif 'SIGMA' in xds.data_vars:
                wts = xds.SIGMA.shape[:2] + (1,) + (xds.SIGMA.shape[-1],)
                wt_da = da.tile(da.reshape(xds.SIGMA.data, wts), (1, 1, len(xds.chan), 1)).rechunk(chunks)
                xds = xds.assign({'DATA_WEIGHT': xarray.DataArray(1/wt_da**2, dims=dimorder)})
        if 'CORRECTED_DATA' in xds.data_vars:
            if 'WEIGHT_SPECTRUM' in xds.data_vars:
                xds = xds.rename({'WEIGHT_SPECTRUM':'CORRECTED_DATA_WEIGHT'})
            elif 'WEIGHT' in xds.data_vars:
                wts = xds.WEIGHT.shape[:2] + (1,) + (xds.WEIGHT.shape[-1],)
                wt_da = da.tile(da.reshape(xds.WEIGHT.data, wts), (1, 1, len(xds.chan), 1)).rechunk(chunks)
                xds = xds.assign({'CORRECTED_DATA_WEIGHT': xarray.DataArray(wt_da, dims=dimorder)}).drop('WEIGHT')

        xds = xds.drop_vars(['WEIGHT', 'SIGMA', 'SIGMA_SPECTRUM', 'WEIGHT_SPECTRUM'], errors='ignore')

        # add in relevant data grouping, spw and polarization attributes
        attrs = {'data_groups': [{}]}
        if ('DATA' in xds.data_vars) and ('DATA_WEIGHT' in xds.data_vars):
            attrs['data_groups'][0][str(len(attrs['data_groups'][0]))] = {'id': str(len(attrs['data_groups'][0])), 'data': 'DATA',
                                                                          'uvw': 'UVW', 'flag': 'FLAG', 'weight': 'DATA_WEIGHT'}
        if ('CORRECTED_DATA' in xds.data_vars) and ('CORRECTED_DATA_WEIGHT' in xds.data_vars):
            attrs['data_groups'][0][str(len(attrs['data_groups'][0]))] = {'id': str(len(attrs['data_groups'][0])), 'data': 'CORRECTED_DATA',
                                                                          'uvw': 'UVW', 'flag': 'FLAG', 'weight': 'CORRECTED_DATA_WEIGHT'}

        for dv in spw_xds.data_vars:
            attrs[dv.lower()] = spw_xds[dv].values[ddi_xds['spectral_window_id'].values[ddi]]
            attrs[dv.lower()] = int(attrs[dv.lower()]) if type(attrs[dv.lower()]) is np.bool_ else attrs[dv.lower()]  # convert bools
        for dv in pol_xds.data_vars:
            attrs[dv.lower()] = pol_xds[dv].values[ddi_xds['polarization_id'].values[ddi]]
            attrs[dv.lower()] = int(attrs[dv.lower()]) if type(attrs[dv.lower()]) is np.bool_ else attrs[dv.lower()]  # convert bools

        # grab the channel frequency values from the spw table data and pol idxs from the polarization table, add spw and pol ids
        chan = attrs.pop('chan_freq')[:len(xds.chan)]
        pol = attrs.pop('corr_type')[:len(xds.pol)]

        # truncate per-chan values to the actual number of channels and move to coordinates
        chan_width = xarray.DataArray(da.from_array(attrs.pop('chan_width')[:len(xds.chan)], chunks=chunks[2]), dims=['chan'])
        effective_bw = xarray.DataArray(da.from_array(attrs.pop('effective_bw')[:len(xds.chan)], chunks=chunks[2]), dims=['chan'])
        resolution = xarray.DataArray(da.from_array(attrs.pop('resolution')[:len(xds.chan)], chunks=chunks[2]), dims=['chan'])

        coords = {'chan': chan, 'pol': pol, 'spw_id': [ddi_xds['spectral_window_id'].values[ddi]], 'pol_id': [ddi_xds['polarization_id'].values[ddi]],
                  'chan_width': chan_width, 'effective_bw': effective_bw, 'resolution': resolution}
        xds = xds.assign_coords(coords).assign_attrs(attrs)
        xds_list += [('xds' + str(ddi), xds)]

    # read other subtables
    skip_tables = ['DATA_DESCRIPTION', 'SORTED_TABLE'] + ignore
    subtables = sorted([tt for tt in os.listdir(infile) if os.path.isdir(os.path.join(infile, tt)) and tt not in skip_tables])
    if 'global' in ddis:
        for ii, subtable in enumerate(subtables):
            if subtable == 'POINTING':  # expand the dimensions of the pointing table
                sxds = tblconv.read_pointing_table(os.path.join(infile, subtable), chunks=chunks[:2]+(20,20))
            else:
                add_row_id = (subtable in ['ANTENNA', 'FIELD', 'OBSERVATION', 'SCAN', 'SPECTRAL_WINDOW', 'STATE'])
                sxds = tblconv.read_simple_table(infile, subtable=subtable, timecols=['TIME'], ignore=ignorecols, add_row_id=add_row_id)
            if len(sxds.dims) != 0: xds_list += [(subtable, sxds)]

    # build the master xds to return
    mxds = xdsio.vis_xds_packager(xds_list)
    return mxds
