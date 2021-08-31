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
#"""
#this module will NOT be included in the api
#"""


def convert_ms2(infile, outfile=None, ddis=None, ignore=['HISTORY'], compressor=None, chunks=(100, 400, 32, 1), sub_chunks=10000, append=False):
    """
    Convert legacy format MS to xarray Visibility Dataset and zarr storage format

    The CASA MSv2 format is converted to the MSv3 schema per the
    specified definition here: https://drive.google.com/file/d/10TZ4dsFw9CconBc-GFxSeb2caT6wkmza/view?usp=sharing

    The MS is partitioned by DDI, which guarantees a fixed data shape per partition. This results in different subdirectories
    under the main vis.zarr folder.  There is no DDI in MSv3, so this simply serves as a partition id in the zarr directory.

    Parameters
    ----------
    infile : str
        Input MS filename
    outfile : str
        Output zarr filename. If None, will use infile name with .vis.zarr extension
    ddis : list
        List of specific DDIs to convert. DDI's are integer values, or use 'global' string for subtables. Leave as None to convert entire MS
    ignore : list
        List of subtables to ignore (case sensitive and generally all uppercase). This is useful if a particular subtable is causing errors.
        Default is None. Note: default is now temporarily set to ignore the HISTORY table due a CASA6 issue in the table tool affecting a small
        set of test cases (set back to None if HISTORY is needed)
    compressor : numcodecs.blosc.Blosc
        The blosc compressor to use when saving the converted data to disk using zarr.
        If None the zstd compression algorithm used with compression level 2.
    chunks: 4-D tuple of ints
        Shape of desired chunking in the form of (time, baseline, channel, polarization), use -1 for entire axis in one chunk. Default is (100, 400, 20, 1)
        Note: chunk size is the product of the four numbers, and data is batch processed by time axis, so that will drive memory needed for conversion.
    sub_chunks: int
        Chunking used for subtable conversion (except for POINTING which will use time/baseline dims from chunks parameter). This is a single integer
        used for the row-axis (d0) chunking only, no other dims in the subtables will be chunked.
    append : bool
        Keep destination zarr store intact and add new DDI's to it. Note that duplicate DDI's will still be overwritten. Default False deletes and replaces
        entire directory.
    Returns
    -------
    xarray.core.dataset.Dataset
      Master xarray dataset of datasets for this visibility set
    """
    import os
    import xarray
    import dask.array as da
    import numpy as np
    import time
    import cngi._utils._table_conversion2 as tblconv
    import cngi._utils._io as xdsio
    import warnings
    import importlib_metadata
    warnings.filterwarnings('ignore', category=FutureWarning)

    # parse filename to use
    infile = os.path.expanduser(infile)
    prefix = infile[:infile.rindex('.')]
    if outfile is None: outfile = prefix + '.vis.zarr'
    outfile = os.path.expanduser(outfile)

    # need to manually remove existing zarr file (if any)
    if not append:
        os.system("rm -fr " + outfile)
        os.system("mkdir " + outfile)

    # as part of MSv3 conversion, these columns in the main table are no longer needed
    ignorecols = ['FLAG_CATEGORY', 'FLAG_ROW', 'DATA_DESC_ID']
    if ignore is None: ignore = []

    # we need to assume an explicit ordering of dims
    dimorder = ['time', 'baseline', 'chan', 'pol']

    # we need the spectral window, polarization, and data description tables for processing the main table
    spw_xds = tblconv.convert_simple_table(infile, outfile='', subtable='SPECTRAL_WINDOW', ignore=ignorecols, nofile=True, add_row_id=True)
    pol_xds = tblconv.convert_simple_table(infile, outfile='', subtable='POLARIZATION', ignore=ignorecols, nofile=True)
    ddi_xds = tblconv.convert_simple_table(infile, outfile='', subtable='DATA_DESCRIPTION', ignore=ignorecols, nofile=True)

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
        print('Processing ddi %i...'% ddi, end='')
        start_ddi = time.time()

        # these columns are different / absent in MSv3 or need to be handled as special cases
        msv2 = ['WEIGHT', 'WEIGHT_SPECTRUM', 'SIGMA', 'SIGMA_SPECTRUM', 'ANTENNA1', 'ANTENNA2', 'UVW']

        # convert columns that are common to MSv2 and MSv3
        xds = tblconv.convert_expanded_table(infile, os.path.join(outfile, 'xds' + str(ddi)), subsel=ddi, ignore=ignorecols + msv2,
                                             compressor=compressor, chunks=chunks, nofile=False)
        if len(xds.dims) == 0: continue

        # convert and append UVW separately so we can handle its special dimension
        uvw_xds = tblconv.convert_expanded_table(infile, os.path.join(outfile, 'tmp'), subsel=ddi, ignore=ignorecols+list(xds.data_vars)+msv2[:-1],
                                                 compressor=compressor, chunks=(chunks[0], chunks[1], 3), nofile=False).rename({'chan':'uvw'})

        uvw_xds.to_zarr(os.path.join(outfile, 'xds' + str(ddi)), mode='a', compute=True, consolidated=True)

        # convert and append the ANTENNA1 and ANTENNA2 columns separately so we can squash the unnecessary time dimension
        ant_xds = tblconv.convert_expanded_table(infile, os.path.join(outfile, 'tmp'), subsel=ddi, compressor=compressor, chunks=chunks[:2],
                                                 ignore=ignorecols + list(xds.data_vars) + msv2[:4] + ['UVW'], nofile=False)

        ant_xds = ant_xds.assign({'ANTENNA1': ant_xds.ANTENNA1.max(axis=0), 'ANTENNA2': ant_xds.ANTENNA2.max(axis=0)}).drop_dims('time')
        ant_xds.to_zarr(os.path.join(outfile, 'xds' + str(ddi)), mode='a', compute=True, consolidated=True)

        # now convert just the WEIGHT and WEIGHT_SPECTRUM (if preset)
        # WEIGHT needs to be expanded to full dimensionality (time, baseline, chan, pol)
        wt_xds = tblconv.convert_expanded_table(infile, os.path.join(outfile, 'tmp'), subsel=ddi, compressor=compressor, chunks=chunks,
                                                ignore=ignorecols + list(xds.data_vars) + msv2[-3:], nofile=False).rename({'chan':'pol'})

        # MSv3 changes to weight/sigma column handling
        # 1. DATA_WEIGHT = 1/sqrt(SIGMA)
        # 2. CORRECTED_DATA_WEIGHT = WEIGHT
        # 3. if SIGMA_SPECTRUM or WEIGHT_SPECTRUM present, use them instead of SIGMA and WEIGHT
        # 4. discard SIGMA, WEIGHT, SIGMA_SPECTRUM and WEIGHT_SPECTRUM from converted ms
        # 5. set shape of DATA_WEIGHT / CORRECTED_DATA_WEIGHT to (time, baseline, chan, pol) padding as necessary
        if 'DATA' in xds.data_vars:
            if 'SIGMA_SPECTRUM' in wt_xds.data_vars:
                wt_xds = wt_xds.rename(dict(zip(wt_xds.SIGMA_SPECTRUM.dims, dimorder))).assign({'DATA_WEIGHT': 1 / wt_xds.SIGMA_SPECTRUM ** 2})
            elif 'SIGMA' in wt_xds.data_vars:
                wts = wt_xds.SIGMA.shape[:2] + (1,) + (wt_xds.SIGMA.shape[-1],)
                wt_da = da.tile(da.reshape(wt_xds.SIGMA.data, wts), (1, 1, len(xds.chan), 1)).rechunk(chunks)
                wt_xds = wt_xds.assign({'DATA_WEIGHT': xarray.DataArray(1 / wt_da ** 2, dims=dimorder)})
        if 'CORRECTED_DATA' in xds.data_vars:
            if 'WEIGHT_SPECTRUM' in wt_xds.data_vars:
                wt_xds = wt_xds.rename(dict(zip(wt_xds.WEIGHT_SPECTRUM.dims, dimorder))).assign({'CORRECTED_DATA_WEIGHT': wt_xds.WEIGHT_SPECTRUM})
            elif 'WEIGHT' in wt_xds.data_vars:
                wts = wt_xds.WEIGHT.shape[:2] + (1,) + (wt_xds.WEIGHT.shape[-1],)
                wt_da = da.tile(da.reshape(wt_xds.WEIGHT.data, wts), (1, 1, len(xds.chan), 1)).rechunk(chunks)
                wt_xds = wt_xds.assign({'CORRECTED_DATA_WEIGHT': xarray.DataArray(wt_da, dims=dimorder)})

        wt_xds = wt_xds.drop([cc for cc in msv2 if cc in wt_xds.data_vars])
        wt_xds.to_zarr(os.path.join(outfile, 'xds' + str(ddi)), mode='a', compute=True, consolidated=True)

        # add in relevant data grouping, spw and polarization attributes
        attrs = {'data_groups': [{}]}
        if ('DATA' in xds.data_vars) and ('DATA_WEIGHT' in wt_xds.data_vars):
            attrs['data_groups'][0][str(len(attrs['data_groups'][0]))] = {'id': str(len(attrs['data_groups'][0])), 'data': 'DATA',
                                                                          'uvw': 'UVW', 'flag': 'FLAG', 'weight': 'DATA_WEIGHT'}
        if ('CORRECTED_DATA' in xds.data_vars) and ('CORRECTED_DATA_WEIGHT' in wt_xds.data_vars):
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
        aux_xds = xarray.Dataset(coords=coords, attrs=attrs)

        aux_xds.to_zarr(os.path.join(outfile, 'xds' + str(ddi)), mode='a', compute=True, consolidated=True)
        xds = xarray.open_zarr(os.path.join(outfile, 'xds' + str(ddi)))

        xds_list += [('xds' + str(ddi), xds)]
        print('Completed ddi %i  process time {:0.2f} s'.format(time.time() - start_ddi) % ddi)

    # clean up the tmp directory created by the weight conversion to MSv3
    os.system("rm -fr " + os.path.join(outfile, 'tmp'))

    # convert other subtables to their own partitions, denoted by 'global_' prefix
    skip_tables = ['DATA_DESCRIPTION', 'SORTED_TABLE'] + ignore
    subtables = sorted([tt for tt in os.listdir(infile) if os.path.isdir(os.path.join(infile, tt)) and tt not in skip_tables])
    if 'global' in ddis:
        start_ddi = time.time()
        for ii, subtable in enumerate(subtables):
            print('processing subtable %i of %i : %s' % (ii, len(subtables), subtable), end='\r')
            if subtable == 'POINTING':  # expand the dimensions of the pointing table
                xds_sub_list = [(subtable, tblconv.convert_expanded_table(os.path.join(infile, subtable), os.path.join(outfile, 'global/POINTING'),
                                                                          compressor=compressor, chunks=chunks, nofile=False))]
            else:
                add_row_id = (subtable in ['ANTENNA', 'FIELD', 'OBSERVATION', 'SCAN', 'SPECTRAL_WINDOW', 'STATE'])
                xds_sub_list = [(subtable, tblconv.convert_simple_table(infile, os.path.join(outfile, 'global'), subtable,
                                                                        timecols=['TIME'], ignore=ignorecols, compressor=compressor,
                                                                        nofile=False, chunks=(sub_chunks, -1), add_row_id=add_row_id))]

            if len(xds_sub_list[-1][1].dims) != 0:
                xds_list += xds_sub_list

        print('Completed subtables  process time {:0.2f} s'.format(time.time() - start_ddi))

    # write sw version that did this conversion to zarr directory
    try:
        version = importlib_metadata.version('cngi-prototype')
    except:
        version = '0.0.0'

    with open(outfile + '/.version', 'w') as fid:
        fid.write('cngi-protoype ' + version + '\n')

    # build the master xds to return
    mxds = xdsio.vis_xds_packager(xds_list)
    print(' ' * 50)

    return mxds
