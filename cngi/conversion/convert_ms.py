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


def convert_ms(infile, outfile=None, ddis=None, ignorecols=None, compressor=None, chunk_shape=(100, 400, 32, 1), append=False, nofile=False):
    """
    Convert legacy format MS to xarray Visibility Dataset and zarr storage format

    This function requires CASA6 casatools module. The CASA MSv2 format is converted to the MSv3 schema per the
    specified definition here: https://drive.google.com/file/d/10TZ4dsFw9CconBc-GFxSeb2caT6wkmza/view?usp=sharing
    
    The MS is partitioned by DDI, which guarentees a fixed data shape per partition. This results in different subdirectories
    under the main vis.zarr folder.  There is no DDI in MSv3, so this simply serves as a partition id in the zarr directory.

    Parameters
    ----------
    infile : str
        Input MS filename
    outfile : str
        Output zarr filename. If None, will use infile name with .vis.zarr extension
    ddis : list
        List of specific DDIs to convert. DDI's are integer values, or use 'global' string for subtables. Leave as None to convert entire MS
    ignorecols : list
        List of columns to ignore. This is useful if a particular column is causing errors, but note that it applies to all tables. Default is None
    compressor : numcodecs.blosc.Blosc
        The blosc compressor to use when saving the converted data to disk using zarr.
        If None the zstd compression algorithm used with compression level 2.
    chunk_shape: 4-D tuple of ints
        Shape of desired chunking in the form of (time, baseline, channel, polarization), use -1 for entire axis in one chunk. Default is (100, 400, 20, 1)
        Note: chunk size is the product of the four numbers, and data is batch processed by time axis, so that will drive memory needed for conversion.
    append : bool
        Keep destination zarr store intact and add new DDI's to it. Note that duplicate DDI's will still be overwritten. Default False deletes and replaces
        entire directory.
    nofile : bool
        Allows legacy MS to be directly read without file conversion. If set to true, no output file will be written and entire MS will be held in memory.
        Requires ~4x the memory of the MS size.  Default is False
    Returns
    -------
    xarray.core.dataset.Dataset
      Master xarray dataset of datasets for this visibility set
    """
    import os
    import xarray
    import numpy as np
    import time
    import cngi._helper.table_conversion as tblconv
    import cngi._helper.io as xdsio
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    # parse filename to use
    infile = os.path.expanduser(infile)
    prefix = infile[:infile.rindex('.')]
    if outfile is None: outfile = prefix + '.vis.zarr'
    outfile = os.path.expanduser(outfile)

    # need to manually remove existing zarr file (if any)
    if (not nofile) and (not append):
        os.system("rm -fr " + outfile)
        os.system("mkdir " + outfile)
    
    # as part of MSv3 conversion, these columns in the main table are no longer needed
    if ignorecols is None: ignorecols = []
    ignorecols = ignorecols + ['FLAG_CATEGORY', 'FLAG_ROW', 'WEIGHT_SPECTRUM', 'DATA_DESC_ID']
    
    # we need the spectral window, polarization, and data description tables for processing the main table
    spw_xds = tblconv.convert_simple_table(infile, outfile='', subtable='SPECTRAL_WINDOW', ignore=ignorecols, nofile=True)
    pol_xds = tblconv.convert_simple_table(infile, outfile='', subtable='POLARIZATION', ignore=ignorecols, nofile=True)
    ddi_xds = tblconv.convert_simple_table(infile, outfile='', subtable='DATA_DESCRIPTION', ignore=ignorecols, nofile=True)

    # let's assume that each DATA_DESC_ID (ddi) is a fixed shape that may differ from others
    # form a list of ddis to process, each will be placed it in its own xarray dataset and partition
    if ddis is None: ddis = list(ddi_xds['d0'].values) + ['global']
    xds_list = []
    
    ####################################################################
    # process each selected DDI from the input MS, assume a fixed shape within the ddi (should always be true)
    # each DDI is written to its own subdirectory under the parent folder
    for ddi in ddis:
        if ddi == 'global': continue  # handled afterwards
        print('Processing ddi', ddi, end='\r')
        start_ddi = time.time()

        xds = tblconv.convert_expanded_table(infile, os.path.join(outfile,'part'+str(ddi)), keys={'TIME': 'time', ('ANTENNA1', 'ANTENNA2'): 'baseline'},
                                             subsel={'DATA_DESC_ID':ddi}, timecols=['time'], dimnames={'d2':'uvw_index', 'd3':'chan', 'd4':'pol'},
                                             ignore=ignorecols + ['SIGMA'], compressor=compressor, chunk_shape=chunk_shape, nofile=nofile)
        
        # add in relevant spw and polarization attributes
        attrs = {}
        for dv in spw_xds.data_vars:
            attrs[dv.lower()] = spw_xds[dv].values[ddi_xds['SPECTRAL_WINDOW_ID'].values[ddi]]
            attrs[dv.lower()] = int(attrs[dv.lower()]) if type(attrs[dv.lower()]) is np.bool_ else attrs[dv.lower()]  # convert bools
        for dv in pol_xds.data_vars:
            attrs[dv.lower()] = pol_xds[dv].values[ddi_xds['POLARIZATION_ID'].values[ddi]]
            attrs[dv.lower()] = int(attrs[dv.lower()]) if type(attrs[dv.lower()]) is np.bool_ else attrs[dv.lower()]  # convert bools

        # grab the channel frequency values from the spw table data and pol idxs from the polarization table, add spw and pol ids
        chan = attrs.pop('chan_freq')[:len(xds.chan)]
        pol = attrs.pop('corr_type')[:len(xds.pol)]
        
        # truncate per-chan values to the actual number of channels
        attrs['chan_width'] = attrs['chan_width'][:len(xds.chan)]
        attrs['effective_bw'] = attrs['effective_bw'][:len(xds.chan)]
        attrs['resolution'] = attrs['resolution'][:len(xds.chan)]
        
        coords = {'chan': chan, 'pol': pol, 'spw_id': [ddi_xds['SPECTRAL_WINDOW_ID'].values[ddi]],
                  'pol_id': [ddi_xds['POLARIZATION_ID'].values[ddi]]}
        aux_xds = xarray.Dataset(coords=coords, attrs=attrs)

        if not nofile:
            aux_xds.to_zarr(os.path.join(outfile, 'part'+str(ddi)), mode='a', compute=True, consolidated=True)
            xds = xarray.open_zarr(os.path.join(outfile,'part'+str(ddi)))
        
        xds_list += [('part'+str(ddi), xds)]
        print('Completed ddi %i  process time %s s' % (ddi, str(time.time()-start_ddi)))

    # convert other subtables to their own partitions, denoted by 'global_' prefix
    skip_tables = ['DATA_DESCRIPTION', 'SORTED_TABLE']
    subtables = sorted([tt for tt in os.listdir(infile) if os.path.isdir(os.path.join(infile, tt)) and tt not in skip_tables])
    if 'global' in ddis:
        for ii, subtable in enumerate(subtables):
            print('processing subtable %i of %i : %s' % (ii, len(subtables), subtable)) #, end='\r')
            xds_list += [(subtable, tblconv.convert_simple_table(os.path.join(infile, subtable),
                                                                 os.path.join(outfile, 'global_'+subtable),
                                                                 timecols=['TIME'], ignore=ignorecols, compressor=compressor, nofile=nofile))]

    # build the master xds to return
    mxds = xdsio.vis_xds_packager(xds_list)
    print(' '*50)
    
    return mxds
