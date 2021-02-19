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


def convert_ms(infile, outfile=None, ddis=None, ignore=['HISTORY'], compressor=None, chunks=(100, 400, 32, 1), append=False):
    """
    Convert legacy format MS to xarray Visibility Dataset and zarr storage format

    This function requires CASA6 casatools module. The CASA MSv2 format is converted to the MSv3 schema per the
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
    import cngi._utils._table_conversion as tblconv
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
    
    # we need the spectral window, polarization, and data description tables for processing the main table
    spw_xds = tblconv.convert_simple_table(infile, outfile='', subtable='SPECTRAL_WINDOW', ignore=ignorecols, nofile=True)
    pol_xds = tblconv.convert_simple_table(infile, outfile='', subtable='POLARIZATION', ignore=ignorecols, nofile=True)
    ddi_xds = tblconv.convert_simple_table(infile, outfile='', subtable='DATA_DESCRIPTION', ignore=ignorecols, nofile=True)

    # let's assume that each DATA_DESC_ID (ddi) is a fixed shape that may differ from others
    # form a list of ddis to process, each will be placed it in its own xarray dataset and partition
    if ddis is None: ddis = list(ddi_xds['d0'].values) + ['global']
    else: ddis = np.atleast_1d(ddis)
    xds_list = []
    
    ####################################################################
    # process each selected DDI from the input MS, assume a fixed shape within the ddi (should always be true)
    # each DDI is written to its own subdirectory under the parent folder
    for ddi in ddis:
        if ddi == 'global': continue  # handled afterwards
        ddi = int(ddi)
        print('Processing ddi', ddi, end='\r')
        start_ddi = time.time()

        # these columns are different / absent in MSv3 or need to be handled as special cases
        msv2 = ['WEIGHT', 'WEIGHT_SPECTRUM', 'SIGMA', 'SIGMA_SPECTRUM', 'UVW']
        
        # convert columns that are common to MSv2 and MSv3
        xds = tblconv.convert_expanded_table(infile, os.path.join(outfile,'xds'+str(ddi)), keys={'TIME': 'time', ('ANTENNA1', 'ANTENNA2'): 'baseline'},
                                             subsel={'DATA_DESC_ID':ddi}, timecols=['time'], dimnames={'d2':'chan', 'd3':'pol'},
                                             ignore=ignorecols + msv2, compressor=compressor, chunks=chunks, nofile=False)
        
        # convert and append UVW separately so we can handle its special dimension
        uvw_chunks = (chunks[0],chunks[1],3) #No chunking over uvw_index
        uvw_xds = tblconv.convert_expanded_table(infile, os.path.join(outfile,'tmp'), keys={'TIME': 'time', ('ANTENNA1', 'ANTENNA2'): 'baseline'},
                                                 subsel={'DATA_DESC_ID': ddi}, timecols=['time'], dimnames={'d2': 'uvw_index'},
                                                 ignore=ignorecols + list(xds.data_vars) + msv2[:-1], compressor=compressor, chunks=uvw_chunks,
                                                 nofile=False)
        uvw_xds.to_zarr(os.path.join(outfile, 'xds'+str(ddi)), mode='a', compute=True, consolidated=True)
        
        # now convert just the WEIGHT and WEIGHT_SPECTRUM (if preset)
        # WEIGHT needs to be expanded to full dimensionality (time, baseline, chan, pol)
        wt_xds = tblconv.convert_expanded_table(infile, os.path.join(outfile,'tmp'), keys={'TIME': 'time', ('ANTENNA1', 'ANTENNA2'): 'baseline'},
                                                subsel={'DATA_DESC_ID':ddi}, timecols=['time'], dimnames={},
                                                ignore=ignorecols + list(xds.data_vars) + msv2[2:], compressor=compressor, chunks=chunks,
                                                nofile=False)
        
        # if WEIGHT_SPECTRUM is present, append it to the main xds as the new WEIGHT column
        # otherwise expand the dimensionality of WEIGHT and add it to the xds
        if 'WEIGHT_SPECTRUM' in wt_xds.data_vars:
            wt_xds = wt_xds.drop_vars('WEIGHT').rename(dict(zip(wt_xds.WEIGHT_SPECTRUM.dims, ['time','baseline','chan','pol'])))
            wt_xds.to_zarr(os.path.join(outfile, 'xds'+str(ddi)), mode='a', compute=True, consolidated=True)
        else:
            wts = wt_xds.WEIGHT.shape[:2] + (1,) + (wt_xds.WEIGHT.shape[-1],)
            wt_da = da.tile(da.reshape(wt_xds.WEIGHT.data, wts), (1,1,len(xds.chan),1)).rechunk(chunks)
            wt_xds = wt_xds.drop_vars('WEIGHT').assign({'WEIGHT':xarray.DataArray(wt_da, dims=['time','baseline','chan','pol'])})
            wt_xds.to_zarr(os.path.join(outfile, 'xds' + str(ddi)), mode='a', compute=True, consolidated=True)
            
        # add in relevant spw and polarization attributes
        attrs = {}
        
        if ('DATA' in xds.data_vars) and ('CORRECTED_DATA' in xds.data_vars):
            attrs['data_groups'] = [{'1':{'id':'1','data':'DATA','uvw':'UVW','flag':'FLAG','weight':'WEIGHT'}, '2':{'id':'2','data':'CORRECTED_DATA','uvw':'UVW','flag':'FLAG','weight':'WEIGHT'}}]
        elif 'DATA' in xds.data_vars:
            attrs['data_groups'] = [{'1':{'id':'1','data':'DATA','uvw':'UVW','flag':'FLAG','weight':'WEIGHT'}}]
        elif 'CORRECTED_DATA' in xds.data_vars:
            attrs['data_groups'] = [{'1':{'id':'1','data':'CORRECTED_DATA','uvw':'UVW','flag':'FLAG','weight':'WEIGHT'}}]
        else:
            print('The data_groups can not be created because no visibility data was found.')

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
        chan_width = xarray.DataArray(da.from_array(attrs.pop('chan_width')[:len(xds.chan)],chunks=chunks[2]), dims=['chan'])
        effective_bw = xarray.DataArray(da.from_array(attrs.pop('effective_bw')[:len(xds.chan)],chunks=chunks[2]), dims=['chan'])
        resolution = xarray.DataArray(da.from_array(attrs.pop('resolution')[:len(xds.chan)],chunks=chunks[2]), dims=['chan'])
        
        coords = {'chan':chan, 'pol':pol, 'spw_id':[ddi_xds['spectral_window_id'].values[ddi]], 'pol_id':[ddi_xds['polarization_id'].values[ddi]],
                  'chan_width':chan_width, 'effective_bw':effective_bw, 'resolution':resolution}
        aux_xds = xarray.Dataset(coords=coords, attrs=attrs)

        aux_xds.to_zarr(os.path.join(outfile, 'xds'+str(ddi)), mode='a', compute=True, consolidated=True)
        xds = xarray.open_zarr(os.path.join(outfile,'xds'+str(ddi)))
        
        xds_list += [('xds'+str(ddi), xds)]
        print('Completed ddi %i  process time {:0.2f} s'.format(time.time()-start_ddi) % ddi)

    # clean up the tmp directory created by the weight conversion to MSv3
    os.system("rm -fr " + os.path.join(outfile,'tmp'))
    
    # convert other subtables to their own partitions, denoted by 'global_' prefix
    skip_tables = ['DATA_DESCRIPTION', 'SORTED_TABLE'] + ignore
    subtables = sorted([tt for tt in os.listdir(infile) if os.path.isdir(os.path.join(infile, tt)) and tt not in skip_tables])
    if 'global' in ddis:
        start_ddi = time.time()
        for ii, subtable in enumerate(subtables):
            print('processing subtable %i of %i : %s' % (ii, len(subtables), subtable), end='\r')
            if subtable == 'POINTING':    # expand the dimensions of the pointing table
                xds_sub_list = [(subtable, tblconv.convert_expanded_table(infile, os.path.join(outfile, 'global'), subtable=subtable,
                                                                          keys={'TIME': 'time', 'ANTENNA_ID': 'antenna_id'}, timecols=['time'],
                                                                          chunks=chunks))]
            else:
                xds_sub_list = [(subtable, tblconv.convert_simple_table(infile, os.path.join(outfile, 'global'), subtable,
                                                                        timecols=['TIME'], ignore=ignorecols, compressor=compressor, nofile=False))]
            
            if len(xds_sub_list[-1][1].dims) != 0:
                # to conform to MSv3, we need to add explicit ID fields to certain tables
                if subtable in ['ANTENNA','FIELD','OBSERVATION','SCAN','SPECTRAL_WINDOW','STATE']:
                    #if 'd0' in xds_sub_list[-1][1].dims:
                    aux_xds = xarray.Dataset(coords={subtable.lower()+'_id':xarray.DataArray(xds_sub_list[-1][1].d0.values,dims=['d0'])})
                    aux_xds.to_zarr(os.path.join(outfile, 'global/'+subtable), mode='a', compute=True, consolidated=True)
                    xds_sub_list[-1] = (subtable, xarray.open_zarr(os.path.join(outfile, 'global/'+subtable)))
            
                xds_list += xds_sub_list
            #else:
            #    print('Empty Subtable:',subtable)
            
        print('Completed subtables  process time {:0.2f} s'.format(time.time() - start_ddi))
    
    # write sw version that did this conversion to zarr directory
    with open(outfile+'/.version', 'w') as fid:
        fid.write('cngi-protoype ' + importlib_metadata.version('cngi-prototype') + '\n')
    
    # build the master xds to return
    mxds = xdsio.vis_xds_packager(xds_list)
    print(' '*50)
    
    return mxds
