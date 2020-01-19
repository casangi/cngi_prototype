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


def ms_to_zarr(infile, outfile=None, ddi=None, compressor=None, chunk_size_mb=None, chan_chunks=None):
    """
    Convert legacy format MS to xarray Visibility Dataset compatible zarr format

    This function requires CASA6 casatools module.

    Parameters
    ----------
    infile : str
        Input MS filename
    outfile : str
        Output zarr filename. If None, will use infile name with .vis.zarr extension
    ddi : int
        Specific ddi to convert. Leave as None to convert entire MS
    compressor : numcodecs.blosc.Blosc
        The blosc compressor to use when saving the converted data to disk using zarr.
        If None the zstd compression algorithm used with compression level 2.
    chunk_size_mb: float
        Zarr chunk size in megabytes.
        If None a chunk size of 512 MB is used.
    chan_chunks: bool
        Chunk along channel and then if necessary along time.
    Returns
    -------
    """
    import os
    from casatools import table as tb
    from casatools import measures
    from numcodecs import Blosc
    import pandas as pd
    import xarray
    import numpy as np
    import time
    import datetime
    from itertools import cycle
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    #Set default parameters 
    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
    if chunk_size_mb is None:
        chunk_size_mb = 512.0
    if chan_chunks is None:
        chan_chunks = False
    
    # parse filename to use
    infile = os.path.expanduser(infile)
    prefix = infile[:infile.rindex('.')]
    if outfile is None:
        outfile = prefix + '.vis.zarr'
        if chan_chunks:
            outfile = prefix + '_chan_chunks.vis.zarr'
    else:
        outfile = os.path.expanduser(outfile)
        
        
    
    # need to manually remove existing parquet file (if any)
    os.system("rm -fr " + outfile)
    os.system("mkdir " + outfile)
    
    MS = tb(infile)
    MS.open(infile, nomodify=True, lockoptions={'option': 'usernoread'})
    
    # let's assume that each DATA_DESC_ID is a fixed shape that may differ from others
    # process each DATA_DESC_ID and place it in its own partition
    ddis = MS.taql('select distinct DATA_DESC_ID from %s' % prefix + '.ms').getcol('DATA_DESC_ID')
    
    MS.close()
    
    ####################################################################
    # process a DDI from the input MS, assume a fixed shape within the ddi (should always be true)
    # each DDI is written to its own subdirectory under the parent folder
    # consequently, different DDI's may be processed in parallel if the MS is opened with no locks
    def processDDI(ddi, infile, outfile, compressor, chunk_size_mb, chan_chunks):
        print('**********************************')
        print('Processing ddi', ddi)
        start_ddi = time.time()
    
        # Open measurement set (ms) select ddi and sort main table by TIME,ANTENNA1,ANTENNA2
        os.system("mkdir " + outfile + '/' + str(ddi))
        tb_tool = tb()
        tb_tool.open(infile, nomodify=True, lockoptions={'option': 'usernoread'})  # allow concurrent reads
        ms_ddi = tb_tool.taql('select * from %s where DATA_DESC_ID = %s ORDERBY TIME,ANTENNA1,ANTENNA2' % (infile, str(ddi)))
        print('Selecting and sorting time ', time.time() - start_ddi)
        start_ddi = time.time()

        col_names = tb_tool.colnames()  # Array with collumns names
        
        # we want times in datetime format, there must be a better way to do this
        me = measures()
        ms_today = me.epoch('iat','today')['m0']['value'] * 24 * 60 * 60
        real_today = datetime.datetime.now().timestamp()
        correction = ms_today - real_today
        times = pd.to_datetime(np.array(ms_ddi.getcol('TIME')) - correction, unit='s')
        unique_times, time_changes, time_idxs = np.unique(times, return_index=True, return_inverse=True)
        n_time = unique_times.shape[0]
        
        ant1_col = np.array(ms_ddi.getcol('ANTENNA1'))
        ant2_col = np.array(ms_ddi.getcol('ANTENNA2'))
        ant1_ant2 = np.hstack((ant1_col[:, np.newaxis], ant2_col[:, np.newaxis]))
        unique_baselines, baseline_idxs = np.unique(ant1_ant2, axis=0, return_inverse=True)
        n_baseline = unique_baselines.shape[0]

        ##############################################
        # start a dictionary of auxiliary coordinates
        aux_coords = {'interval':('time', ms_ddi.getcol('INTERVAL')[time_changes]),
                  'field_id':('time', ms_ddi.getcol('FIELD_ID')[time_changes]),
                  'scan_number':('time', ms_ddi.getcol('SCAN_NUMBER')[time_changes]),
                  'antennas':(['baseline', 'pair'], unique_baselines)}

        #############################################
        # look up spw and pol ids as starting point
        tb_tool_meta = tb()
        tb_tool_meta.open(infile + "/DATA_DESCRIPTION", nomodify=True, lockoptions={'option': 'usernoread'})
        spectral_window_id = tb_tool_meta.getcol("SPECTRAL_WINDOW_ID")[ddi]
        polarization_id = tb_tool_meta.getcol("POLARIZATION_ID")[ddi]
        tb_tool_meta.close()

        ##############################################
        # Begin meta data dictionary that will make up attributes section
        meta = {'DDI': ddi, 'SPECTRAL_WINDOW_ID': spectral_window_id, 'POLARIZATION_ID': polarization_id,
                'AUTO_CORRELATIONS': np.any(ant1_col == ant2_col)}

        ##############################################
        # process each meta data table in its own special way
        # there probably isn't any general purpose way to do this
        #

        # build metadata structure from remaining table fields
        # these will be added to the xarray dataset attributes section with a prefix indicating which table it came from
        table_prefixes = {'ANTENNA':'ANT_', 'DOPPLER':'DOP_', 'FEED':'FEED_', 'FIELD':'FIELD_', 'FLAG_CMD':'FCMD_',
                          'FREQ_OFFSET':'FOFF_', 'HISTORY':'HIST_', 'OBSERVATION':'OBS_', 'POINTING':'POINT_',
                          'POLARIZATION':'POL_', 'PROCESSOR':'PROC_', 'SOURCE':'SRC_', 'SPECTRAL_WINDOW':'SPW_',
                          'STATE':'STATE_', 'SYSCAL':'SCAL_', 'WEATHER':'WEATH_'}
        # loop over all tables in MS
        for tt in table_prefixes.keys():
            if os.path.isdir(os.path.join(infile, tt)):
                # process table according to its own special construction
                tb_tool_meta.open(os.path.join(infile, tt), nomodify=True, lockoptions={'option': 'usernoread'})
                if tt in ['FEED', 'FREQ_OFFSET', 'SOURCE', 'SYSCAL']:
                    tb_tool_meta = tb_tool_meta.taql('select * from %s where SPECTRAL_WINDOW_ID = %s' %
                                                (os.path.join(infile, tt), str(spectral_window_id)))
                if tb_tool_meta.nrows() == 0: continue
                for col in tb_tool_meta.colnames():
                    try:  # hard
                        if col in ['SPECTRAL_WINDOW_ID', 'NUM_RECEPTORS', 'NUM_POLY', 'NUM_LINES']: continue  # don't need
                        if col in ['CLI_COMMAND']: continue  # currently broken
                        if col in ['CHAN_FREQ', 'CHAN_WIDTH', 'EFFECTIVE_BW', 'RESOLUTION']:
                            aux_coords[col.lower()] = ('chan', tb_tool_meta.getcol(col, spectral_window_id, 1)[:, 0])
                        elif col == 'CORR_TYPE':
                            aux_coords[col.lower()] = ('pol', tb_tool_meta.getcol(col, polarization_id, 1)[:, 0])
                        elif col == 'CORR_PRODUCT':
                            aux_coords[col.lower()] = (['receptor', 'pol'], tb_tool_meta.getcol(col, polarization_id, 1)[:, :, 0])
                        elif col == 'POL_RESPONSE':
                            meta[table_prefixes[tt] + col + '_REAL'] = np.real(tb_tool_meta.getcol(col).transpose())
                            meta[table_prefixes[tt] + col + '_IMAG'] = np.imag(tb_tool_meta.getcol(col).transpose())
                        else:
                            meta[table_prefixes[tt] + col] = tb_tool_meta.getcol(col).transpose()
                    except Exception:  # sometimes bad columns break the table tool (??)
                        print("WARNING: can't process column %s of table %s" % (col, tt))

                tb_tool_meta.close()

        n_chan = len(aux_coords['chan_freq'][1])
        n_pol = len(aux_coords['corr_type'][1])
        
        if chan_chunks:
            chunk_size = np.int(np.ceil((chunk_size_mb * 10 ** 6) / (n_baseline * 1 * n_pol * 16)))
            chan_chunk_size = 1
            if chunk_size > n_time:
                chan_chunk_size = np.int(np.ceil((chunk_size_mb * 10 ** 6) / (n_baseline * n_time * n_pol * 16)))
                chunk_size = n_time
        else:
            chunk_size = min(np.int(np.ceil((chunk_size_mb * 10 ** 6) // (n_baseline * n_chan * n_pol * 16))), n_time)
            print('n_time:', n_time, '  n_baseline:', n_baseline,'  n_chan:', n_chan, '  n_pol:', n_pol, '  number of time steps in chunk:', chunk_size)

        for key in meta.keys():  # zarr doesn't like numpy bools?
            if (type(meta[key]).__module__ == np.__name__) and (meta[key].dtype == 'bool') and (meta[key].ndim == 0):
                meta[key] = bool(meta[key])

        coords = {'time': unique_times, 'baseline': np.arange(n_baseline), 'chan': aux_coords.pop('chan_freq')[1],
                  'pol': aux_coords.pop('corr_type')[1], 'uvw': np.array(['uu', 'vv', 'ww'])}

        print('Metadata processing time ', time.time() - start_ddi)
        start_ddi = time.time()

        ######
        for cc, start_row_indx in enumerate(range(0, n_time, chunk_size)):
            rtestimate = ''
            if cc > 0:
                rtestimate = ', remaining time est %s s' % str(int(((time.time()-start_ddi)/cc)*(n_time/chunk_size-cc)))
            print('processing chunk %s of %s' % (str(cc), str(n_time//chunk_size)) + rtestimate, end='\r')
            chunk = np.arange(min(chunk_size, n_time - start_row_indx)) + start_row_indx
            end_idx = time_changes[chunk[-1]+1] if chunk[-1]+1 < len(time_changes) else len(time_idxs)
            idx_range = np.arange(time_changes[chunk[0]], end_idx)

            dict_x_data_array = {}
            
            for col_name in col_names:
                try:
                    if col_name in ["DATA_DESC_ID","INTERVAL","FIELD_ID","SCAN_NUMBER"]: continue
                    
                    selected_col = ms_ddi.getcol(col_name, idx_range[0], len(idx_range)).transpose()
                    if (selected_col.dtype == 'bool') & (selected_col.ndim == 0): selected_col = bool(selected_col)
                    
                    if col_name in ["UVW"]:  # n_row x 3 -> n_time x n_baseline x 3
                        assert (selected_col.ndim == 2), 'Column dimensions not correct'
                        data = np.full((len(chunk), n_baseline, selected_col.shape[1]), np.nan, dtype=selected_col.dtype)
                        data[time_idxs[idx_range]-chunk[0], baseline_idxs[idx_range],:] = selected_col
                        dict_x_data_array[col_name] = xarray.DataArray(data, dims=['time', 'baseline','uvw'])
                    
                    elif selected_col.ndim == 1:  # n_row -> n_time x n_baseline
                        if col_name == "FLAG_ROW":
                            data = np.ones((len(chunk), n_baseline), dtype=selected_col.dtype)
                        else:
                            data = np.full((len(chunk), n_baseline), np.nan, dtype=selected_col.dtype)
                        data[time_idxs[idx_range]-chunk[0], baseline_idxs[idx_range]] = selected_col
                        dict_x_data_array[col_name] = xarray.DataArray(data, dims=['time', 'baseline'])
                    
                    elif selected_col.ndim == 2:
                        assert (selected_col.shape[1] == n_pol), 'Column dimensions not correct'
                        data = np.full((len(chunk), n_baseline, n_pol), np.nan, dtype=selected_col.dtype)
                        data[time_idxs[idx_range]-chunk[0], baseline_idxs[idx_range],:] = selected_col
                        dict_x_data_array[col_name] = xarray.DataArray(data, dims=['time', 'baseline', 'pol'])
                    
                    elif selected_col.ndim == 3:
                        assert (selected_col.shape[1] == n_chan) & (selected_col.shape[2] == n_pol), 'Column dimensions not correct'
                        if col_name == "FLAG":
                            data = np.ones((len(chunk), n_baseline, n_chan, n_pol), dtype=selected_col.dtype)
                        else:
                            data = np.full((len(chunk), n_baseline, n_chan, n_pol), np.nan, dtype=selected_col.dtype)
                        data[time_idxs[idx_range]-chunk[0], baseline_idxs[idx_range], :, :] = selected_col
                        dict_x_data_array[col_name] = xarray.DataArray(data, dims=['time', 'baseline', 'chan', 'pol'])
                    
                except Exception:  # sometimes bad columns break the table tool (??)
                    print("WARNING: can't process column %s" % col_name)
                    col_names = [_ for _ in col_names if _ != col_name]
    
            coords.update({'time':unique_times[chunk]})
            
            if chan_chunks:
                x_dataset = xarray.Dataset(dict_x_data_array, coords=coords).chunk({'time': chunk_size, 'baseline': None, 'chan': chan_chunk_size, 'pol': None, 'uvw': None})
            else:
                x_dataset = xarray.Dataset(dict_x_data_array, coords=coords).chunk({'time': chunk_size, 'baseline': None, 'chan': None, 'pol': None, 'uvw': None})

            if cc > 0:
                xds = xarray.open_zarr(outfile + '/' + str(ddi))
                x_dataset = xarray.concat([xds, x_dataset], dim='time')
            #if cc == 0:
            encoding = dict(zip(list(x_dataset.data_vars), cycle([{'compressor': compressor}])))
            x_dataset.to_zarr(outfile + '/' + str(ddi), mode='w', encoding=encoding)
            #else:
            #    x_dataset.to_zarr(outfile + '/' + str(ddi), mode='a', append_dim='time')
       
        # Add non dimensional auxiliary coordinates and attributes
        aux_coords.update({'time': unique_times})
        x_dataset = xarray.Dataset(coords=aux_coords, attrs=meta)
        x_dataset.to_zarr(outfile + '/' + str(ddi), mode='a')
        
        ms_ddi.close()
        
        x_dataset = xarray.open_zarr(outfile + '/' + str(ddi))
        print(x_dataset)
        
        print('Completed ddi', ddi, ' process time ', time.time() - start_ddi)
        print('**********************************')

    #########################################
    
    # Parallelize with direct interface
    client = None  # GetFrameworkClient()
    if ddi is not None:
        processDDI(ddi, infile, outfile, compressor, chunk_size_mb, chan_chunks)
    elif client is None:
        for ddi in ddis:
            processDDI(ddi, infile, outfile, compressor, chunk_size_mb, chan_chunks)
    else:
        jobs = client.map(processDDI, ddis,
                          np.repeat(infile, len(ddis)),
                          np.repeat(outfile, len(ddis)),
                          np.repeat(compressor, len(ddis)),
                          np.repeat(chunk_size_mb, len(ddis)), 
                          np.repeat(chan_chunks, len(ddis)))

        # block until complete
        for job in jobs: job.result()
        print('Complete.')
