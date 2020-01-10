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


##################################################################################
# A collection of functions used to convert a measurement file to a xarray dataset
# and saving it to disk using zarr.
###################################################################################
import os
import numpy as np
from numba import jit
from itertools import cycle
import time
#print("numpy version: %s" % (np.__version__))


####################################################################
# Data must be numpy array
@jit(nopython=True)
def _no_repeat_array(data):
    step_indx = np.where(data[:-1] != data[1:])[0] + 1
    step_indx_with_0 = np.zeros(len(step_indx) + 1, dtype=np.int64)
    step_indx_with_0[1:] = step_indx
    no_repeat_data = data[step_indx_with_0]
    return no_repeat_data, step_indx_with_0


####################################################################
# only the top diagional is populated. Therefore ant1 <= ant2
def _create_baseline_mat(ant_id_list, auto_corr_flag):
    n_ant = len(ant_id_list)
    max_indx = np.max(ant_id_list)

    baseline_mat = np.zeros((max_indx + 1, max_indx + 1), dtype=np.int32)

    if auto_corr_flag == True:
        n_baseline = int((n_ant ** 2 + n_ant) / 2)
        baseline_indx = 0
        for i, ant1_indx in enumerate(ant_id_list):
            for ant2_indx in ant_id_list[i:]:
                # print(ant1_indx,ant2_indx)
                baseline_mat[ant1_indx, ant2_indx] = baseline_indx
                baseline_indx = baseline_indx + 1
    else:
        n_baseline = int((n_ant ** 2 - n_ant) / 2)
        baseline_indx = 0
        for i, ant1_indx in enumerate(ant_id_list[:-1]):
            for ant2_indx in ant_id_list[i + 1:]:
                # print(ant1_indx,ant2_indx)
                baseline_mat[ant1_indx, ant2_indx] = baseline_indx
                baseline_indx = baseline_indx + 1

    assert baseline_indx == n_baseline, "Baseline matrix has incorrect number of entries."
    return baseline_mat


####################################################################

def _time_baseline_info(ms_ddi, n_row):
    time_col = np.array(ms_ddi.getcol('TIME'))
    time_step, time_step_indx = _no_repeat_array(time_col)
    n_time = len(time_step_indx)

    ####Checking Code####
    time_indx = 0
    col_time_indx = np.zeros(n_row, dtype=np.int32)
    for i in range(n_row):
        col_time_indx[i] = time_indx
        if i < n_row - 1:
            if time_col[i] != time_col[i + 1]:
                time_indx = time_indx + 1
    assert n_time == col_time_indx[-1] + 1, "Number of time steps are incorrect."
    ####Checking Code####

    # Baseline Stuff
    n_baseline_per_timestep = np.diff(time_step_indx)
    n_baseline_per_timestep = np.append(n_baseline_per_timestep,
                                        n_row - time_step_indx[-1])  # Add last timestep number of baselines

    # Check if any autocorrelations are present
    ant1_col = np.array(ms_ddi.getcol('ANTENNA1'))
    ant2_col = np.array(ms_ddi.getcol('ANTENNA2'))
    auto_corr_flag = False
    if np.sum(ant1_col == ant2_col):
        auto_corr_flag = True

    ant1_id_list = np.unique(ant1_col)
    ant2_id_list = np.unique(ant2_col)
    ant_id_list = sorted(list(set(ant1_id_list).union(set(ant2_id_list))))
    n_ant = len(ant_id_list)
    baseline_mat = _create_baseline_mat(ant_id_list, auto_corr_flag)

    if auto_corr_flag == True:
        n_baseline = int((n_ant ** 2 + n_ant) / 2)
    else:
        n_baseline = int((n_ant ** 2 - n_ant) / 2)

    baseline_ant_pairs = np.zeros((n_baseline, 2), dtype=np.int)
    k = 0
    if auto_corr_flag == True:
        for i in range(n_ant):
            for j in range(i, n_ant):
                baseline_ant_pairs[k, :] = [i, j]
                k = k + 1
    else:
        for i in range(n_ant - 1):
            for j in range(i + 1, n_ant):
                baseline_ant_pairs[k, :] = [i, j]
                k = k + 1
    assert k == n_baseline, "Number of baselines incorrect"

    return n_time, n_ant, n_baseline, time_step, time_step_indx, n_baseline_per_timestep, auto_corr_flag, baseline_mat, baseline_ant_pairs, ant1_col, ant2_col


####################################################################
def _freq_pol_info(ms_ddi, infile, ddi, col_names):
    from casatools import table as tb

    tb_tool_meta = tb()
    tb_tool_meta.open(infile + "/DATA_DESCRIPTION", nomodify=True, lockoptions={'option': 'usernoread'})
    spectral_window_id = tb_tool_meta.getcol("SPECTRAL_WINDOW_ID")
    polarization_id = tb_tool_meta.getcol("POLARIZATION_ID")
    tb_tool_meta.close()

    tb_tool_meta.open(infile + "/SPECTRAL_WINDOW", nomodify=True, lockoptions={'option': 'usernoread'})
    ref_frequency = tb_tool_meta.getcol("REF_FREQUENCY", spectral_window_id[ddi], 1)
    chan_freq = tb_tool_meta.getcol("CHAN_FREQ", spectral_window_id[ddi], 1)
    chan_width = tb_tool_meta.getcol("CHAN_WIDTH", spectral_window_id[ddi], 1)
    effective_bw = tb_tool_meta.getcol("EFFECTIVE_BW", spectral_window_id[ddi], 1)
    resolution = tb_tool_meta.getcol("RESOLUTION", spectral_window_id[ddi], 1)
    total_bandwidth = tb_tool_meta.getcol("TOTAL_BANDWIDTH", spectral_window_id[ddi], 1)[0]
    n_chan = tb_tool_meta.getcol("NUM_CHAN", spectral_window_id[ddi], 1)[0]
    tb_tool_meta.close()

    tb_tool_meta.open(infile + "/POLARIZATION", nomodify=True, lockoptions={'option': 'usernoread'})
    corr_type = tb_tool_meta.getcol("CORR_TYPE", polarization_id[ddi], 1)
    corr_product = tb_tool_meta.getcol("CORR_PRODUCT", polarization_id[ddi], 1)
    n_pol = tb_tool_meta.getcol("NUM_CORR", polarization_id[ddi], 1)[0]
    tb_tool_meta.close()

    # Get n_chan and n_pol
    if "DATA" in col_names:
        selected_col = ms_ddi.getcol("DATA", 0, 1).transpose()
        n_chan_check = selected_col.shape[1]
        n_pol_check = selected_col.shape[2]
    elif "CORRECTED_DATA" in col_names:
        selected_col = ms_ddi.getcol("CORRECTED_DATA", 0, 1).transpose()
        n_chan_check = selected_col.shape[1]
        n_pol_check = selected_col.shape[2]
    elif "MODEL_DATA" in col_names:
        selected_col = ms_ddi.getcol("MODEL_DATA", 0, 1).transpose()
        n_chan_check = selected_col.shape[1]
        n_pol_check = selected_col.shape[2]
    else:
        assert False, "Could not determine number of channels and polarizations."

    assert n_chan_check == n_chan, "Number of channels not consistant"
    assert n_pol_check == n_pol, "Number of polarizations not consistant"

    chan_freq = chan_freq[:, 0]  # Convert from n_chan x 1 -> n_chan
    return n_chan, n_pol, ref_frequency, chan_freq


@jit(nopython=True, cache=True)
def _decompose_row_4d(data, n_row, chunk_time_col_indx, chunk_col_baseline_indx, selected_col):
    for row in range(n_row):
        time_indx = chunk_time_col_indx[row]
        baseline_indx = chunk_col_baseline_indx[row]
        data[time_indx, baseline_indx, :, :] = selected_col[row, :, :]


@jit(nopython=True, cache=True)
def _decompose_row_3d(data, n_row, chunk_time_col_indx, chunk_col_baseline_indx, selected_col):
    for row in range(n_row):
        time_indx = chunk_time_col_indx[row]
        baseline_indx = chunk_col_baseline_indx[row]
        data[time_indx, baseline_indx, :] = selected_col[row, :]


@jit(nopython=True, cache=True)
def _decompose_row_2d(data, n_row, chunk_time_col_indx, chunk_col_baseline_indx, selected_col):
    for row in range(n_row):
        time_indx = chunk_time_col_indx[row]
        baseline_indx = chunk_col_baseline_indx[row]
        data[time_indx, baseline_indx] = selected_col[row]


####################################################################
def convert_ms_data_ndim(col_name, selected_col, chunksize, n_baseline, chunk_time_col_indx, chunk_col_baseline_indx,
                         n_chan, n_pol):
    dim = selected_col.shape
    n_row = dim[0]
    skip = False

    if col_name in ["UVW"]:  # n_row x 3 -> n_time x n_baseline x 3
        assert (dim[1] == 3), 'Collumn dimentions not correct'
        data = np.zeros((chunksize, n_baseline, dim[1]), dtype=selected_col.dtype)
        data.fill(np.nan)

        _decompose_row_3d(data, n_row, chunk_time_col_indx, chunk_col_baseline_indx, selected_col)
        dim_names = ['time', 'baseline', 'uvw']
    ###########################################################################################
    ###########################################################################################
    ###########################################################################################
    elif col_name in ["TIME_CENTROID", "EXPOSURE", "FLAG_ROW"]:  # n_row -> n_time x n_baseline

        # Flag all missing values
        if col_name == "FLAG_ROW":
            data = np.ones((chunksize, n_baseline), dtype=selected_col.dtype)
        else:
            data = np.zeros((chunksize, n_baseline), dtype=selected_col.dtype)
            data.fill(np.nan)

        _decompose_row_2d(data, n_row, chunk_time_col_indx, chunk_col_baseline_indx, selected_col)
        dim_names = ['time', 'baseline']
    ###########################################################################################
    ###########################################################################################
    ###########################################################################################
    elif col_name in ["SIGMA", "WEIGHT"]:  # n_row x n_pol -> n_time x n_baseline x n_pol
        assert (dim[1] == n_pol), 'Collumn dimentions not correct'
        data = np.zeros((chunksize, n_baseline, dim[1]), dtype=selected_col.dtype)
        data.fill(np.nan)

        _decompose_row_3d(data, n_row, chunk_time_col_indx, chunk_col_baseline_indx, selected_col)
        dim_names = ['time', 'baseline', 'pol']
    ###########################################################################################
    ###########################################################################################
    ###########################################################################################
    elif col_name in ["SIGMA_SPECTRUM", "WEIGHT_SPECTRUM", "FLAG", "DATA", "MODEL_DATA","CORRECTED_DATA"]:  # n_row x n_chan x n_pol -> n_time x n_baseline x n_chan x n_pol
        assert (dim[1] == n_chan) & (dim[2] == n_pol), 'Collumn dimentions not correct'

        # Flag all missing values
        if col_name == "FLAG":
            data = np.ones((chunksize, n_baseline, dim[1], dim[2]), dtype=selected_col.dtype)
        else:
            data = np.zeros((chunksize, n_baseline, dim[1], dim[2]), dtype=selected_col.dtype)
            data.fill(np.nan)

        _decompose_row_4d(data, n_row, chunk_time_col_indx, chunk_col_baseline_indx, selected_col)
        
        dim_names = ['time', 'baseline', 'chan', 'pol']

    ###########################################################################################
    ###########################################################################################
    ###########################################################################################
    else:
        # print("Did not convert col ", col_name)
        skip = True
        data = 0
        dim_names = 0

    return data, dim_names, skip


####################################################################
# process a DDI from the input MS, assume a fixed shape within the ddi (should always be true)
# each DDI is written to its own subdirectory under the parent folder
# consequently, different DDI's may be processed in parallel if the MS is opened with no locks
def _processDDI(ddi, infile, outfile, compressor):
    import xarray as xr
    import dask.array as da
    from casatools import table as tb
    

    print('**********************************')
    print('Processing ddi', ddi)
    start_ddi = time.time()

    # Open measurement set (ms) select ddi and sort main table by FIELD_ID,SCAN_NUMBER,TIME,ANTENNA1,ANTENNA2
    tmp = os.system("mkdir " + outfile + '/' + str(ddi))
    tb_tool = tb()
    tb_tool.open(infile, nomodify=True, lockoptions={'option': 'usernoread'})  # allow concurrent reads
    start = time.time()
    ms_ddi = tb_tool.taql(
        'select * from %s where DATA_DESC_ID = %s ORDERBY FIELD_ID,SCAN_NUMBER,TIME,ANTENNA1,ANTENNA2' % (
        infile, str(ddi)))
    print('Selecting and sorting time ', time.time() - start)

    n_row = ms_ddi.nrows()  # Number of rows
    col_names = tb_tool.colnames()  # Array with collumns names

    # print('Number of rows in ddi', ddi, 'is', n_row)

    # start = time.time()
    n_time, n_ant, n_baseline, time_step, time_step_indx, n_baseline_per_timestep, auto_corr_flag, baseline_mat, baseline_ant_pairs, ant1_col, ant2_col = _time_baseline_info(
        ms_ddi, n_row)
    # print('time_baseline_info time ', time.time() - start)

    # start = time.time()
    n_chan, n_pol, ref_frequency, chan_freq = _freq_pol_info(ms_ddi, infile, ddi, col_names)
    # print('freq_pol_info time ', time.time() - start)

    chunksize = np.int(np.ceil((256 * 10 ** 6) / (n_baseline * n_chan * n_pol * 16)))
    print('n_time', n_time, ',n_baseline', n_baseline, ',n_chan', n_chan, ',n_pol', n_pol, ',chunksize', chunksize)

    time_col = np.array(ms_ddi.getcol('TIME'))

    # Non dimentional coordinates
    interval_step = (ms_ddi.getcol('INTERVAL'))[time_step_indx]
    field_id_step = (ms_ddi.getcol('FIELD_ID'))[time_step_indx]
    scan_number_step = (ms_ddi.getcol('SCAN_NUMBER'))[time_step_indx]

    for cc, start_row_indx in enumerate(range(0, n_time, chunksize)):
        # print(cc,start_row_indx)
        chunk = np.arange(min(chunksize, n_time - start_row_indx)) + start_row_indx
        time_chunk = time_step[chunk]
        current_chunksize = min(chunksize, n_time - start_row_indx)
        # print('current_chunksize is ',current_chunksize)

        n_row_in_chunk = np.sum(n_baseline_per_timestep[start_row_indx:start_row_indx + current_chunksize])
        ant1_chunk = ant1_col[time_step_indx[start_row_indx]:time_step_indx[start_row_indx] + n_row_in_chunk]
        ant2_chunk = ant2_col[time_step_indx[start_row_indx]:time_step_indx[start_row_indx] + n_row_in_chunk]
        chunk_col_baseline_indx = baseline_mat[ant1_chunk, ant2_chunk]
        chunk_time_col = time_col[time_step_indx[start_row_indx]:time_step_indx[start_row_indx] + n_row_in_chunk]

        time_indx = 0
        chunk_time_col_indx = np.zeros(n_row, dtype=np.int32)
        for i in range(n_row_in_chunk):
            chunk_time_col_indx[i] = time_indx
            if i < n_row_in_chunk - 1:
                if chunk_time_col[i] != chunk_time_col[i + 1]:
                    time_indx = time_indx + 1

        coords, dict_x_data_array = {'time': time_chunk, 'baseline': np.arange(n_baseline), 'chan': chan_freq,
                                     'pol': np.arange(n_pol), 'uvw': np.arange(3)}, {}

        '''
        interval_chunk = interval_step[chunk]
        field_id_chunk = field_id_step[chunk]
        scan_number_chunk = scan_number_step[chunk]
        coords['interval'] =  ('time', interval_chunk)
        coords['field_id'] =  ('time', field_id_chunk)
        coords['scan_number'] =  ('time', np.squeeze(scan_number_chunk))
        '''

        for col_name in col_names:
            #try:
            if True:
                if col_name in ["FLAG_CATEGORY", "DATA_DESC_ID"]: continue

                selected_col = ms_ddi.getcol(col_name, time_step_indx[start_row_indx], n_row_in_chunk).transpose()
                if selected_col.dtype == 'bool': selected_col = selected_col.astype(np.int)

                # start = time.time()
                data, dim_names, skip = convert_ms_data_ndim(col_name, selected_col, current_chunksize, n_baseline,
                                                             chunk_time_col_indx, chunk_col_baseline_indx, n_chan,
                                                             n_pol)
                # print('convert time ', col_name, time.time() - start)

                if skip == False:
                    dict_x_data_array[col_name] = xr.DataArray(data, dims=dim_names)
            #except Exception:  # sometimes bad columns break the table tool (??)
            #    print("WARNING: can't process column %s" % (col_name))
            #    col_names = [_ for _ in col_names if _ != col_name]

        # print('#')
        # print({'DATA':dict_x_data_array['DATA']})
        # print(coords)
        # x_dataset = xr.Dataset( dict_x_data_array, coords=coords).chunk({'time':len(coords['time']),'baseline':None,'chan':None,'pol':None,'uvw':None})

        
        x_dataset = xr.Dataset(dict_x_data_array, coords=coords).chunk(
            {'time': len(coords['time']), 'baseline': None, 'chan': None, 'pol': None, 'uvw': None})

        if cc == 0:
            encoding = dict(zip(list(x_dataset.data_vars), cycle([{'compressor': compressor}])))
            x_dataset.to_zarr(outfile + '/' + str(ddi), mode='w', encoding=encoding)
        else:
            x_dataset.to_zarr(outfile + '/' + str(ddi), mode='a', append_dim='time')
        
    # coords, dict_x_data_array = {'baseline':np.arange(n_baseline),'ant_pair':np.arange(2)}, {}
    # dict_x_data_array['baseline_ant_pairs'] = xr.DataArray(baseline_ant_pairs, dims=['baseline','ant_pair'])
    # x_dataset = xr.Dataset(dict_x_data_array, coords=coords)

    # Add non dimentional coordinates
    coords = {}
    coords['time'] = time_step
    coords['interval'] = ('time', da.from_array(interval_step,chunks=(chunksize,)))
    coords['field_id'] = ('time', da.from_array(field_id_step,chunks=(chunksize,)))
    coords['scan_number'] = ('time', da.from_array(scan_number_step,chunks=(chunksize,)))

    x_dataset = xr.Dataset(coords=coords)

    # Add attributes
    # Should this be attributes?
    scan_number_col = ms_ddi.getcol("SCAN_NUMBER").transpose()
    no_repeat_scan_number, scan_number_step_indx = _no_repeat_array(scan_number_col.astype(int)) 
    scan_number_indx = np.array([list(no_repeat_scan_number), list(scan_number_step_indx)], np.int)
    
    field_id_col = ms_ddi.getcol("FIELD_ID").transpose()
    no_repeat_field_id, field_id_step_indx = _no_repeat_array(field_id_col.astype(int))       #(2 x [n_field x 1])
    field_id_indx = np.array([list(no_repeat_field_id), list(field_id_step_indx)],np.int)
    
    n_scan  = len(np.unique(no_repeat_scan_number))
    n_field = len(np.unique(no_repeat_field_id))

    x_dataset.attrs['baseline_ant_pairs'] = baseline_ant_pairs
    x_dataset.attrs['auto_corr_flag'] = auto_corr_flag
    x_dataset.attrs['ref_frequency'] = ref_frequency
    x_dataset.attrs['scan_number_indx'] = scan_number_indx
    x_dataset.attrs['field_id_indx'] = field_id_indx
    x_dataset.attrs['n_scan'] = n_scan
    x_dataset.attrs['n_field'] = n_field
    
    x_dataset.to_zarr(outfile + '/' + str(ddi), mode='a')

    ms_ddi.close()
    print('Completed ddi', ddi, ' process time ', time.time() - start_ddi)
    print('**********************************')



############################################################################
def ms_to_zarr_numba(infile, outfile=None, ddi=None, compressor=None):
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
    compressor : blosc
        The blosc compressor to use when saving the converted data to disk using zarr.
        If None the zstd compression algorithm used with compression level 2.

    Returns
    -------
    """

    from casatools import table as tb
    from numcodecs import Blosc

    if compressor == None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)

    # Parse filename to use
    prefix = infile[:infile.rindex('.')]
    if outfile == None: outfile = prefix + '.vis.zarr'

    # Need to manually remove existing directory (if any)
    tmp = os.system("rm -fr " + outfile)
    tmp = os.system("mkdir " + outfile)

    # Get each DATA_DESC_ID (ddi). Visbility data for each ddi is assumed to have fixed shape.
    tb_tool = tb(infile)
    tb_tool.open(infile, nomodify=True, lockoptions={'option': 'usernoread'})
    ddis = tb_tool.taql('select distinct DATA_DESC_ID from %s' % prefix + '.ms').getcol('DATA_DESC_ID')
    tb_tool.close()

    # Parallelize with direct interface
    client = None  # GetFrameworkClient()
    if ddi != None:
        _processDDI(ddi, infile, outfile, compressor)
    elif client == None:
        for ddi in ddis:
            _processDDI(ddi, infile, outfile, compressor)
    else:
        jobs = client.map(_processDDI, ddis,
                          np.repeat(infile, len(ddis)),
                          np.repeat(outfile, len(ddis)),
                          np.repeat(compressor, len(ddis)))

        # block until complete
        for job in jobs: job.result()
        print('Complete.')

