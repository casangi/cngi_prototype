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
'''
Test Dataset

wget https://bulk.cv.nrao.edu/almadata/public/working/sis14_twhya_calibrated_flagged.ms.tar

mstransform('sis14_twhya_calibrated_flagged.ms',
             outputvis='sis14_twhya_field_5_lsrk.ms',
             regridms=True, outframe='LSRK', datacolumn='corrected', field='5')

tclean(vis='sis14_twhya_field_5_lsrk.ms/',
       imagename='first_image',
       spw='',
       specmode='mfs',
       deconvolver='hogbom',
       nterms=1,
       gridder='standard',
       imsize=[250,250],
       cell=['0.1arcsec'],
       weighting='natural',
       threshold='0mJy',
       niter=5000,
       savemodel='modelcolumn')

from cngi.conversion import convert_ms
vis_mxds = convert_ms('sis14_twhya_field_5_lsrk.ms/',chunks=(27,21,32,2))


'''

def solve_calibration(vis_mxds, cal_xds, solve_parms, sel_parms):
    """
    .. todo::
        This function is not yet implemented
    
    Calculate antenna gain solutions according to the parameters in solpars.
    The input dataset has been pre-averaged/processed and the model visibilities exist
    
    Iteratively solve the system of equations g_i g_j* = V_data_ij/V_model_ij  for all ij.
    Construct a separate solution for each timestep and channel in the input dataset.
    
    Options :
    
    amp, phase or both
    solution type (?) G-term, D-term, etc...
    Data array for which to calculate solutions. Default='DATA'
    
    TBD :
    
    Single method with options for solutions of different types ?
    Or, separate methods for G/B, D, P etc.. : solve_B, solve_D, solve_B, etc...
          
    Returns
    -------
    
    """

    print('######################### Start solve_calibration #########################')
    import numpy as np
    from numba import jit
    import time
    import math
    import dask.array.fft as dafft
    import xarray as xr
    import dask.array as da
    import matplotlib.pylab as plt
    import dask
    import copy, os
    from numcodecs import Blosc
    from itertools import cycle
    import itertools
    
    from cngi._utils._check_parms import _check_sel_parms, _check_existence_sel_parms

    _mxds = vis_mxds.copy(deep=True)
    #_cal_xds = cal_xds.copy(deep=True)
    _sel_parms = copy.deepcopy(sel_parms)
    _solve_parms = copy.deepcopy(solve_parms)
    
    assert('xds' in _sel_parms), "######### ERROR: xds must be specified in sel_parms" #Can't have a default since xds names are not fixed.
    _vis_xds = _mxds.attrs[_sel_parms['xds']]



    n_chunks_in_each_dim = _vis_xds[sel_parms["data_group_in"]["data"]].data.numblocks
    chunk_sizes = _vis_xds[sel_parms["data_group_in"]["data"]].chunks
    n_pol = _vis_xds.dims['pol']
 
    assert n_chunks_in_each_dim[3] == 1, "Chunking is not allowed on pol dim."

    #Iter over time,baseline,chan
    iter_chunks_indx = itertools.product(np.arange(n_chunks_in_each_dim[0]), np.arange(n_chunks_in_each_dim[1]),
                                         np.arange(n_chunks_in_each_dim[2]), np.arange(n_chunks_in_each_dim[3]))
                                         

    cal_solution_list = _ndim_list(n_chunks_in_each_dim)
  
    # Build graph
    for c_time, c_baseline, c_chan, c_pol in iter_chunks_indx:
        print(c_time,c_baseline,c_chan,c_pol)
        cal_solution_chunk = dask.delayed(_solve_calibration_chunk)(
            _vis_xds[sel_parms["data_group_in"]["data"]].data.partitions[c_time, c_baseline, c_chan, c_pol],
            _vis_xds[sel_parms["data_group_in"]["model_data"]].data.partitions[c_time, c_baseline, c_chan, c_pol],
            _vis_xds[sel_parms["data_group_in"]["weight"]].data.partitions[c_time, c_baseline, c_chan, c_pol],
            dask.delayed(_solve_parms))
            
        #print(cal_solution_chunk)
        cal_solution_list[c_time][c_baseline][c_chan][c_pol] = da.from_delayed(cal_solution_chunk,(chunk_sizes[0][c_time],chunk_sizes[1][c_baseline],chunk_sizes[2][c_chan],chunk_sizes[3][c_pol]),dtype=np.complex)
        
    cal_solution = da.block(cal_solution_list)
    
    return cal_solution
    

def _solve_calibration_chunk(cal_data, model_data, weight, solve_parms):
    print('cal_data chunk shape is', cal_data.shape)
    print('model_data chunk shape is', model_data.shape)
    print('weight chunk shape is', weight.shape)
    return cal_data
    
    
def _ndim_list(shape):
    return [_ndim_list(shape[1:]) if len(shape) > 1 else None for _ in range(shape[0])]
