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

from numba import jit
import numba

def self_cal(vis_mxds, solve_parms, sel_parms):
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

    print('######################### Start self_cal #########################')
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
    from ._calibration_utils._check_calibration_parms import _check_self_cal
    from cngi._utils._check_parms import _check_sel_parms, _check_existence_sel_parms

    _mxds = vis_mxds.copy(deep=True)
    _sel_parms = copy.deepcopy(sel_parms)
    _solve_parms = copy.deepcopy(solve_parms)
    
    assert('xds' in _sel_parms), "######### ERROR: xds must be specified in sel_parms" #Can't have a default since xds names are not fixed.
    _vis_xds = _mxds.attrs[_sel_parms['xds']]
    
    assert(_check_self_cal(_solve_parms)), "######### ERROR: solve_parms checking failed"

    _check_sel_parms(_vis_xds,_sel_parms,new_or_modified_data_variables={'corrected_data':'CORRECTED_DATA','corrected_data_weight':'CORRECTED_DATA_WEIGHT','corrected_flag':'CORRECTED_FLAG','flag_info':'FLAG_INFO'})
    
    # data_groups with model_data is not handled correctly by converter.
    _sel_parms["data_group_in"]["model_data"] = "MODEL_DATA"
    _sel_parms["data_group_out"]["model_data"] = "MODEL_DATA"

    chunk_sizes = list(_vis_xds[_sel_parms["data_group_in"]["data"]].chunks)
    chunk_sizes[1] = (np.sum(chunk_sizes[1]),)
    chunk_sizes[2] = (np.sum(chunk_sizes[2]),)
    chunk_sizes[3] = (np.sum(chunk_sizes[3]),)
    n_pol = _vis_xds.dims['pol']
 
    #assert n_chunks_in_each_dim[3] == 1, "Chunking is not allowed on pol dim."
    n_chunks_in_each_dim = list(_vis_xds[_sel_parms["data_group_in"]["data"]].data.numblocks)
    n_chunks_in_each_dim[1] = 1 #baseline
    n_chunks_in_each_dim[2] = 1 #chan
    n_chunks_in_each_dim[3] = 1 #pol

    #Iter over time,baseline,chan
    iter_chunks_indx = itertools.product(np.arange(n_chunks_in_each_dim[0]), np.arange(n_chunks_in_each_dim[1]),
                                         np.arange(n_chunks_in_each_dim[2]), np.arange(n_chunks_in_each_dim[3]))
                                         

    vis_corrected_list = _ndim_list(n_chunks_in_each_dim)
    weight_corrected_list = _ndim_list(n_chunks_in_each_dim)
    flag_corrected_list = _ndim_list(n_chunks_in_each_dim)
    finfo_list = _ndim_list(n_chunks_in_each_dim)
  
    # Build graph
    for c_time, c_baseline, c_chan, c_pol in iter_chunks_indx:
        #c_time, c_baseline, c_chan, c_pol
        cal_solution_chunk = dask.delayed(_gain_selfcal_chunk)(
            _vis_xds[_sel_parms["data_group_in"]["data"]].data.partitions[c_time, :, :, :],
            _vis_xds[_sel_parms["data_group_in"]["model_data"]].data.partitions[c_time, :, :, :],
            _vis_xds[_sel_parms["data_group_in"]["weight"]].data.partitions[c_time, :, :, :],
            _vis_xds[_sel_parms["data_group_in"]["flag"]].data.partitions[c_time,  :, :, :],
            _vis_xds["ANTENNA1"].data.partitions[:],
            _vis_xds["ANTENNA2"].data.partitions[:],
            dask.delayed(_solve_parms))
            
        
        #print(cal_solution_chunk)
        vis_corrected_list[c_time][c_baseline][c_chan][c_pol] = da.from_delayed(cal_solution_chunk[0],(chunk_sizes[0][c_time],chunk_sizes[1][c_baseline],chunk_sizes[2][c_chan],chunk_sizes[3][c_pol]),dtype=np.complex)
        
        weight_corrected_list[c_time][c_baseline][c_chan][c_pol] = da.from_delayed(cal_solution_chunk[1],(chunk_sizes[0][c_time],chunk_sizes[1][c_baseline],chunk_sizes[2][c_chan],chunk_sizes[3][c_pol]),dtype=np.complex)
        
        flag_corrected_list[c_time][c_baseline][c_chan][c_pol] = da.from_delayed(cal_solution_chunk[2],(chunk_sizes[0][c_time],chunk_sizes[1][c_baseline],chunk_sizes[2][c_chan],chunk_sizes[3][c_pol]),dtype=np.complex)
        
        finfo_list[c_time][c_baseline][c_chan][c_pol] = da.from_delayed(cal_solution_chunk[3],(3,),dtype=np.complex)
        
    _vis_xds[_sel_parms['data_group_out']['corrected_data']] = xr.DataArray(da.block(vis_corrected_list),dims=_vis_xds[_sel_parms['data_group_in']['data']].dims).chunk(_vis_xds[_sel_parms["data_group_in"]["data"]].chunks)
    _vis_xds[_sel_parms['data_group_out']['corrected_data_weight']] = xr.DataArray(da.block(weight_corrected_list),dims=_vis_xds[_sel_parms['data_group_in']['data']].dims).chunk(_vis_xds[_sel_parms["data_group_in"]["data"]].chunks)
    _vis_xds[_sel_parms['data_group_out']['corrected_flag']] = xr.DataArray(da.block(flag_corrected_list),dims=_vis_xds[_sel_parms['data_group_in']['weight']].dims).chunk(_vis_xds[_sel_parms["data_group_in"]["data"]].chunks)
    
    #Will this be added to the data group model?
    _vis_xds[_sel_parms['data_group_out']['flag_info']] = xr.DataArray(np.sum(da.block(finfo_list),axis=(0,1,2)),dims={'flag_info':3})
    #_vis_xds.attrs['flag_info'] = xr.DataArray(np.sum(da.block(finfo_list),axis=(0,1,2)),dims={'flag_info':3})

    
    #Update data_group
    _vis_xds.attrs['data_groups'][0] = {**_vis_xds.attrs['data_groups'][0], **{_sel_parms['data_group_out']['id']:_sel_parms['data_group_out']}}
    
    print('######################### Created self_cal graph #########################')
    return _mxds
    

#def _solve_calibration_chunk(cal_data, model_data, weight, solve_parms):
#    print('cal_data chunk shape is', cal_data.shape)
#    print('model_data chunk shape is', model_data.shape)
#    print('weight chunk shape is', weight.shape)
#    return cal_data
    
    
def _ndim_list(shape):
    return [_ndim_list(shape[1:]) if len(shape) > 1 else None for _ in range(shape[0])]



# v1.0  (gmoellen, 2021Mar31)  Initial shared version.
# v1.1  (gmoellen, 2021Apr01) Excluded ACs by flagging, also avoid them in first guess
#                   Added some timing and solution counting hooks

import numpy as np
from scipy import optimize
import time

def _gain_selfcal_chunk(cal_data,model_data,weight,flag,antenna1,antenna2,solve_parms):
    """
    Solves for antenna-based complex gains, then corrects and returns the supplied visibilities.

    Inputs
    ------
    cal_data : complex visibility data, array of complex, shape=[Ntime,Nbl,Nchan,Ncorr]
    model_data : complex model visibility data, array of complex, shape=[Ntime,Nbl,Nchan,Ncorr]
    weight : weights corresponding to cal_data, array of float, shape=[Ntime,Nbl,Nchan,Ncorr]
    weight : flag corresponding to cal_data, array of float, shape=[Ntime,Nbl,Nchan,Ncorr]
    antenna1: antenna Ids for first antenna in baselines shape=[Nbl]
    antenna2: antenna Ids for second antenna in baselines shape=[Nbl]
    solve_parms : dictionary of solve parameters
    ginfo : if an empty dictionary is specified, it will be filled with various info about the solve (if key='glist' exists, save solution iterestions)

    Returns
    -------
    (vis_corrected,
     weight_corrected,
     flag_corrected,
     inflag,
     outflag,
     nsample) : tuple containing corrected vis, weights, and flags arrays, and input flag count, output flag count, and total sample count

    Notes
    -----

    This function does a rudimentary gain calibration of visibilities.....
    More info TBD

    See _make_selfcal_params above for solve_parms options


    TBD
    ---
    o Report solve statistics
    o Support all solints
    o Handle time axis (solution timestamps, in general)
    o Handle missing/flagged refant better (in guess, refant apply, etc.)
    o Return gain info rather than (or in addition to) corrected vis/weight

    """
    
    #print(cal_data.shape)
    
    if solve_parms['ginfo']:
        ginfo={}
        ginfo['glist'] = []
    else:
        ginfo=None

    t0=time.time()

    # make copies of input data, model, weights to work with
    vis=cal_data.copy()
    mod=model_data.copy()
    wt=weight.copy()

    # Flag count on input
    inflag=np.sum(flag[np.isfinite(flag)])
    nsample=np.size(flag)  # total number of complex vis samples

    # zero data and weights at flags and nans and acs
    #  (Q: Can I assume that nans are same in vis, weight, mod, flag?  Here, I do....)
    acrows=np.array(list(antenna1==antenna2)*vis.shape[0]).reshape(vis.shape[0],vis.shape[1],1,1)
    flag_or_nan_or_ac=np.logical_or(np.logical_or(flag,np.isnan(weight)),acrows)
    vis[flag_or_nan_or_ac]=complex(0.0)
    mod[flag_or_nan_or_ac]=complex(1.0)   # use 1.0, since we divide by this below (only)
    wt[flag_or_nan_or_ac]=0.0

    # Nominal shapes
    (Ntime,Nbaseline,Nchan,Ncorr)=vis.shape

    # Discern net antenna Id list
    solve_ant_ids=np.unique(np.hstack((antenna1,antenna2)))
    Nant=len(solve_ant_ids)

    # Discern antenna indices per baseline
    #  ( Such that Vcorrect=Vobs/Gsol[:,antenna_i,:,:]/np.conj(Gsol[:,antenna_j,:,:]) )
    antenna_i=np.array([np.where(solve_ant_ids==antenna1_id)[0][0] for antenna1_id in antenna1])
    antenna_j=np.array([np.where(solve_ant_ids==antenna2_id)[0][0] for antenna2_id in antenna2])

    # Discern refant index (default to first in solve_ant_ids list)
    refant_i=0
    if solve_parms['refant_id']!=None and solve_parms['refant_id'] in solve_ant_ids:
        refant_i=np.where(solve_ant_ids==solve_parms['refant_id'])[0][0]

    # Slice out parallel-hands only for gain solving
    #  (NB: assumes canonical correlation order)
    if Ncorr==4:
        vis=vis[:,:,:,0::3]
        mod=mod[:,:,:,0::3]
        wt=wt[:,:,:,0::3]
        Ncorr=2

    # Current shape:  [Ntime,Nbaseline,Nchan,Ncorr] (Ncorr<=2)

    # Form vis/mod ratio
    X=vis/mod
    M2=np.real(mod*np.conj(mod))
    Xwt=wt*M2

    # Average over channels
    if Nchan>1:
        # chan axis=2
        #  (X,Xwt)=np.average(X,2,Xwt,True)  # (this can't handle zeros! grrr)
        X=np.sum(X*Xwt,2)
        Xwt=np.sum(Xwt,2)
        non_zero_wt=Xwt>0.0
        X[non_zero_wt]/=Xwt[non_zero_wt]
    else:
        X=X[:,:,0,:]
        Xwt=Xwt[:,:,0,:]

    # Current shape: [Ntime,Nbaseline,Ncorr] (Ncorr<=2)

    # Discern scope of gain polarization solving
    # Nominally ('G'), same as number of correlations
    Npol=Ncorr
    Ncorrperpol=1     # use below in reduced chi2 calculation
    if solve_parms['gaintype']=='T':
        # Doing only one solution
        Npol=1
        if Ncorr==2:
            #  Must average correlations together
            # corr axis=2
            X=0.5*(X[:,:,0]+X[:,:,1])
            dem=Xwt[:,:,0]+Xwt[:,:,1]
            Xwt=4.0*Xwt[:,:,0]*Xwt[:,:,1]
            Xwt[dem>0.0]/=dem[dem>0.0]
            # keep corr axis (required for broadcasting)
            X=X.reshape(list(X.shape)+[1])
            Xwt=Xwt.reshape(list(Xwt.shape)+[1])
            Ncorr=1
            Ncorrperpol=2
    
    # Current shape: [Ntime,Nbaseline,Ncorr] (Ncorr<=2)

    # Handle solint (rudimentary, for now)
    #  If solint='int' do every integration
    #    otherwise, do one solution this chunk
    Nsoltime=Ntime
    if solve_parms['solint']!='int':
        # time axis=0
        #  (X,Xwt)=np.average(X,0,Xwt,True)    # (this can't handle zeros! grrr)
        X=np.sum(X*Xwt,0)
        Xwt=np.sum(Xwt,0)
        non_zero_wt=Xwt>0.0
        X[non_zero_wt]/=Xwt[non_zero_wt]

        # put back time axis (required for broadcasting)
        X=X.reshape([1]+list(X.shape))
        Xwt=Xwt.reshape([1]+list(Xwt.shape))

        # will do only one solution in loop below
        Nsoltime=1


    # Divide by amplitudes for phase-only
    if solve_parms['phase_only']:
        Xamp=np.absolute(X)
        nonzero=Xamp>0.0
        X[nonzero]/=Xamp[nonzero]
        Xwt*=np.square(Xamp)

        
    # Complex gain array into which result will be packed
    # gain shape=[Ntime,Nant,Nchan=1,Npol]   (NB: Npol != Ncorr, in general)
    Gsol=np.ones((Nsoltime,Nant,1,Npol),dtype=complex)
    Gerr=np.zeros((Nsoltime,Nant,1,Npol),dtype=float)

    # Prepare ancillary info gathering
    allglist=None
    glist=None
    if type(ginfo)==dict:
        Gnfev=np.zeros((Nsoltime,Npol),dtype=int)
        Goptim=np.zeros((Nsoltime,Npol),dtype=float)
        Gcost=np.zeros((Nsoltime,Npol),dtype=float)
        Gstat=np.zeros((Nsoltime,Npol),dtype=int)
        GX2r=np.zeros((Nsoltime,Npol),dtype=float)
        if 'glist' in ginfo.keys():
            allglist=[]

    t1=time.time()
    #print('t1',t1-t0)


    # Loop over times and pols, solving for (scalar!) gains in each
    nsolve=0
    nbadsolve=0
    for itime in range(Nsoltime):
        
        # (ancillary info gathering)
        if allglist!=None:
            allglist.append([])
        
        for ipol in range(Npol):

            # (ancillary info gathering)
            if allglist!=None:
                allglist[itime].append([])
                glist=[]   # forces per solve-iteration gain gathering (expensive)

            # data for current time and pol
            thisX=X[itime,:,ipol]
            thisXwt=Xwt[itime,:,ipol]

            # shape=[Nbaseline]

            # Enforce adequate baselines per antenna
            #  NB: this updates thisXwt, if insufficient data found
            badants=_flag_bad_ants(Nant,thisXwt,antenna_i,antenna_j,solve_parms['minblperant'])

            # advance to next ipol,time if this one has no data
            if np.sum(thisXwt)==0.0:
                nbadsolve+=1
                continue

            # Make a first guess based on baselines to refant
            g0=_guess_scalar_gain_from_vis(Nant,thisX,thisXwt,refant_i,antenna_i,antenna_j)

            # Nominal parameter bounds for solve are wide open
            #  (solver works with floats, so x2)
            logain=np.array([-np.inf]*Nant*2)
            higain=np.array([np.inf]*Nant*2)

            # Constrain refant to zero phase == zero imag part
            # TBD: also constrain refant to A=1 for phase-only?  (not clear this matters...)
            logain[2*refant_i+1]=-1e-15
            higain[2*refant_i+1]=1e-15

            # Enforce zero for under-constrained antennas
            # Turning this on makes things SLOW  (more iterations in solve itself)
            # But keeping it off renders blown-up solutions for bad antennas....
            #   Apparently, this doesn't affect good antennas, but not entirely clear....
            # Need further study...
            #
            #for iant in badants:
            #    logain[2*iant]=logain[2*iant+1]=-1e-15
            #    higain[2*iant]=higain[2*iant+1]=1e-15

            # We need sqrt(Xwt) for residual calculator
            thisXsig=np.sqrt(thisXwt)

            # We need float view of g0
            g0_as_float=g0.view(dtype=float)

#            # do the solve!
#            sol=optimize.least_squares(fun=_residual_for_scalar_gain,
#                                       x0=g0_as_float,
#                                       bounds=(logain,higain),
#                                       args=(thisX,thisXsig,antenna_i,antenna_j,glist))
            # do the solve!
            sol=optimize.least_squares(fun=_residual_for_scalar_gain,
                                       x0=g0_as_float,
                                       bounds=(logain,higain),
                                       args=(thisX,thisXsig,antenna_i,antenna_j))

            # Save the complex solved gains
            Gsol[itime,:,0,ipol]=sol.x.view(dtype=complex)

            # Calculate apparent reduced Chi2
            R=sol.fun.view(dtype=complex)
            X2r=sol.cost/np.sum(thisXwt>0.0)

            # (ancillary info gathering)
            if type(ginfo)==dict:
                GX2r[itime,ipol]=X2r
                Gnfev[itime,ipol]=sol.nfev
                Goptim[itime,ipol]=sol.optimality
                Gcost[itime,ipol]=sol.cost
                Gstat[itime,ipol]=sol.status
                if glist!=None:
                    allglist[itime][ipol]=np.array(glist)
                


            # Extract formal errors from Jacobian
            #  (real,imag parts of hessian same, so [::2])
            Jf=sol.jac
            hessian_diag=np.diag(np.dot(np.transpose(Jf),Jf))[::2]
            non_zero_hess=hessian_diag>0.0
            # include X2r correction here
            Gerr[itime,non_zero_hess,0,ipol]=1/np.sqrt(X2r*hessian_diag[non_zero_hess])

            # count full solves
            nsolve+=1

    # Form gain amplitudes for calculation below
    Gamp=np.absolute(Gsol)
            
    # Enforce phase-only
    if solve_parms['phase_only']:
        non_zero_amp=Gamp>0.0
        Gsol[non_zero_amp]/=Gamp[non_zero_amp]
        Gerr[non_zero_amp]/=Gamp[non_zero_amp]

    # Flag bad gains
    Gsnr=np.ones(Gerr.shape,dtype=float)*-1.0
    Gflag=np.zeros(Gerr.shape,dtype=bool)
    non_zero_err=Gerr>0.0
    Gsnr[non_zero_err]=Gamp[non_zero_err]/Gerr[non_zero_err]
    Gflag[Gsnr<solve_parms['minsnr']]=True
    
    # Ensure flagged gains are zero  (ensures corrected weights are zero)
    Gsol[Gflag]=complex(0.0)

    t2=time.time()
    #print('t2',t2-t1)
    
    # Apply the calibration to the input data
    vis_corr,wt_corr,fl_corr=_correct_by_gain(cal_data,weight,flag,Gsol,Gflag,antenna_i,antenna_j)

    # Flag count on output
    outflag=np.sum(flag[np.isfinite(flag)])

    t3=time.time()
    #print('t3',t3-t2)



    # (ancillary info gathering)
    if type(ginfo)==dict:
        ginfo['Gsol']=Gsol
        ginfo['Gerr']=Gerr
        ginfo['Gsnr']=Gsnr
        ginfo['Gflag']=Gflag
        ginfo['sol']={}
        ginfo['sol']['GX2r']=GX2r
        ginfo['sol']['Gcost']=Gcost
        ginfo['sol']['Gnfev']=Gnfev
        ginfo['sol']['Goptim']=Goptim
        ginfo['sol']['Gstat']=Gstat
        ginfo['stat']={}
        ginfo['stat']['inflag']=inflag
        ginfo['stat']['outflag']=outflag
        ginfo['stat']['nsample']=nsample
        ginfo['nsolve']=nsolve
        ginfo['nbadsolve']=nbadsolve
        ginfo['times']={}
        ginfo['times']['prep']=t1-t0
        ginfo['times']['solve']=t2-t1
        ginfo['times']['corr']=t3-t2
        if 'glist' in ginfo.keys():
            ginfo['glist']=allglist
        print(ginfo)
    # Return results
    #return (vis_corr,wt_corr,fl_corr,inflag,outflag,nsample,ginfo)
    finfo = np.array([inflag,outflag,nsample])
    
    return vis_corr,wt_corr,fl_corr,finfo

#@jit(nopython=True, cache=True, nogil=True)
#def _residual_for_scalar_gain(current_gain_float,vis_over_mod,sqrt_weight,antenna_i,antenna_j):
#    """
#    Returns weighted "chi" (residual) for supplied gains and model-normalized visibilities.  Supplied to
#    optimize.least_squares for gain solving.
#
#    Inputs
#    ------
#    current_gain : gain values, as float (from optimize.least_squares), shape=[2*Nant], vector of float
#    vis_over_mod : model-normalized visibilites shape=[Nbl], vector of complex
#    sqrt_weight : sqrt(weight) corresponding to vis_over_mod shape=[Nbl], vector of float
#    antenna_i : antenna indices for first antenna in baselines shape=[Nbl]
#    antenna_j : antenna indices for second antenna in baselines shape=[Nbl]
#
#    Returns
#    -------
#    R : vector of weighted residuals, cast as float (for optimize.least_squares)
#
#    Notes
#    -----
#    R = (V-GiGj*)*sqrtwt, such that chi2=sum(RR*)
#
#    """
#
#    # cast supplied float gain info as complex
#    #current_gain=current_gain_float.view(numba.complex128)
#    current_gain=current_gain_float.view(numba.complex64)
#
#    # This is 'weighted chi' (the square of which is chi-squared), essentially
#    R=(vis_over_mod - current_gain[antenna_i]*np.conj(current_gain[antenna_j]))*sqrt_weight
#
#    # shouldn't be necessary  (via sqrt_weight=0.0)
#    #R[sqrt_weight==0.0]=complex(0.0)
#
#    # ...as float array (optimize.least_squares works only with float info...)
#    #Rf=R.view(numba.double)
#    Rf=R.view(numba.float32)
#
#    return Rf
    

def _residual_for_scalar_gain(current_gain_float,vis_over_mod,sqrt_weight,antenna_i,antenna_j,glist=None):
    """
    Returns weighted "chi" (residual) for supplied gains and model-normalized visibilities.  Supplied to
    optimize.least_squares for gain solving.

    Inputs
    ------
    current_gain : gain values, as float (from optimize.least_squares), shape=[2*Nant], vector of float
    vis_over_mod : model-normalized visibilites shape=[Nbl], vector of complex
    sqrt_weight : sqrt(weight) corresponding to vis_over_mod shape=[Nbl], vector of float
    antenna_i : antenna indices for first antenna in baselines shape=[Nbl]
    antenna_j : antenna indices for second antenna in baselines shape=[Nbl]
    glist : if a list is specified, append current gains to it

    Returns
    -------
    R : vector of weighted residuals, cast as float (for optimize.least_squares)

    Notes
    -----
    R = (V-GiGj*)*sqrtwt, such that chi2=sum(RR*)

    """

    # cast supplied float gain info as complex
    current_gain=current_gain_float.view(dtype=complex)

    if type(glist)==list:
        glist.append(list(current_gain))

    # This is 'weighted chi' (the square of which is chi-squared), essentially
    R=(vis_over_mod - current_gain[antenna_i]*np.conj(current_gain[antenna_j]))*sqrt_weight

    # shouldn't be necessary  (via sqrt_weight=0.0)
    #R[sqrt_weight==0.0]=complex(0.0)

    # ...as float array (optimize.least_squares works only with float info...)
    Rf=R.view(dtype=float)

    return Rf


def _flag_bad_ants(Nant,vis_over_mod_weight,antenna_i,antenna_j,minblperant):
    """
    Recursively flags baselines until all remaining (or none) have specified
    minimum baselines per antenna

    Parameters
    ----------
    Nant : Number of antennas
    vis_over_mod_weight : weights corresponding to vis_over_mod, shape=[Nbl]
    refant_i : antenna index of refant
    antenna_i : antenna indices for first antenna in baselines shape=[Nbl]
    antenna_j : antenna indices for second antenna in baselines shape=[Nbl]
    minblperant : minimum number of baselies per antenna to require (default=4, which is formal minimum for non-degenerate constraints)

    Returns
    -------
    badants : 1d array of bad antenna indices

    Notes
    -----
    Using the baseline-dependent weights, this function determines the number of non-zero-weight baselines available for each antenna, applies the specified threshold, and recursive recalculates the counts until



    """

    # Base data availabililty on non-zero baseline weights
    wtsum=np.sum(vis_over_mod_weight)
    lastwtsum=wtsum+1.0

    # We must loop until baseline attrition stops (weight sum stops decreasing)
    #  (each round can flag additional baselines, which erodes counts)
    while wtsum<lastwtsum:
        # Remember current weight sum
        lastwtsum=wtsum

        # Baselines available per antenna
        Nblperant=np.array([np.sum(vis_over_mod_weight[(antenna_i==iant) | (antenna_j==iant)]>0.0) for iant in range(Nant)])

        # Bad ones are those that fail the required threshold
        badant=np.arange(Nant)[Nblperant<minblperant]

        # Set weights on all baselines to bad antennas to zero
        #  (NB: this is redundant for N>1 passes...)
        for iant in badant:
            vis_over_mod_weight[(antenna_i==iant) | (antenna_j==iant)]=0.0
        wtsum=np.sum(vis_over_mod_weight)

        # If weight sum reaches zero, opt out of unnecessary last iteration
        if wtsum==0.0:
            break

    # return list of bad antennas (for setting solver bounds)
    return badant




def _guess_scalar_gain_from_vis(Nant,vis_over_mod,vis_over_mod_weight,refant_i,antenna_i,antenna_j):
    """
    Makes a guess at scalar complex gain from visibililties on baselines to refant.

    Parameters
    ----------
    Nant : Number of antennas
    vis_over_mod : model-normalized complex visibility vector, shape=[Nbl]
    vis_over_mod_weight : weights corresponding to vis_over_mod, shape=[Nbl]
    refant_i : antenna index of refant
    antenna_i : antenna indices for first antenna in baselines shape=[Nbl]
    antenna_j : antenna indices for second antenna in baselines shape=[Nbl]

    Returns
    -------
    G0 : gain guess shapee=[Nant]

    Notes
    -----
    This function uses the baselines to the specified reference antenna to generate a gain estimate
    suitable for supplying to a gain solver.  The refant's guess has phase=0.0 and amplitude equal
    to the square roo of the amplitudes on the refant baselines.  The guess for the others is the (properly
    conjugated) baseline visibilities, divided by the refant's guess amplitude (ensures ~voltage units).

    Required: Nbl=Nant(Nant-1)/2

    TBD
    ---
    o Handle cases where some or all refant baselines are flagged
    o Use visibility weights in amplitude average

    """


    # Make a more substantitive guess from baselines to refant
    
    # TBD:  Handle flagged baselines to refant below!

    # Nominal guess is all ones
    g0=np.ones(Nant,dtype=complex)
    gwt=np.zeros(Nant,dtype=float)

    # masks to select baselines to refant
    not_acs=antenna_i!=antenna_j
    mask_j=np.logical_and(antenna_j==refant_i,not_acs)   # selects baselines i-refant
    mask_i=np.logical_and(antenna_i==refant_i,not_acs)   # selects baselines refant-j
    mask_ij=np.logical_or(mask_i,mask_j)                 # all refant baselines

    ants_i_to_refant=antenna_i[mask_j]
    ants_j_to_refant=antenna_j[mask_i]
    ants_to_refant=np.hstack((ants_i_to_refant,ants_j_to_refant))  # all ants except refant

    # NB: In vis (power) units to start
    g0[ants_i_to_refant]=vis_over_mod[mask_j]
    g0[ants_j_to_refant]=np.conj(vis_over_mod[mask_i])
    gwt[ants_to_refant]=vis_over_mod_weight[mask_ij]

    if np.sum(gwt) > 0.0:
        # set refant to mean amp of others
        A=np.average(np.absolute(g0),0,gwt)
        g0[refant_i]=complex(A)

        # Divide all by sqrt(A) to convert all to gain (~voltage) units
        g0/=np.sqrt(A)
    else:
        # refant is no good, so just use 1,0
        g0[:]=complex(1.0)
                
    # Return the result
    return g0

def _correct_by_gain(vis,weight,flag,gain,gainflag,antenna_i,antenna_j):
    """
    Corrects supplied visibilities and weights with (antenna-based) multiplicative gain, and returns the result.
    

    Parameters
    ----------
    vis : visibilities to be corrected, with shape=[Ntime,Nbaseline,Nchan,Ncorr]  (complex)
    weight : weights corresponding to vis, same shape (float)
    flag : flags corresponding to vis, same shape (bool)
    gain : antenna-based gains with which to correct the data, with shape=[Ntime,Nant,Nchan=1,Npol]
    gainflag : flags corresponding to gain, same shape
    antenna_i : antenna indices for first antenna in baselines shape=[Nbl]
    antenna_j : antenna indices for second antenna in baselines shape=[Nbl]

    Returns
    -------
    (vis_corrected,weight_corrected,flag_corrected) : tuple containing corrected vis, weights, and flags

    Notes
    -----
    Vcorr = V/Gi/conj(Gj), wtcorr=wt*|Gi|^2|Gj]^2

    Handles single- or dual-pol (single-channel) gain, as well as single-, dual- or four-correlation visibilities
    with any number of channels.  Calibrates the supplied weights also.

    wtcorr=0 where Gi*conj(Gj)=0

    
    TBD:
    o process flags explicitly

    """

    # copy input data for in-place correction (and return)
    vis_corrected=np.zeros(vis.shape,dtype=complex)
    weight_corrected=weight.copy()
    flag_corrected=flag.copy()

    # discern corr/pol shapes
    Ncorr=vis.shape[3]
    Npol=gain.shape[3]

    # organize distribution on correlation axis
    pol_i=[]
    pol_j=[]
    if Npol==1:
        # do scalar correction on corr axis (simple broadcasting)
        pol_i=[0]*Ncorr
        pol_j=[0]*Ncorr
    else:
        # Npol=2, so must carefully distribute on corr axis
        # Ncorr=1 case (shouldn't be able to reach here in this case, though)
        if Ncorr==1:
            # in cal demo, can't reach here;
            #   in future, this should be pol state sensitive
            pol_i=[0]
            pol_j=[0]
        elif Ncorr==2:
            pol_i=[0,1]
            pol_j=[0,1]
        elif Ncorr==4:
            pol_i=[0,0,1,1]
            pol_j=[0,1,0,1]

    # Gi.Gj*, distributed on baseline and corr axes
    GG=gain[:,antenna_i,:,:][:,:,:,pol_i]*np.conj(gain[:,antenna_j,:,:][:,:,:,pol_j])
    AA2=np.real(GG*np.conj(GG))    # The amplitude^2, for weight calibration

    # detect zeros, to avoid divide-by-zero on apply
    #   Note zero amp gains will zero the weights
    zero_amp=AA2==0.0
    GG[zero_amp]=complex(1.0)
    vis_corrected = vis / GG   # broadcasts on time axis
    weight_corrected*=AA2      # calibrate the weights

    # output flags are OR on gain flags
    #  would prefer to only zero visibilities flagged _here_, but there is a broadcasting problem...
    flag_by_gain = gainflag[:,antenna_i,:,:][:,:,:,pol_i] | gainflag[:,antenna_j,:,:][:,:,:,pol_j]
    flag_corrected|=flag_by_gain
    vis_corrected[flag_corrected]=complex(0.0)   # zero flagged visibilities
        
    return vis_corrected,weight_corrected,flag_corrected


def _make_selfcal_parms(gaintype='T',solint='int',refant_id=None,phase_only=False,minsnr=0.0,minblperant=4):
    """
    Return a dictionary containing sensibly defaulted selfcal parameters suitable for supplyint to _gain_selfcal_chunk.

    Inputs
    ------
    gaintype : 'G' or 'T' (default), for dual- or single-pol gain solution, respectively
    solint : 'int' or 'all', for per-time or all-time solution interval
    refant_id : reference antenna Id for phase (phase=0.0 enforced, used for first-guess) If None, will use first antenna.
    phase_only : if True, solve for and correct only the phase (default=False)
    minsnr : Threshold gain SNR for a good solution; effectively flag data not meeting this threshold (default=0.0)
    minblperant : require at least this many unflagged baselines per antenna, per solution (default: 4)
    
    """

    sp={}
    sp['gaintype']=gaintype
    sp['solint']=solint
    sp['refant_id']=refant_id
    sp['phase_only']=phase_only
    sp['minsnr']=minsnr
    sp['minblperant']=minblperant

    return sp
