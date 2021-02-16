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

#ducting - code is complex and might fail after some time if parameters is wrong (time waisting). Sensable values are also checked. Gives printout of all wrong parameters. Dirty images alone has x parametrs.
#explore BEAM subtable? (keep it seperate with version numbers?)
#Create function that creates beam subtable from zpc files or functions add to mxds?

import numpy as np
import xarray as xr
import dask.array as da
import time
import matplotlib.pyplot as plt
from numba import jit
import numba

def _calc_parallactic_angles_for_gcf(mxds,ant_ra_dec,gcf_parms,sel_parms):
    #Calculate the
    pa = _calc_parallactic_angles_for_vis_dataset(mxds,ant_ra_dec,gcf_parms,sel_parms)
    
    if gcf_parms['average_pa']:
        pa_subset = _average_parallactic_to_match_map(mxds,pa)
    else:
        pa_subset = pa
    
    start = time.time()
    pa_centers, pa_dif = _find_optimal_set_angle(pa_subset.data.compute(),gcf_parms['pa_step'])
    print('Time to find optimal pa set ', time.time()-start)
    '''
    print('*****')
    print('pa_centers',pa_centers.data.compute()*180/np.pi)
    print('cf_time_map',cf_time_map.data.compute())
    print('pa_map',pa_map.data.compute())
    print('*****')

    _check_pa(cf_time_map, pa_centers, pa_dif, pa_map, pa, beam_pair_id, mxds, gcf_parms)
    
    #return cf_time_map, pa_centers, pa_dif
    return cf_time_map, pa_centers, pa_dif, pa_map
    '''
    vis_dataset = mxds.attrs[sel_parms['xds']]
    time_chunksize = vis_dataset[sel_parms['data']].chunks[0]
    ant_chunksize= pa_dif.shape[1]
    
    pa_centers = xr.DataArray(da.from_array(pa_centers), dims=('pa'))
    pa_dif = xr.DataArray(da.from_array(pa_dif,chunks=(time_chunksize,ant_chunksize)), dims=('time','beam'))
    #pa = xr.DataArray(da.from_array(pa,chunks=(time_chunksize,ant_chunksize)), dims=('time','beam'))
    
    return pa, pa_centers, pa_dif
    
def _check_pa(cf_time_map, pa_centers, pa_dif, pa_map, pa, beam_pair_id, mxds, gcf_parms):
    
    print(beam_pair_id)
    
    pa = pa.data.compute()
    #pa[:,1] = pa[:,1] + 0.1
    n_time = pa.shape[0]
    n_ant = pa.shape[1]
    n_beam_pairs = beam_pair_id.shape[0]
    
    unique_model_id = mxds.ANTENNA['model_id'].data.compute()
    beam_model = mxds.ANTENNA['MODEL'].data.compute()
    
    print(unique_model_id)
    print(beam_pair_id)
    
    print(beam_model)
    cf_time_map = cf_time_map.data.compute()
    pa_map = pa_map.data.compute()
    pa_centers = pa_centers.data.compute()
    
    for ii in range(n_time):
        for jj in range(n_beam_pairs):
            i_beam_0 = np.where(beam_model == beam_pair_id[jj,0])[0][0]
            i_beam_1 = np.where(beam_model == beam_pair_id[jj,1])[0][0]
            pa_0 = pa[ii,i_beam_0]
            pa_1 = pa[ii,i_beam_1]
            
            #print(i_beam_0,i_beam_1)
            #print(pa_0,pa_1)
            
            time_indx = cf_time_map[ii]
            #print('****')
            #print(time_indx,'***',jj,'***', pa_map, '***',pa_centers )
            #print('****')
            pa_cf_0 = pa_centers[pa_map[time_indx,jj,0]]
            pa_cf_1 = pa_centers[pa_map[time_indx,jj,1]]
            
            ang_dif_0 = _ang_dis(pa_0,pa_cf_0)
            ang_dif_1 = _ang_dis(pa_1,pa_cf_1)
            
            
            
            if gcf_parms['pa_step'] < ang_dif_0 or gcf_parms['pa_step'] < ang_dif_1:
                print('Problem')
                print(ii,jj,beam_pair_id[jj,0], beam_pair_id[jj,1],gcf_parms['pa_step'], ang_dif_0*180/np.pi, ang_dif_1*180/np.pi)
                print(pa_cf_0*180/np.pi,pa_cf_1*180/np.pi)
                print(pa_0*180/np.pi,pa_1*180/np.pi)
                print('*****')
            #gcf_parms['pa_step']
                
def _ang_dis(ang1,ang2):
    ang_dif = ang1-ang2
    ang_dif = np.abs((ang_dif + np.pi)%(2*np.pi) - np.pi)
    return ang_dif
    

def _average_parallactic_to_match_map(mxds,pa):
# Average PA for all feeds that have same beam model.
    beam_ids = mxds.beam_ids.data.compute()
    n_beam = len(beam_ids)
    feed_beam_ids = mxds.FEED.beam_id.data.compute()
    
    pa_mean = np.zeros((pa.shape[0],n_beam))
    
    for i,id in enumerate(beam_ids):
        pa_mean[:,i] = np.mean(pa[:,feed_beam_ids==id],axis=1)
        
    time_chunksize = pa.chunks[0]
    beam_chunksize = mxds.beam_ids.chunks[0]

    pa_mean = xr.DataArray(da.from_array(pa_mean,chunks=(time_chunksize,beam_chunksize)),{'time':pa.time,'beam':beam_ids}, dims=('time','beam'))
    return pa_mean

def _calc_parallactic_angles_for_vis_dataset(mxds,ra_dec,gcf_parms,sel_parms):
    from astropy.coordinates import (EarthLocation, SkyCoord,
                                     AltAz, CIRS)
    import astropy.units as u
    from astropy.time import Time
    import copy
    vis_dataset = mxds.attrs[sel_parms['xds']]
    
    antenna_ids = mxds.antenna_ids.data
    n_ant = len(antenna_ids)
    n_time = vis_dataset.dims['time']
    
#    if gcf_parms['use_pointing_table_parallactic_angle']:
#        #if False:
#        ### Using pointing table
#        print('Using Pointing dataset to calculate parallactic angles.')
#        #print(mxds.POINTING.DIRECTION)
#        ra_dec = mxds.POINTING.DIRECTION.interp(time=vis_dataset.time,assume_sorted=False,method=gcf_parms['interpolation_method']).data.compute()[:,:,0,:]
#        #else:
#    else:
#        #### Using field table
#        print('Using Field dataset to calculate parallactic angles.')
#        field_dataset = mxds.attrs['FIELD']
#
#        field_id = np.max(vis_dataset.FIELD_ID,axis=1).compute() #np.max ignores int nan values (nan values are large negative numbers for int).
#        n_field = field_dataset.dims['d0']
#        n_time = vis_dataset.dims['time']
#
#        ra_dec = field_dataset.PHASE_DIR.isel(d0=field_id).data.compute()
#
#        #print(ra_dec.shape)
#        if n_field != 1:
#            ra_dec = ra_dec[:,0,:]
#
#        ra_dec = np.tile(ra_dec[:,None,:],(1,n_ant,1))
    
    ra = ra_dec[:,:,0].data.compute()
    dec = ra_dec[:,:,1].data.compute()
    #print(ra.shape)
    phase_center = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='fk5') #XXXXXXXXXXXXXXXXXX fk5 epoch is J2000, when to switch over to ICRS?


    if gcf_parms['use_array_ref_pos']:
        telescope_name = mxds.attrs['OBSERVATION'].TELESCOPE_NAME.values[0]
        if telescope_name=='EVLA': telescope_name='VLA'
        observing_location = EarthLocation.of_site(telescope_name) #XXXXXXXXXXXXXXXXXX Need to check against values in CASA data repo
        x = np.tile(np.array([observing_location.x.value])[None,:],(n_time,n_ant))
        y = np.tile(np.array([observing_location.x.value])[None,:],(n_time,n_ant))
        z = np.tile(np.array([observing_location.x.value])[None,:],(n_time,n_ant))
    else:
        ant_pos = np.tile(mxds.ANTENNA.POSITION.values[None,:,:],(n_time,1,1))
        x = ant_pos[:,:,0]
        y = ant_pos[:,:,1]
        z = ant_pos[:,:,2]
        
        #x[:,1] = x[:,1] + 100000
        #y[:,1] = y[:,1] + 100000
        #z[:,1] = z[:,1] + 500000
    observing_location = EarthLocation.from_geocentric(x=x*u.m, y=y*u.m,z=z*u.m)
    #print('ol',observing_location.shape)
    
    obs_time = np.tile(vis_dataset.time.values.astype(str)[:,None],(1,n_ant))
    obs_time = Time(obs_time, format='isot', scale='utc', location=observing_location)
    #print('time',obs_time.shape)
    
    start = time.time()
    pa = _calc_parallactic_angles(obs_time, observing_location, phase_center)
    print('Time to calc pa ', time.time()-start)
    
    #print(pa.shape)
    
#    print('pa',pa.shape)
#    plt.figure()
#    plt.plot(pa[0,:].T)
#    plt.show()
    
    time_chunksize = vis_dataset[sel_parms['data']].chunks[0]
    ant_chunksize= mxds.ANTENNA.POSITION.chunks[0]
    
    #print(da.from_array(pa,chunks=(time_chunksize,ant_chunksize)))
    
    ########Remove remove
    #pa[:,1] = pa[:,1] + 0.1
    
    pa = xr.DataArray(da.from_array(pa,chunks=(time_chunksize,ant_chunksize)),{'time':vis_dataset.time,'ant':antenna_ids}, dims=('time','ant'))
    return pa


def _calc_parallactic_angles(times, observing_location, phase_center):
    """
    Computes parallactic angles per timestep for the given
    reference antenna position and field centre.
    
    Based on https://github.com/ska-sa/codex-africanus/blob/068c14fb6cf0c6802117689042de5f55dc49d07c/africanus/rime/parangles_astropy.py
    """
    from astropy.coordinates import (EarthLocation, SkyCoord,
                                     AltAz, CIRS)
    from astropy.time import Time
    from astropy import units
    import numpy as np
    
    pole = SkyCoord(ra=0, dec=90, unit=units.deg, frame='fk5')

    cirs_frame = CIRS(obstime=times)
    pole_cirs = pole.transform_to(cirs_frame)
    phase_center_cirs = phase_center.transform_to(cirs_frame)

    altaz_frame = AltAz(location=observing_location, obstime=times)
    pole_altaz = pole_cirs.transform_to(altaz_frame)
    phase_center_altaz = phase_center_cirs.transform_to(altaz_frame)
    
    #print('the zen angle is',phase_center_altaz.zen)
    #print('the zen angle is',pole_altaz.zen)
        
    return phase_center_altaz.position_angle(pole_altaz).value

def _make_cf_time_map(mxds,pa,beam_pair_id,gcf_parms,sel_parms):

    #Finds all the antennas where the pa
    #start = time.time()
    #pa_centers,cf_time_map,pa_dif = _find_ant_with_large_pa_diff(pa.data.compute(),gcf_parms['ant_pa_step'])
    #print('Time to make cf_time_map ', time.time()-start)
    #pa_centers = np.array(pa_centers)
    
    start = time.time()
    pa_centers, pa_dif = _find_optimal_set_angle(pa.data.compute(),gcf_parms['pa_step'])
    print('Time to find optimal pa set ', time.time()-start)
    
    print(pa_centers*180/np.pi)
    
    print(pa_dif)
    
    #pa_centers = np.array(pa_centers)
    
    
    #pa_time_beam_pairs,pa_pair_map,pa_centers,pa_dif = _find_optimal_set_angle(pa.data.compute(),gcf_parms['pa_step'],beam_pair_id,mxds.ANTENNA['model_id'].data.compute())
    
    '''
    start = time.time()
    pa_time_beam_pairs_unique , ind = np.unique(pa_time_beam_pairs,axis=0, return_index=True)
    pa_time_beam_pairs_unique = pa_time_beam_pairs_unique[np.argsort(ind)]
    #print(pa_time_beam_pairs)
    #print(pa_time_beam_pairs_unique) #NBNBNBNBNBNBNB
    cf_time_map, pa_map = _make_map(pa_time_beam_pairs_unique,pa_time_beam_pairs,pa_pair_map,pa_centers)
    print('Time to make cf_time_map ', time.time()-start)
    '''
    
    '''
    #############################################################
    #CASA way of doing things
    ang_dif_list = np.zeros(len(pa))
    print(ang_dif_list.shape,len(pa))
    pa_cf_count = 1
    cur_pa = pa[0]
    for ii in range(1,len(pa)):
        ang_dif = cur_pa - pa[ii]
        ang_dif = np.abs((ang_dif + np.pi)%(2*np.pi) - np.pi)
        
        if ang_dif > gcf_parms['pa_step']:
            pa_cf_count = pa_cf_count + 1
            cur_pa = pa[ii]
            ang_dif_list[ii] = 0
        else:
            ang_dif_list[ii] = ang_dif
            
    pa_dif = pa_dif*180/np.pi
    ang_dif_list = ang_dif_list*180/np.pi
    
    plt.figure()
    plt.plot(pa_dif,'.',label='ngcasa')
    plt.plot(ang_dif_list,'.',label='casa')
    plt.xlabel('Time Step')
    plt.ylabel('Difference Angle (degrees)')
    plt.title('pa_step 1 degree')
    plt.legend()
    
    plt.figure()
    plt.hist(pa_dif, bins='auto')
    plt.title('ngCASA')
    plt.xlabel('Difference Angle (degrees)')
    
    plt.figure()
    plt.hist(ang_dif_list, bins='auto')
    plt.title('CASA')
    plt.xlabel('Difference Angle (degrees)')
    
    print(np.mean(pa_dif),np.mean(ang_dif_list))
    print(np.median(pa_dif),np.median(ang_dif_list))
    print(np.var(pa_dif),np.var(ang_dif_list))
    print(len(pa_centers),pa_cf_count)
    plt.show()
    #############################################################
    '''
    
    '''
    pa_chunksize = int(np.ceil(len(pa_centers)/gcf_parms['cf_pa_num_chunk']))
    time_chunksize = mxds.attrs[sel_parms['xds']][sel_parms['data']].chunks[0]
    
    n_beam = pa_dif.shape[1]
    n_beam_pair = pa_map.shape[1]
    
    cf_time_map = xr.DataArray(da.from_array(cf_time_map,chunks=(time_chunksize)), dims=('time'))
    pa_centers = xr.DataArray(da.from_array(pa_centers,chunks=(pa_chunksize)), dims=('pa'))
    pa_dif = xr.DataArray(da.from_array(pa_dif,chunks=(time_chunksize,n_beam)), dims=('time','beam'))
    pa_map = xr.DataArray(da.from_array(pa_map,chunks=(pa_chunksize,n_beam_pair,2)), dims=('cf_time','beam_pair','pair'))

    
    #return 0,0,0
    return cf_time_map, pa_centers, pa_dif, pa_map
    '''
    return 0,0,0,0
  
@jit(nopython=True,cache=True)
def _make_map(pa_time_beam_pairs_unique,pa_time_beam_pairs,pa_pair_map,pa_centers):
    
    n_time = len(pa_time_beam_pairs)
    n_beam_pairs = pa_time_beam_pairs.shape[1]
    
    cf_time_map = np.zeros((n_time),numba.u4)
    
    for ii in range(n_time):
        indx_list = [42]
        for jj in range(n_beam_pairs):
            beam_pair_indx = pa_time_beam_pairs[ii,jj]
            mapping_indx = np.where(pa_time_beam_pairs_unique[:,jj] == beam_pair_indx)
            
            for mi in mapping_indx[0]:
                indx_list.append(mi)
            
        indx_list.pop(0)
        indx_arr = np.array(indx_list)
        indx_counted = np.bincount(indx_arr)
        sel_indx = np.bincount(indx_arr).argmax()
        
        if indx_counted[sel_indx] < n_beam_pairs:
            print('ERROR time aligment of PA pairs failed')
        
        cf_time_map[ii] = sel_indx
        
        
    n_unique_time = pa_time_beam_pairs_unique.shape[0]
    n_unique_beam_pairs = pa_time_beam_pairs_unique.shape[1]
    pa_map = np.zeros((n_unique_time,n_unique_beam_pairs,2),numba.u4)
    
    #print('pa_map shape ',pa_map.shape)
    
    for ii in range(n_unique_time):
        for jj in range(n_unique_beam_pairs):
            pa_map[ii,jj,0] = pa_pair_map[pa_time_beam_pairs_unique[ii,jj],0]
            pa_map[ii,jj,1] = pa_pair_map[pa_time_beam_pairs_unique[ii,jj],1]
        
    #print(pa_time_beam_pairs)
    #print(pa_time_beam_pairs_unique)
    #print(cf_pa_pair_map)
    return cf_time_map, pa_map
   
@jit(nopython=True,cache=True)
def _find_optimal_set_angle(nd_vals,val_step):
    vals_flat = np.ravel(nd_vals)
    n_vals = len(vals_flat)
    neighbours = np.zeros((n_vals,n_vals),numba.b1)

    for ii in range(n_vals):
        for jj in range(n_vals):
            #https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
            ang_dif = vals_flat[ii]-vals_flat[jj]
            ang_dif = np.abs((ang_dif + np.pi)%(2*np.pi) - np.pi)
            
            #neighbours_dis[ii,jj] = ang_dif
            
            if ang_dif <= val_step:
             neighbours[ii,jj] = True
             
    neighbours_rank = np.sum(neighbours,axis=1)
    vals_centers = [42.0] #Dummy value to let numba know what dtype of list is
    lonely_neighbour = True
    while lonely_neighbour:
        #if True:
        neighbours_rank = np.sum(neighbours,axis=1)
        highest_ranked_neighbour_indx = np.argmax(neighbours_rank)
        
        if neighbours_rank[highest_ranked_neighbour_indx]==0:
            lonely_neighbour = False
        else:
            group_members = np.where(neighbours[highest_ranked_neighbour_indx,:]==1)[0]
            vals_centers.append(vals_flat[highest_ranked_neighbour_indx]) #no outliers
            #vals_centers.append(np.median(vals_flat[neighbours[highest_ranked_neighbour_indx,:]])) #best stats
            #vals_centers.append(np.mean(vals_flat[neighbours[highest_ranked_neighbour_indx,:]])) #?
            
            for group_member in group_members:
                for ii in range(n_vals):
                    neighbours[group_member,ii] = 0
                    neighbours[ii,group_member] = 0
                    
    vals_centers.pop(0)
    vals_centers = np.array(vals_centers)
    

    n_time = nd_vals.shape[0]
    n_beam = nd_vals.shape[1]
    vals_dif = np.zeros(nd_vals.shape,numba.f8)
    
    for ii in range(n_time):
        for kk in range(n_beam):
            min_dif = 42.0 #Dummy value to let numba know what dtype of list is
            group_indx = -1
            for jj in range(len(vals_centers)):
                #https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
                ang_dif = nd_vals[ii,kk]-vals_centers[jj]
                ang_dif = np.abs((ang_dif + np.pi)%(2*np.pi) - np.pi)
                
                if min_dif > ang_dif:
                    min_dif = ang_dif
            
            vals_dif[ii,kk] = min_dif
            

    
    return vals_centers, vals_dif
    ################################################################################################################
    
    
def q(pa,pa_step,beam_pair_id,beam_ids):
    n_pa_centers = len(pa_centers)

    pa_beam_indx = np.zeros((n_time,n_beam),numba.u4)
    pa_dif = np.zeros((n_time,n_beam),numba.f8)
    
    for ii in range(n_time):
        for kk in range(n_beam):
            min_dif = 42.0 #Dummy value to let numba know what dtype of list is
            group_indx = -1
            for jj in range(len(pa_centers)):
                #https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
                ang_dif = pa[ii,kk]-pa_centers[jj]
                ang_dif = np.abs((ang_dif + np.pi)%(2*np.pi) - np.pi)
                
                if min_dif > ang_dif:
                    min_dif = ang_dif
                    pa_center_indx = jj
            
            pa_beam_indx[ii,kk] = pa_center_indx
            pa_dif[ii,kk] = min_dif
            
    n_pa_center_pairs = int((n_pa_centers**2 + n_pa_centers)/2)
    #print(n_pa_center_pairs)
    
    n_beam_pairs = len(beam_pair_id)
    pa_time_beam_pairs = np.zeros((n_time,n_beam_pairs),numba.u4)
    
    
    pa_pair_map = np.zeros((n_pa_center_pairs,2),numba.u4)
    n = 0
    for ii in range(n_pa_centers):
        for jj in range(ii,n_pa_centers):
            pa_pair_map[n,:] = [ii,jj]
            n = n+1
    
    for ii in range(n_time):
        m = 0
        for jj in range(n_beam):
            for kk in range(jj,n_beam):
                pa_beam_1_indx = pa_beam_indx[ii,jj]
                pa_beam_2_indx = pa_beam_indx[ii,kk]
                
                if pa_beam_1_indx > pa_beam_2_indx:
                    temp1 = pa_beam_1_indx
                    pa_beam_1_indx = pa_beam_2_indx
                    pa_beam_2_indx = temp1
                
                beam_pair_indx = int(pa_beam_1_indx*(2*n_pa_centers-pa_beam_1_indx-1)/2 + pa_beam_2_indx)
                
                #if (pa_pair_map[beam_pair_indx,0] != pa_beam_1_indx) or (pa_pair_map[beam_pair_indx,1] != pa_beam_2_indx):
                #    print('equal',n_pa_centers,beam_pair_indx,pa_pair_map[beam_pair_indx,:],pa_beam_1_indx,pa_beam_2_indx)
                #print(pa_beam_1_indx,pa_beam_2_indx)
                
                pa_time_beam_pairs[ii,m] = beam_pair_indx
                #print(m,pa_beam_1_indx,pa_beam_2_indx,beam_pair_indx)
                m=m+1
                
                
    #pa_time_beam_pairs_unique , ind = np.unique(pa_time_beam_pairs,axis=0, return_index=True)
    #pa_time_beam_pairs_unique = pa_time_beam_pairs_unique[np.argsort(ind)]
    
    
    
    pa_time_beam_pairs_unique = np.zeros(pa_time_beam_pairs.shape,numba.u4) - 42

    
    
    #for ii in range(n_beam_pairs):
    #    temp2 = np.unique(pa_time_beam_pairs[:,ii])
    #    pa_time_beam_pairs_unique[:len(temp2),ii] = temp2
        
    #print(pa_time_beam_pairs)
    #print(pa_time_beam_pairs_unique)
    #print(beam_pair_id)
 
    '''
    for mm in range(n_beam_pairs):
        unique_beam_pa = np.unique(cf_pa[:,mm])
        print(unique_beam_pa.shape,unique_beam_pa)
    print(cf_pa)
    '''

    
    #print(np.unique(cf_pa,axis=0))
    
    '''
    for ii in range(n_time):
        for jj in range(n_beam_pairs):
            flat_indx =
            cf_pa[ii,jj] =
    
    
    
    
    for ii in range(n_time):
        for jj in range(n_beam_pairs):
            cf_pa[ii,jj] =
    
    
    
    for ii in range(n_time):
        m = 0
        for jj in range(n_beam):
            for kk in range(jj,n_beam):
                beam_1_id = beam_ids[jj]
                beam_2_id = beam_ids[kk]
                
                beam_pair_id,beam_ids
                cf_pa[ii,m] =
                m = m+1
    '''
    
        
    
    return pa_time_beam_pairs,pa_pair_map,pa_centers,pa_dif
    #return pa_centers,cf_pa_map,pa_dif
