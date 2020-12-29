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

import numpy as np
import xarray as xr
import dask.array as da
import time
import matplotlib.pyplot as plt
from numba import jit
import numba

def _calc_parallactic_angles_for_gcf(mxds,gcf_parms,sel_parms):
    pa = _calc_parallactic_angles_for_vis_dataset(mxds,gcf_parms,sel_parms)
    cf_time_map, pa_centers, pa_dif = _make_cf_time_map(mxds,pa,gcf_parms,sel_parms)
    
    return cf_time_map, pa_centers, pa_dif


def _calc_parallactic_angles_for_vis_dataset(mxds,gcf_parms,sel_parms):
    from astropy.coordinates import (EarthLocation, SkyCoord,
                                     AltAz, CIRS)
    import astropy.units as u
    from astropy.time import Time
    vis_dataset = mxds.attrs[sel_parms['xds']]
    
    antenna_ids = mxds.antenna_ids.data
    n_ant = len(antenna_ids)
    
    #try:
    if True:
        ### Using pointing table
        print('Using Pointing dataset to calculate parallactic angles.')

        #print(mxds.pointing_ra_dec)
        ra_dec = mxds.pointing_ra_dec.interp(time=vis_dataset.time,assume_sorted=False,method=gcf_parms['interpolation_method'])
        #print(mxds.pointing_ra_dec)
        ra_dec = np.mean(ra_dec.data, axis=1)
        
        #print(ra_dec.shape)
        
    else:
        #### Using field table
        print('Using Field dataset to calculate parallactic angles.')
        field_dataset = mxds.attrs['FIELD']
        
        field_id = np.max(vis_dataset.FIELD_ID,axis=1).compute()
        n_field = field_dataset.dims['d0']
        n_time = vis_dataset.dims['time']
        
        ra_dec = field_dataset.PHASE_DIR.isel(d0=field_id).data.compute()
        
        #print(ra_dec.shape)
        if n_field != 1:
            ra_dec = ra_dec[:,0,:]
        
    ra = ra_dec[:,0]
    dec = ra_dec[:,1]
    
    phase_center = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='fk5') #fk5 epoch is J2000
    
    
    #telescope_name = mxds.attrs['OBSERVATION'].TELESCOPE_NAME.values[0]
    #observing_location = EarthLocation.of_site(telescope_name)
    
    #observing_location = EarthLocation(lat=34.1*u.degree, lon=-107.6*u.degree,height=2114.89*u.m,ellipsoid='WGS84')
    x = mxds.ANTENNA.POSITION[0,0].values
    y = mxds.ANTENNA.POSITION[0,1].values
    z = mxds.ANTENNA.POSITION[0,2].values
    observing_location = EarthLocation.from_geocentric(x=x*u.m, y=y*u.m,z=z*u.m)
    reference_time = Time(vis_dataset.time.values.astype(str), format='isot', scale='utc', location=observing_location)
    
    #if n_field == 1:
    #    phase_center = [phase_center]
    
    start = time.time()
    pa = _calc_parallactic_angles(reference_time, observing_location, phase_center)
    print('Time to calc pa ', time.time()-start)
    
    '''
    print(pa)
    q = pa%(2*np.pi)
    print('q',q)
    q[q > np.pi] = q[q > np.pi] - 2*np.pi
    print('q',q)
    
    print(mxds.POINTING.DIRECTION.values[0,:][0])
    print(ra_dec.values[0,:])
    print(mxds.POINTING.DIRECTION.values[0,:][0]-ra_dec.values[0,:])
    '''
    
    time_chunksize = vis_dataset[sel_parms['data']].chunks[0][0]
    pa = xr.DataArray(da.from_array(pa,chunks=(time_chunksize)), dims={'time'})
    
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

def _make_cf_time_map(mxds,pa,gcf_parms,sel_parms):
    
    start = time.time()
    pa_centers,cf_time_map,pa_dif = find_optimal_pa_set(pa.data.compute(),gcf_parms['pa_step'])
    print('Time to make cf_time_map ', time.time()-start)
    pa_centers = np.array(pa_centers)
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
            
    plt.figure()
    plt.plot(pa_dif,label='ngcasa')
    plt.plot(ang_dif_list,label='casa')
    plt.legend()
    
    plt.figure()
    plt.hist(pa_dif, bins='auto')
    
    plt.figure()
    plt.hist(ang_dif_list, bins='auto')
    
    
    
    print(np.mean(pa_dif),np.mean(ang_dif_list))
    print(np.median(pa_dif),np.median(ang_dif_list))
    print(np.var(pa_dif),np.var(ang_dif_list))
    print(len(pa_centers),pa_cf_count)
    plt.show()
    #############################################################
    '''
    
    pa_chunksize = np.ceil(len(pa_centers)/gcf_parms['cf_pa_num_chunk'])
    
    time_chunksize = mxds.attrs[sel_parms['xds']][sel_parms['data']].chunks[0][0]
    cf_time_map = xr.DataArray(da.from_array(cf_time_map,chunks=(time_chunksize)), dims={'time'})
    pa_centers = xr.DataArray(da.from_array(pa_centers,chunks=(pa_chunksize)), dims={'pa'})
    pa_dif = xr.DataArray(da.from_array(pa_dif,chunks=(time_chunksize)), dims={'time'})
    
    
    return cf_time_map, pa_centers, pa_dif

@jit(nopython=True,cache=True)
def find_optimal_pa_set(pa,pa_step):
    n_time  = len(pa)
    
    neighbours = np.zeros((n_time,n_time),numba.b1)
    #neighbours = np.zeros((n_time,n_time+1),bool)
    
    #neighbours_dis = np.zeros((n_time,n_time))
    
    for ii in range(n_time):
        for jj in range(n_time):
            #https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
            ang_dif = pa[ii]-pa[jj]
            ang_dif = np.abs((ang_dif + np.pi)%(2*np.pi) - np.pi)
            
            #neighbours_dis[ii,jj] = ang_dif
            
            if ang_dif <= pa_step:
             neighbours[ii,jj] = True
             
    '''
    plt.figure()
    plt.imshow(neighbours_dis)
    
    plt.figure()
    plt.imshow(np.abs(neighbours_dis))
    plt.show()
    '''
    
    neighbours_rank = np.sum(neighbours,axis=1)
    
    group = np.zeros((n_time,n_time+1),numba.u4)
    #group = np.zeros((n_time,n_time+1),int)
    
    pa_centers = [42.0] #Dummy value to let numba know what dtype of list is
    lonely_neighbour = True
    while lonely_neighbour:
        #if True:
        neighbours_rank = np.sum(neighbours,axis=1)
        highest_ranked_neighbour_indx = np.argmax(neighbours_rank)
        
        if neighbours_rank[highest_ranked_neighbour_indx]==0:
            lonely_neighbour = False
        else:
            group_members = np.where(neighbours[highest_ranked_neighbour_indx,:]==1)[0]
            pa_centers.append(pa[highest_ranked_neighbour_indx])
            
            for group_member in group_members:
                for ii in range(n_time):
                    neighbours[group_member,ii] = 0
                    neighbours[ii,group_member] = 0
                    
    pa_centers.pop(0)
    #pa_centers.sort() #slow
    
    cf_pa_map = np.zeros((n_time),numba.u4)
    pa_dif = np.zeros((n_time),numba.f8)
    
    for ii in range(n_time):
        min_dif = 42.0
        group_indx = -1
        for jj in range(len(pa_centers)):
            #https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
            ang_dif = pa[ii]-pa_centers[jj]
            ang_dif = np.abs((ang_dif + np.pi)%(2*np.pi) - np.pi)
            
            if min_dif > ang_dif:
                min_dif = ang_dif
                group_indx = jj
        
        cf_pa_map[ii] = group_indx
        pa_dif[ii] = min_dif
    
    return pa_centers,cf_pa_map,pa_dif
