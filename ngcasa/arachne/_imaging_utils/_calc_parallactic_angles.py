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
    pa = _calc_parallactic_angles_for_vis_dataset(mxds,sel_parms)
    cf_time_map, pa_centers = _make_cf_time_map(mxds,pa,gcf_parms)
    
    return pa, cf_time_map

def _calc_parallactic_angles_for_vis_dataset(mxds,sel_parms):
    from astropy.coordinates import (EarthLocation, SkyCoord,
                                     AltAz, CIRS)
    import astropy.units as u
    from astropy.time import Time
    
    vis_dataset = mxds.attrs[sel_parms['xds']]
    field_dataset = mxds.attrs['FIELD']
    
    field_id = np.max(vis_dataset.FIELD_ID,axis=1).compute()
    
    ra = field_dataset.PHASE_DIR[:,0,0].data.compute()
    dec = field_dataset.PHASE_DIR[:,0,1].data.compute()
    phase_center = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='fk5') #fk5 epoch is J2000
    
    telescope_name = mxds.attrs['OBSERVATION'].TELESCOPE_NAME.values[0]
    observing_location = EarthLocation.of_site(telescope_name)
    #observing_location = EarthLocation(lat=34.1*u.degree, lon=-107.6*u.degree,height=2114.89*u.m,ellipsoid='WGS84')
    #x = mxds.ANTENNA.POSITION[:,0].values
    #y = mxds.ANTENNA.POSITION[:,1].values
    #z = mxds.ANTENNA.POSITION[:,2].values
    #observing_location = EarthLocation.from_geocentric(x=x*u.m, y=y*u.m,z=z*u.m)
    reference_time = Time(vis_dataset.time.values.astype(str), format='isot', scale='utc', location=observing_location)
    
    n_field = field_dataset.dims['d0']
    n_time = vis_dataset.dims['time']
    
    if n_field == 1:
        phase_center = [phase_center]
    
    '''
    #Slow
    pa_slow = np.zeros((n_time))
    start = time.time()
    for time_indx in range(n_time):
        pa_slow[time_indx] = _calc_parallactic_angles(reference_time[time_indx], observing_location, phase_center[field_id[time_indx]])
    print('Time to calc pa ', time.time()-start)
    '''
    #Larger number of calculations (a lot of unnecessary calculations), but better vectorized for astropy interface
    pa_redunant = np.zeros((n_time,n_field))
    pa = np.zeros((n_time))
    start = time.time()
    for field_indx in range(n_field):
        pa_redunant[:,field_indx] = _calc_parallactic_angles(reference_time, observing_location, phase_center[field_indx])
    
    for time_indx in range(n_time):
        pa[time_indx] = pa_redunant[time_indx,field_id[time_indx]]
    print('Time to calc pa ', time.time()-start)
    
    time_chunksize = vis_dataset[sel_parms['data']].chunks[0][0]
    
    pa = xr.DataArray(da.from_array(pa,chunks=(time_chunksize)), dims={'time'})
    
    '''
    n_field = field_dataset.dims['d0']
    pa = np.zeros((vis_dataset.dims['time'],n_field))
    start = time.time()
    for field_indx in range(n_field):
        pa[:,field_indx] = _calc_parallactic_angles(reference_time, observing_location, phase_center[field_indx])
    print('Time to calc pa ', time.time()-start)
    '''
    
    '''
    field_ra_dec[0,0,0] = np.nan
    
    
    ra = field_ra_dec[:,0,0].data#.compute()
    dec = field_ra_dec[:,0,1].data#.compute()
    print(ra)
    phase_center = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='fk5') #fk5 epoch is J2000
    
    print(phase_center)
    
    
    #observing_location = EarthLocation.of_site('vla')
    telescope_name = mxds.attrs['OBSERVATION'].TELESCOPE_NAME.values[0]
    observing_location = EarthLocation.of_site(telescope_name)
    reference_time = Time(vis_dataset.time.values.astype(str), format='isot', scale='utc', location=observing_location)
    
    
    start = time.time()
    pa = _calc_parallactic_angles(reference_time, observing_location, phase_center)
    print('Time to calc pa ', time.time()-start)
    
    print(pa)
    '''
    '''
    
    #print(field_dataset.PHASE_DIR.isel(d0=vis_dataset.FIELD_ID.data.compute()[:,0]))
    
    
    
    #reference_time = Time(vis_dataset.time.values.astype(str), format='isot', scale='utc', location=observing_location)
    
    #ra = vis_dataset.FIELD_ID
    
    #phase_center = SkyCoord(ra=*u.rad, dec=dec*u.rad, frame='fk5') #fk5 epoch is J2000
    
    ra = global_dataset.FIELD_PHASE_DIR[0,0,0].values
    dec = global_dataset.FIELD_PHASE_DIR[0,1,0].values
    phase_center = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='fk5') #fk5 epoch is J2000

    n_ant = global_dataset.dims['antenna']
    pa = np.zeros((vis_dataset.dims['time'],n_ant))
    
    start = time.time()
    for ant_pos_indx in range(n_ant):
        x = global_dataset.ANT_POSITION[ant_pos_indx,0].values
        y = global_dataset.ANT_POSITION[ant_pos_indx,1].values
        z = global_dataset.ANT_POSITION[ant_pos_indx,2].values
        
        #observing_location = EarthLocation(lat=34.1*u.degree, lon=-107.6*u.degree,height=2114.89*u.m,ellipsoid='WGS84')
        #observing_location = EarthLocation.of_site('vla')
        observing_location = EarthLocation.from_geocentric(x=x*u.m, y=y*u.m,z=z*u.m)
        
        reference_time = Time(vis_dataset.time.values.astype(str), format='isot', scale='utc', location=observing_location)
        
        pa[:,ant_pos_indx] = _calc_parallactic_angles(reference_time, observing_location, phase_center)
    print('Time to calc pa ', time.time()-start)
        
    time_chunksize = vis_dataset[sel_parms['data']].chunks[0][0]
    pa = xr.DataArray(da.from_array(pa,chunks=(time_chunksize,n_ant)), dims={'time','ant'})
    '''
    return pa
    


def _calc_parallactic_angles(times, observing_location, phase_center):
    """
    Computes parallactic angles per timestep for the given
    reference antenna position and field centre.
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
    
    return phase_center_altaz.position_angle(pole_altaz).value

def _make_cf_time_map(mxds,pa,gcf_parms):
    cf_time_map = 0
    
    start = time.time()
    pa_centers,cf_time_map = find_optimal_pa_set(pa.data.compute(),gcf_parms['pa_step'])
    print('Time to make cf_time_map ', time.time()-start)
    pa_centers = np.array(pa_centers)
    

    '''
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
    plt.plot(cf_time_map[:,1])
    plt.plot(ang_dif_list)
    
    plt.figure()
    plt.hist(cf_time_map[:,1], bins='auto')
    
    plt.figure()
    plt.hist(ang_dif_list, bins='auto')
    
    plt.show()
    
    print(np.mean(cf_time_map[:,1]),np.mean(ang_dif_list))
    print(np.median(cf_time_map[:,1]),np.median(ang_dif_list))
    print(np.var(cf_time_map[:,1]),np.var(ang_dif_list))
    print(pa_cf_count)
    '''
    
    return cf_time_map, pa_centers

@jit(nopython=True)
def find_optimal_pa_set(pa,pa_step):
    print(pa)
    print(pa_step)
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
    
    cf_pa_map = np.zeros((n_time,2),numba.f8)
    
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
        
        cf_pa_map[ii,0] = group_indx
        cf_pa_map[ii,1] = min_dif
            
          

    
    '''
    #group = [[None for _ in range(1)] for _ in range(n_time)]
    group = [[] for _ in range(n_time)]
    print(group)
    
    
    for ii in range(n_time):
        for jj in range(n_time):
            
            if np.abs(pa[ii]-pa[jj]) <= pa_step:
                group[ii] = group[ii] + jj
                
    print(group)
    '''
    
    
    return pa_centers,cf_pa_map
