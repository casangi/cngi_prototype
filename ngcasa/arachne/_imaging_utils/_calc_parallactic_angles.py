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

def _calc_parallactic_angles_for_gcf(vis_dataset,global_dataset):
    
    pa = _calc_parallactic_angles_for_vis_dataset(vis_dataset,global_dataset)
    
    return pa

def _calc_parallactic_angles_for_vis_dataset(vis_dataset,global_dataset):
    from astropy.coordinates import (EarthLocation, SkyCoord,
                                     AltAz, CIRS)
    import astropy.units as u
    from astropy.time import Time
    
    ra = global_dataset.FIELD_PHASE_DIR[0,0,0].values
    dec = global_dataset.FIELD_PHASE_DIR[0,1,0].values
    phase_center = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='fk5') #fk5 epoch is J2000

    pa = np.zeros((global_dataset.dims['antenna'],vis_dataset.dims['time']))
    
    for ant_pos_indx in range(global_dataset.dims['antenna']):
        x = global_dataset.ANT_POSITION[ant_pos_indx,0].values
        y = global_dataset.ANT_POSITION[ant_pos_indx,1].values
        z = global_dataset.ANT_POSITION[ant_pos_indx,2].values
        
        #observing_location = EarthLocation(lat=34.1*u.degree, lon=-107.6*u.degree,height=2114.89*u.m,ellipsoid='WGS84')
        #observing_location = EarthLocation.of_site('vla')
        observing_location = EarthLocation.from_geocentric(x=x*u.m, y=y*u.m,z=z*u.m)
        
        reference_time = Time(vis_dataset.time.values.astype(str), format='isot', scale='utc', location=observing_location)
        
        pa[ant_pos_indx,:] = _calc_parallactic_angles(reference_time, observing_location, phase_center)
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
    return (phase_center_altaz.position_angle(pole_altaz).value)%(np.pi) - np.pi

