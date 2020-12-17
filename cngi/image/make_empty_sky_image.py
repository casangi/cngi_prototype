#   Copyright 2020 AUI, Inc. Washington DC, USA
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


def make_empty_sky_image(img_dataset,phase_center,image_size,cell_size,chan_coords,chan_width,pol_coords,time_coords,direction_reference='FK5',projection='SIN', spectral_reference='lsrk',velocity_type='radio',
num_taylor_terms=0):
    """
    Create an img_dataset with only coordinates (no datavariables).
    The image dimensionality is either:
        l, m, time, chan, pol
    or
        l, m, time, taylor, pol (if num_taylor_terms > 0)

    Parameters
    ----------
    img_dataset : xarray.Dataset
        Empty dataset (dataset = xarray.Dataset())
    phase_center : array of number, length = 2, units = rad
        Image phase center.
    image_size : array of int, length = 2, units = rad
        Number of x and y axis pixels in image.
    cell_size : array of number, length = 2, units = rad
        Cell size of x and y axis pixels in image.
    chan_coords : xarray.DataArray
        The center frequency of each image channel.
    chan_width : xarray.DataArray
        The frequency width of each image channel.
    pol_coords : xarray.DataArray
        The polarization code for each image polarization.
    time_coords : xarray.DataArray
        The time for each image time step.
    direction_reference : str, default = 'FK5'
    projection : str, default = 'SIN'
    spectral_reference : str, default = 'lsrk'
    velocity_type : str, default = 'radio'
    num_taylor_terms : int, default =0
    Returns
    -------
    xarray.Dataset
        new xarray Datasets
    """
    import xarray as xr
    import numpy as np

    from astropy.wcs import WCS
    rad_to_deg =  180/np.pi
    w = WCS(naxis=2)
    w.wcs.crpix = np.array(image_size)//2
    w.wcs.cdelt = np.array(cell_size)*rad_to_deg
    w.wcs.crval = np.array(phase_center)*rad_to_deg
    w.wcs.ctype = ['RA---'+projection,'DEC--'+projection]
    
    x = np.arange(image_size[0])
    y = np.arange(image_size[1])
    X, Y = np.meshgrid(x, y,indexing='ij')
    ra, dec = w.wcs_pix2world(X, Y, 1)
    
    image_center = np.array(image_size)//2
    l = np.arange(-image_center[0], image_size[0]-image_center[0])*cell_size[0]
    m = np.arange(-image_center[1], image_size[1]-image_center[1])*cell_size[1]
    
    if num_taylor_terms == 0:
        coords = {'time':time_coords.data,'chan': chan_coords.data, 'pol': pol_coords.data, 'chan_width' : ('chan',chan_width.data),'l':l,'m':m,'right_ascension' : (('l','m'),ra/rad_to_deg),'declination' : (('l','m'),dec/rad_to_deg)}
        img_dataset.attrs['axis_units'] =  ['rad', 'rad', 'time', 'Hz', 'pol']
    else:
        coords = {'time':time_coords.data,'taylor': np.arange(num_taylor_terms), 'pol': pol_coords.data,'l':np.arange(image_size[0]),'m':np.arange(image_size[1]),'right_ascension' : (('l','m'),ra/rad_to_deg),'declination' : (('l','m'),dec/rad_to_deg)}
        img_dataset.attrs['axis_units'] =  ['rad', 'rad', 'time', 'taylor', 'pol']
    
    img_dataset = img_dataset.assign_coords(coords)
        
    img_dataset.attrs['direction_reference'] = direction_reference
    img_dataset.attrs['spectral_reference'] = spectral_reference
    img_dataset.attrs['velocity_type'] = velocity_type
    
    return img_dataset
