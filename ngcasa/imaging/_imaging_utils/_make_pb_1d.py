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

#Simple 1D Cases
#Airy Disk dish, blockage, freq
#Gaussian halfWidth
#Poly
#Cos Polyp
#Inverse Poly coeff

def _airy_disk(freq_chan,pol,make_pb_parms):
    '''
    Does not yet handle beam squint
    dish_diameters : list of int
    blockage_diameters : list of int
    frequencies : list of number
    '''

    import numpy as np
    import scipy.constants
    from scipy.special import jn
    
    cell = make_pb_parms['cell']
    image_size = make_pb_parms['imsize']
    center_indx = make_pb_parms['center_indx']
    list_dish_diameters = make_pb_parms['list_dish_diameters']
    list_blockage_diameters = make_pb_parms['list_blockage_diameters']
    ipower = make_pb_parms['ipower']
    pb_limit = make_pb_parms['pb_limit']
    
    c = scipy.constants.c #299792458
    k = (2*np.pi*freq_chan)/c
    
    if not center_indx:
        center_indx = image_size//2
        
    x = np.arange(-center_indx[0], image_size[0]-center_indx[0])*cell[0]
    y = np.arange(-center_indx[1], image_size[1]-center_indx[1])*cell[1]
    
    airy_disk_size = (image_size[0],image_size[1],len(freq_chan),1,len(list_blockage_diameters)) #len(pol) is set initially to 1. For now, the PB is assumed the same. This will change.
    airy_disk =  np.zeros(airy_disk_size)
    
    for i, (dish_diameter, blockage_diameter) in enumerate(zip(list_dish_diameters, list_blockage_diameters)):
        
        aperture = dish_diameter/2
        x_grid, y_grid = np.meshgrid(y,x)
        
        r_grid = (np.sin(np.sqrt(x_grid**2 + y_grid**2))[:,:,None]*k*aperture) #d0 x d1 x chan
        r_grid[center_indx[0],center_indx[1],:] = 1.0 #Avoid the 0/0 for the centre value. The value should be 1.0.
        
        if blockage_diameter==0.0:
            airy_disk[:,:,:,0,i] = (2.0*jn(1,r_grid)/r_grid)**ipower
        else:
            area_ratio = (dish_diameter/blockage_diameter)**2
            length_ratio = (dish_diameter/blockage_diameter)
            airy_disk[:,:,:,0,i] = ((area_ratio * 2.0 * jn(1,r_grid)/r_grid   - 2.0 * jn(1, r_grid * length_ratio)/(r_grid * length_ratio) )/(area_ratio - 1.0))**ipower
            
    airy_disk[center_indx[0],center_indx[1],:,0,:] = 1.0 #Fix centre value
    airy_disk[airy_disk<pb_limit] = 0.0
    airy_disk = np.tile(airy_disk,(1,1,1,len(pol),1))
    
    return airy_disk
    

