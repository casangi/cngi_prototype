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

#Simple 1D Cases
#Airy Disk dish, blockage, freq
#Gaussian halfWidth
#Poly
#Cos Polyp
#Inverse Poly coeff

#Formula for obscured airy pattern found in https://en.wikipedia.org/wiki/Airy_disk (see Obscured Airy pattern section)
# If ipower is 1 the voltage pattern is returned if ipower is 2 the primary beam is returned.
def _airy_disk(freq_chan,pol,pb_parms,grid_parms):
    '''
    Does not yet handle beam squint
    dish_diameters : list of int
    blockage_diameters : list of int
    frequencies : list of number
    '''

    import numpy as np
    import scipy.constants
    from scipy.special import jn
    
    cell = grid_parms['cell_size']
    image_size = grid_parms['image_size']
    image_center = grid_parms['image_center']
    
    list_dish_diameters = pb_parms['list_dish_diameters']
    list_blockage_diameters = pb_parms['list_blockage_diameters']
    ipower = pb_parms['ipower']
    
    c = scipy.constants.c #299792458
    k = (2*np.pi*freq_chan)/c
    
        
    x = np.arange(-image_center[0], image_size[0]-image_center[0])*cell[0]
    y = np.arange(-image_center[1], image_size[1]-image_center[1])*cell[1]
    
    airy_disk_size = (image_size[0],image_size[1],len(freq_chan),1,len(list_blockage_diameters)) #len(pol) is set initially to 1. For now, the PB is assumed the same. This will change.
    airy_disk =  np.zeros(airy_disk_size)
    
    for i, (dish_diameter, blockage_diameter) in enumerate(zip(list_dish_diameters, list_blockage_diameters)):
        
        aperture = dish_diameter/2
        x_grid, y_grid = np.meshgrid(x,y,indexing='ij')
        
        #r_grid = (np.sin(np.sqrt(x_grid**2 + y_grid**2))[:,:,None]*k*aperture) #d0 x d1 x chan
        r_grid = (np.sqrt(x_grid**2 + y_grid**2)[:,:,None]*k*aperture) #d0 x d1 x chan
        r_grid[image_center[0],image_center[1],:] = 1.0 #Avoid the 0/0 for the centre value.
        
        if blockage_diameter==0.0:
            airy_disk[:,:,:,0,i] = (2.0*jn(1,r_grid)/r_grid)**ipower
        else:
            e = blockage_diameter/dish_diameter
            airy_disk[:,:,:,0,i] = (( 2.0 * jn(1,r_grid)/r_grid   - 2.0 * e * jn(1, r_grid * e)/r_grid )/(1.0 - e**2))**ipower
    
    airy_disk[image_center[0],image_center[1],:,0,:] = 1.0 #Fix centre value
    airy_disk = np.tile(airy_disk,(1,1,1,len(pol),1))
    
    return airy_disk
    

#Formula for obscured airy pattern found in casa6/casa5/code/synthesis/TransformMachines/PBMath1DAiry.cc/h
# If ipower is 1 the voltage pattern is returned if ipower is 2 the primary beam is returned.
def _casa_airy_disk(freq_chan,pol,pb_parms,grid_parms):
    '''
    Does not yet handle beam squint
    dish_diameters : list of int
    blockage_diameters : list of int
    frequencies : list of number
    '''

    import numpy as np
    import scipy.constants
    from scipy.special import jn
    
    cell = grid_parms['cell_size']
    image_size = grid_parms['image_size']
    image_center = grid_parms['image_center']
    
    list_dish_diameters = pb_parms['list_dish_diameters']
    list_blockage_diameters = pb_parms['list_blockage_diameters']
    ipower = pb_parms['ipower']
    
    c = scipy.constants.c #299792458
    k = (2*np.pi*freq_chan)/c
    
        
    x = np.arange(-image_center[0], image_size[0]-image_center[0])*cell[0]
    y = np.arange(-image_center[1], image_size[1]-image_center[1])*cell[1]
    
    airy_disk_size = (image_size[0],image_size[1],len(freq_chan),1,len(list_blockage_diameters)) #len(pol) is set initially to 1. For now, the PB is assumed the same. This will change.
    airy_disk =  np.zeros(airy_disk_size)
    
    for i, (dish_diameter, blockage_diameter) in enumerate(zip(list_dish_diameters, list_blockage_diameters)):
        
        aperture = dish_diameter/2
        x_grid, y_grid = np.meshgrid(x,y,indexing='ij')
        
        #r_grid = (np.sin(np.sqrt(x_grid**2 + y_grid**2))[:,:,None]*k*aperture) #d0 x d1 x chan
        r_grid = (np.sqrt(x_grid**2 + y_grid**2)[:,:,None]*k*aperture) #d0 x d1 x chan
        r_grid[image_center[0],image_center[1],:] = 1.0 #Avoid the 0/0 for the centre value.
        
        if blockage_diameter==0.0:
            airy_disk[:,:,:,0,i] = (2.0*jn(1,r_grid)/r_grid)**ipower
        else:
            area_ratio = (dish_diameter/blockage_diameter)**2
            length_ratio = (dish_diameter/blockage_diameter)
            airy_disk[:,:,:,0,i] = ((area_ratio * 2.0 * jn(1,r_grid)/r_grid   - 2.0 * jn(1, r_grid * length_ratio)/(r_grid * length_ratio) )/(area_ratio - 1.0))**ipower
    
    airy_disk[image_center[0],image_center[1],:,0,:] = 1.0 #Fix centre value
    airy_disk = np.tile(airy_disk,(1,1,1,len(pol),1))
    
    return airy_disk
    


#Functions used during the creatiuon of the gridding convolution functions.
#Formula for obscured airy pattern found in https://en.wikipedia.org/wiki/Airy_disk (see Obscured Airy pattern section)
# If ipower is 1 the voltage pattern is returned if ipower is 2 the primary beam is returned.
def _airy_disk_rorder(freq_chan,pol,pb_parms,grid_parms):
    '''
    Does not yet handle beam squint
    dish_diameters : list of int
    blockage_diameters : list of int
    frequencies : list of number
    '''

    import numpy as np
    import scipy.constants
    from scipy.special import jn
    
    cell = grid_parms['cell_size']
    image_size = grid_parms['image_size']
    image_center = grid_parms['image_center']
    
    list_dish_diameters = pb_parms['list_dish_diameters']
    list_blockage_diameters = pb_parms['list_blockage_diameters']
    ipower = pb_parms['ipower']
    
    c = scipy.constants.c #299792458
    k = (2*np.pi*freq_chan)/c
    
    x = np.arange(-image_center[0], image_size[0]-image_center[0])*cell[0]
    y = np.arange(-image_center[1], image_size[1]-image_center[1])*cell[1]
    
    airy_disk_size = (len(list_blockage_diameters),len(freq_chan),1,image_size[0],image_size[1]) #len(pol) is set initially to 1. For now, the PB is assumed the same. This will change.
    airy_disk =  np.zeros(airy_disk_size)
    
    for i, (dish_diameter, blockage_diameter) in enumerate(zip(list_dish_diameters, list_blockage_diameters)):
        
        aperture = dish_diameter/2
        x_grid, y_grid = np.meshgrid(x,y,indexing='ij')
        
        #r_grid = (np.sin(np.sqrt(x_grid**2 + y_grid**2))[:,:,None]*k*aperture) #d0 x d1 x chan
        r_grid = np.moveaxis((np.sqrt(x_grid**2 + y_grid**2)[:,:,None]*k*aperture),2,0) #chan x d0 x d1
        r_grid[:,image_center[0],image_center[1]] = 1.0 #Avoid the 0/0 for the centre value.
        
        if blockage_diameter==0.0:
            airy_disk[i,:,0,:,:] = (2.0*jn(1,r_grid)/r_grid)**ipower
        else:
            e = blockage_diameter/dish_diameter
            airy_disk[i,:,0,:,:] = (( 2.0 * jn(1,r_grid)/r_grid   - 2.0 * e * jn(1, r_grid * e)/r_grid )/(1.0 - e**2))**ipower
    
    airy_disk[:,:,0,image_center[0],image_center[1]] = 1.0 #Fix centre value
    #airy_disk[airy_disk<pb_limit] = 0.0
    airy_disk = np.tile(airy_disk,(1,1,len(pol),1,1))
    
    return airy_disk

#Formula for obscured airy pattern found in casa6/casa5/code/synthesis/TransformMachines/PBMath1DAiry.cc/h
# If ipower is 1 the voltage pattern is returned if ipower is 2 the primary beam is returned.
def _casa_airy_disk_rorder(freq_chan,pol,pb_parms,grid_parms):
    '''
    Does not yet handle beam squint
    dish_diameters : list of int
    blockage_diameters : list of int
    frequencies : list of number
    '''

    import numpy as np
    import scipy.constants
    from scipy.special import jn
    
    cell = grid_parms['cell_size']
    image_size = grid_parms['image_size']
    image_center = grid_parms['image_center']
    
    list_dish_diameters = pb_parms['list_dish_diameters']
    list_blockage_diameters = pb_parms['list_blockage_diameters']
    ipower = pb_parms['ipower']
    
    c = scipy.constants.c #299792458
    k = (2*np.pi*freq_chan)/c
    
    x = np.arange(-image_center[0], image_size[0]-image_center[0])*cell[0]
    y = np.arange(-image_center[1], image_size[1]-image_center[1])*cell[1]
    
    airy_disk_size = (len(list_blockage_diameters),len(freq_chan),1,image_size[0],image_size[1]) #len(pol) is set initially to 1. For now, the PB is assumed the same. This will change.
    airy_disk =  np.zeros(airy_disk_size)
    
    for i, (dish_diameter, blockage_diameter) in enumerate(zip(list_dish_diameters, list_blockage_diameters)):
        
        aperture = dish_diameter/2
        x_grid, y_grid = np.meshgrid(x,y,indexing='ij')
        
        #r_grid = (np.sin(np.sqrt(x_grid**2 + y_grid**2))[:,:,None]*k*aperture) #d0 x d1 x chan
        r_grid = np.moveaxis((np.sqrt(x_grid**2 + y_grid**2)[:,:,None]*k*aperture),2,0) #chan x d0 x d1
        r_grid[:,image_center[0],image_center[1]] = 1.0 #Avoid the 0/0 for the centre value.
        
        if blockage_diameter==0.0:
            airy_disk[i,:,0,:,:] = (2.0*jn(1,r_grid)/r_grid)**ipower
        else:
            area_ratio = (dish_diameter/blockage_diameter)**2
            length_ratio = (dish_diameter/blockage_diameter)
            airy_disk[i,:,0,:,:] = ((area_ratio * 2.0 * jn(1,r_grid)/r_grid   - 2.0 * jn(1, r_grid * length_ratio)/(r_grid * length_ratio) )/(area_ratio - 1.0))**ipower
    
    airy_disk[:,:,0,image_center[0],image_center[1]] = 1.0 #Fix centre value
    airy_disk = np.tile(airy_disk,(1,1,len(pol),1,1))
    
    return airy_disk
