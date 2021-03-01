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


def _calc_phase_gradient_pointings(mxds,pointing_ra_dec,gcf_parms,sel_parms):
    pointing_set = _find_optimal_set_pointing(pointing_ra_dec.data.compute(),gcf_parms['pointing_step'])
    
    pointing_set = xr.DataArray(da.from_array(pointing_set,chunks=(pointing_set.shape[0],pointing_set.shape[1])), dims=('cf_pointing','pair'))
    
    return pointing_set
    
#Ideas: split problem by field and solve for each field separately.
#https://stackoverflow.com/questions/15882202/minimum-number-of-circles-with-radius-r-to-cover-n-points
#https://cs.stanford.edu/people/paulliu/files/cccg-2016-long.pdf
#https://link.springer.com/chapter/10.1007/978-3-642-35452-6_18
#minimum geometric disk cover
@jit(nopython=True,cache=True)
def _find_optimal_set_pointing(nd_vals,val_step):
    ra = np.ravel(nd_vals[:,:,0])
    dec = np.ravel(nd_vals[:,:,1])
 
    
    n_vals = len(ra)
    neighbours = np.zeros((n_vals,n_vals),numba.b1)
    #neighbours = np.zeros((n_vals,n_vals),bool)

    for ii in range(n_vals):
        for jj in range(n_vals):
            #https://stjerneskinn.com/angular-distance-between-stars.htm
            #http://spiff.rit.edu/classes/phys373/lectures/radec/radec.html
            #dis = np.abs(np.arccos(np.cos(dec[ii])*np.cos(dec[jj])*np.cos(ra[jj]-ra[ii]) + np.sin(dec[ii])*np.sin(dec[jj])))
            dis = np.sqrt(((ra[ii]-ra[jj])*np.cos(dec[ii]))**2 + (dec[ii]-dec[jj])**2)
            
            #neighbours_dis[ii,jj] = ang_dif
            if dis <= val_step:
                neighbours[ii,jj] = True
             
    neighbours_rank = np.sum(neighbours,axis=1)
    vals_centers = [[42.0,42.0]] #Dummy value to let numba know what dtype of list is
    lonely_neighbour = True
    while lonely_neighbour:
        #if True:
        neighbours_rank = np.sum(neighbours,axis=1)
        highest_ranked_neighbour_indx = np.argmax(neighbours_rank)
        
        if neighbours_rank[highest_ranked_neighbour_indx]==0:
            lonely_neighbour = False
        else:
            group_members = np.where(neighbours[highest_ranked_neighbour_indx,:]==1)[0]
            vals_centers.append([ra[highest_ranked_neighbour_indx],dec[highest_ranked_neighbour_indx]]) #no outliers
            #vals_centers.append([np.median(ra[neighbours[highest_ranked_neighbour_indx,:]], np.median(ra[neighbours[highest_ranked_neighbour_indx,:]]])) #best stats
            #vals_centers.append([np.mean(ra[neighbours[highest_ranked_neighbour_indx,:]], np.mean(ra[neighbours[highest_ranked_neighbour_indx,:]]])) #?
            
            for group_member in group_members:
                for ii in range(n_vals):
                    neighbours[group_member,ii] = 0
                    neighbours[ii,group_member] = 0
                    
    vals_centers.pop(0)
    vals_centers = np.array(vals_centers)
    
    
    return vals_centers
    
    ##############################################################################################
    
    
#Apply Phase Gradient
#How to use WCS with Python https://astropy4cambridge.readthedocs.io/en/latest/_static/Astropy%20-%20WCS%20Transformations.html
#http://learn.astropy.org/rst-tutorials/synthetic-images.html?highlight=filtertutorials
#https://www.atnf.csiro.au/people/Mark.Calabretta/WCS/
#https://www.atnf.csiro.au/people/mcalabre/WCS/wcslib/structwcsprm.html#aadad828f07e3affd1511e533b00da19f
#https://docs.astropy.org/en/stable/api/astropy.wcs.Wcsprm.html
#The topix conversion in CASA is done at casacore/coordinates/Coordinates/Coordinate.cc, Coordinate::toPixelWCS
#    cout << "Coordinate::toPixelWCS " << std::setprecision(20) << pixel << ",*," << wcs.crpix[0] << "," << wcs.crpix[1] << ",*," << wcs.cdelt[0] << "," << wcs.cdelt[1] << ",*," << wcs.crval[0] << "," << wcs.crval[1] << ",*," << wcs.lonpole << ",*," << wcs.latpole << ",*," << wcs.ctype[0] << "," << wcs.ctype[1] << ",*," << endl;
#cout << " world phi, theta "  << std::setprecision(20) << world << ",*,"<< phi << ",*," << theta << ",*," << endl;
#Fortran numbering issue
def make_phase_gradient(phase_dir,gcf_parms,grid_parms):
    from astropy.wcs import WCS
    rad_to_deg =  180/np.pi

    phase_center = gcf_parms['image_phase_center']
    w = WCS(naxis=2)
    w.wcs.crpix = grid_parms['image_size_padded']//2
    w.wcs.cdelt = grid_parms['cell_size']*rad_to_deg
    w.wcs.crval = phase_center*rad_to_deg
    w.wcs.ctype = ['RA---SIN','DEC--SIN']
    
    #print('phase_dir ',phase_dir)
    pix_dist = np.array(w.all_world2pix(phase_dir[0]*rad_to_deg, 1)) - grid_parms['image_size_padded']//2
    pix = -(pix_dist)*2*np.pi/(grid_parms['image_size_padded']*gcf_parms['oversampling'])
    
    image_size = gcf_parms['resize_conv_size']
    center_indx = image_size//2
    x = -np.arange(-center_indx[0], image_size[0]-center_indx[0])*math.copysign(1.0,grid_parms['cell_size'][0])
    y = np.arange(-center_indx[1], image_size[1]-center_indx[1])*math.copysign(1.0,grid_parms['cell_size'][1])
    x_grid, y_grid = np.meshgrid(x,y,indexing='ij')
    
    phase_gradient = np.moveaxis(np.exp(1j*(x_grid[:,:,None]*pix[:,0] + y_grid[:,:,None]*pix[:,1])),2,0)
    
    return phase_gradient


def _calc_ant_pointing_ra_dec(mxds,use_pointing_table,gcf_parms,sel_parms):

    vis_dataset = mxds.attrs[sel_parms['xds']]
    

    if use_pointing_table:
        ant_ra_dec = mxds.POINTING.DIRECTION.interp(time=vis_dataset.time,assume_sorted=False,method=gcf_parms['interpolation_method'])[:,:,0,:]
        ant_ra_dec = ant_ra_dec.chunk({"time":vis_dataset[sel_parms['data']].chunks[0][0]})
    else:
        antenna_ids = mxds.antenna_ids.data
        field_dataset = mxds.attrs['FIELD']
        field_id = np.max(vis_dataset.FIELD_ID,axis=1).compute() #np.max ignores int nan values (nan values are large negative numbers for int).
        n_field = field_dataset.dims['d0']
        ant_ra_dec = field_dataset.PHASE_DIR.isel(d0=field_id)
        if n_field != 1:
            ant_ra_dec = ant_ra_dec[:,0,:]
        ant_ra_dec = ant_ra_dec.expand_dims('ant',1)
        n_ant = len(antenna_ids)
        ant_ra_dec = da.tile(ant_ra_dec.data,(1,n_ant,1))
        
        time_chunksize = mxds.attrs[sel_parms['xds']][sel_parms['data']].chunks[0][0]
        ant_ra_dec =  xr.DataArray(ant_ra_dec,{'time':vis_dataset.time,'ant':antenna_ids}, dims=('time','ant','pair')).chunk({'time':time_chunksize,'ant':n_ant,'pair':2})

    return ant_ra_dec
