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


##########################################
def test_dirty_image(show_plots=False):
    """
    Unit test that compares the standard gridder in CNGI with that in CASA.
    For the test to run sis14_twhya_field5_mstrans_lsrk_old.zarr and sis14_twhya_casa_ximage.zarr
    are required (placed in cngi_prototype/data/).

    Parameters
    ----------
    show_plots : bool
        If true plots are shown of CASA and CNGI gridded visibilities and dirty images.
    Returns
    -------
    pass_test : bool
        Is True if CNGI values are close enough to CASA values.
    """
    
    import xarray as xr
    import os
    import matplotlib.pylab as plt
    import numpy as np
    from cngi.gridding import dirty_image
    


    # Load measurement dataset
    #outfile = os.getcwd() + '/data/sis14_twhya_field5_mstrans_lsrk.vis.zarr/0'
    outfile = os.getcwd() + '/data/sis14_twhya_field5_mstrans_lsrk.vis.zarr/0'
    vis_xds = xr.open_zarr(outfile)
    
    #weights = (vis_xds.WEIGHT.values[:, :, 0] + vis_xds.WEIGHT.values[:, :, 1])/2 # Casa averages the weights for different polarizations. See code/msvis/MSVis/VisImagingWeight.cc VisImagingWeight::unPolChanWeight
    weight = vis_xds.data_vars['WEIGHT']
    weight_avg = xr.DataArray.expand_dims((weight[:, :, 0] + weight[:, :, 1]) / 2, dim=['n_pol'], axis=2)
    vis_xds = vis_xds.isel(pol=[0]) 
    vis_xds['WEIGHT'] = weight_avg
    
    grid_parms = {}
    grid_parms['chan_mode'] = 'continuum'#cube or continuum
    grid_parms['imsize'] =  [200,400]
    grid_parms['cell'] = [0.08, 0.08]
    grid_parms['oversampling'] = 100
    grid_parms['support'] = 7
    grid_parms['to_disk'] =  False
    #grid_parms['outfile'] = 'gridded_vis_cube.zarr' #'gridded_vis_cont.zarr' # 'gridded_vis_cube.zarr'


    img_xds = dirty_image(vis_xds,grid_parms)
    
    n_xy = (np.array(grid_parms['imsize'])).astype(np.int)
    n_xy_padded = (1.2*n_xy).astype(int)
    start_xy = (n_xy_padded // 2 - n_xy // 2)
    end_xy = start_xy + n_xy
    corrected_dirty_image = (img_xds.DIRTY_IMAGE[start_xy[0]:end_xy[0], start_xy[1]:end_xy[1],0,0])
    
    # Load CASA data
    outfile = os.getcwd() + '/data/sis14_twhya_field5_mstrans_lsrk_ximage.zarr'
    casa_img_xds = xr.open_zarr(outfile)
    casa_corrected_dirty_image = casa_img_xds['residual'].values[:, :, 0, 0]
    
    if show_plots == True:
        fig0, ax0 = plt.subplots(1, 2, sharey=True)
        im0 = ax0[0].imshow(casa_corrected_dirty_image)
        im1 = ax0[1].imshow(corrected_dirty_image)
        ax0[0].title.set_text('CASA Dirty Image')
        ax0[1].title.set_text('CNGI Dirty Image')
        fig0.colorbar(im0, ax=ax0[0], fraction=0.046, pad=0.04)
        fig0.colorbar(im1, ax=ax0[1], fraction=0.046, pad=0.04)
        plt.show()
        
        plt.figure()
        plt.imshow(casa_corrected_dirty_image - corrected_dirty_image)
        plt.title('Difference Dirty Image')
        plt.colorbar()
        plt.show()
    
    corrected_dirty_image = corrected_dirty_image / np.max(np.abs(corrected_dirty_image))
    casa_corrected_dirty_image = casa_corrected_dirty_image / np.max(np.abs(casa_corrected_dirty_image))

    # Calculate max error
    max_error_corrected_dirty_image = np.max(np.abs(corrected_dirty_image - casa_corrected_dirty_image))

    # Calculate root mean square error
    rms_error_corrected_dirty_image = np.linalg.norm(corrected_dirty_image - casa_corrected_dirty_image, 'fro')
    
    pass_test = False
    print('*******************************************************************************')
    print('Gridded and image values have been normalized before calculating error values')
    print('Max error between CASA and CNGI dirty images ', max_error_corrected_dirty_image)
    print('RMS error between CASA and CNGI dirty images ', rms_error_corrected_dirty_image)
    if (max_error_corrected_dirty_image < 1.2886e-07) and (rms_error_corrected_dirty_image < 1.616e-06):
        print('Test Pass')
        pass_test = True
    else:
        print('Test Fail')
        pass_test = False
    print('*******************************************************************************')
    
    return pass_test

if __name__ == '__main__':
    test_dirty_image(show_plots=True)
    


    
    
    
    
    
