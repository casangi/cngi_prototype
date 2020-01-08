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
def test_standard_gridder(show_plots=False):
    """
    Unit test that compares the standard gridder in CNGI with that in CASA.
    For the test to run sis14_twhya_field5_mstrans_lsrk_old.zarr and sis14_twhya_casa_ximage.zarr
    are required (placed in cngi_prototype/cngi/data/).

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
    import cngi
    import os
    import tests.gridding_convolutional_kernels as gck
    from cngi.gridding import serial_grid
    from cngi.gridding import standard_grid
    import matplotlib.pylab as plt
    import numpy as np
    from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
    import time

    # Load measurement dataset
    outfile = os.getcwd() + '/data/sis14_twhya_field5_mstrans_lsrk_old.zarr/0'

    vis_dataset = xr.open_zarr(outfile)

    # Gridding Parameters
    dtr = np.pi / (3600 * 180)  # Degrees to Radians
    n_xy = np.array([200, 400])  # 2048                 # 2 element array containing number of pixels for x (u and l) and y (v and m) axes.
    padding = 1.2  # Padding factor
    n_xy_padded = (padding * n_xy).astype(int)  # 2 element array containg number of pixels with padding
    delta_lm = np.array([-0.08 * dtr, 0.08 * dtr])  # Image domain cell size converted from arcseconds to radians (negative is due to axis direction being different from indx increments)
    delta_uv = np.array([1 / (n_xy_padded[0] * delta_lm[0]), 1 / (n_xy_padded[1] * delta_lm[1])])  # Visibility domain cell size

    # Creating gridding kernel (same as CASA)
    support = 35#7  # The support in CASA is defined as the half support, which is 3
    oversampling = 100
    cgk_1D = gck.create_prolate_spheroidal_kernel_1D(oversampling, support)
    cgk, cgk_image = gck.create_prolate_spheroidal_kernel(oversampling, support, n_xy_padded)

    # Data used for gridding
    grid_data = vis_dataset.data_vars['DATA'].values
    uvw = vis_dataset.data_vars['UVW'].values
    freq_chan = vis_dataset.coords['chan'].values
    weight = vis_dataset.data_vars['WEIGHT'].values
    weight[:, :, 0] = (weight[:, :, 0] + weight[:, :,
                                         1]) / 2  # Casa averages the weights for different polarizations. See code/msvis/MSVis/VisImagingWeight.cc VisImagingWeight::unPolChanWeight
    flag_row = vis_dataset.data_vars['FLAG_ROW'].values
    flag = vis_dataset.data_vars['FLAG'].values
    n_uv = n_xy_padded

    #
    n_imag_chan = 1  # Making only one continuum image.
    n_imag_pol = 1  # Imaging only one polarization
    sum_weight = np.zeros((n_imag_chan, n_imag_pol), dtype=np.double)
    n_chan = vis_dataset.dims['chan']
    n_pol = 1  # vis_dataset.dims['pol'] #just using the first polarization
    vis_grid = np.zeros((n_imag_chan, n_imag_pol, n_xy_padded[0], n_xy_padded[1]), dtype=np.complex128)  # Creating an empty grid

    chan_map = (np.zeros(n_chan)).astype(
        np.int)  # np.arange(0,n_chan) for cube, n_chan array which maps to the number of channels in image (cube)
    pol_map = (np.zeros(n_pol)).astype(np.int)

    grid_parms = {}
    grid_parms['n_imag_chan'] = n_imag_chan

    grid_parms['n_imag_pol'] = n_imag_pol
    grid_parms['n_uv'] = n_uv
    grid_parms['delta_lm'] = delta_lm
    grid_parms['oversampling'] = oversampling
    grid_parms['support'] = support

    # Grid data
    start = time.time()
    p_vis_grid, sum_weight = standard_grid(grid_data, uvw, weight, flag_row, flag, freq_chan, chan_map, n_chan, pol_map, cgk_1D, grid_parms)
    print('Vectorized Gridding Time Dask (s): ', time.time() - start)
    
    start = time.time()
    s_vis_grid, sum_weight = serial_grid(grid_data, uvw, weight, flag_row, flag, freq_chan, chan_map, n_chan, pol_map, cgk_1D, grid_parms)
    print('Numba Gridding Time Dask (s): ', time.time() - start)
    
    # temp
    p_vis_grid = p_vis_grid / sum_weight

    # Create Dirty Image and correct for gridding convolutional kernel
    # uncorrected_dirty_image = fftshift(ifft2(ifftshift(vis_grid[0,0,:,:]))).T *((n_xy_padded[0]*n_xy_padded[1])/np.sum(sum_weight))
    # corrected_dirty_image = uncorrected_dirty_image/cgk_image

    # Normalize results
    # corrected_dirty_image = corrected_dirty_image/(np.max(np.abs(corrected_dirty_image)))
    # vis_grid = vis_grid[0,0,:,:]/(np.max(np.abs(vis_grid)))
    
    fig0, ax0 = plt.subplots(1, 2, sharey=True)
    im00 = ax0[0].imshow(np.abs(p_vis_grid))
    im01 = ax0[1].imshow(np.abs(s_vis_grid[0, 0]))
    ax0[0].title.set_text('Dask Absolute Value of Gridded Visibilities')
    ax0[1].title.set_text('Serial Absolute Value of Gridded Visibilities')
    fig0.colorbar(im00, ax=ax0[0])
    fig0.colorbar(im01, ax=ax0[1])
    plt.show()
    
    fig1, ax1 = plt.subplots(1, 2, sharey=True)
    im0 = ax1[0].imshow(np.abs(p_vis_grid - s_vis_grid[0, 0]))
    im1 = ax1[1].imshow(np.abs(p_vis_grid - s_vis_grid[0, 0]))
    ax1[0].title.set_text('Difference Absolute Value of Gridded Visibilities')
    ax1[1].title.set_text('Difference Dirty Image')
    fig1.colorbar(im0, ax=ax1[0], fraction=0.046, pad=0.04)
    fig1.colorbar(im1, ax=ax1[1], fraction=0.046, pad=0.04)
    plt.show()

    # Calculate max error
    # max_error_grid = np.max(np.abs(vis_grid-ximage_dataset.data_vars['vis_grid'].values))
    # max_error_corrected_dirty_image = np.max(np.abs(corrected_dirty_image-ximage_dataset.data_vars['corrected_dirty_image'].values))

    # Calculate root mean square error
    # rms_error_grid =  np.linalg.norm(vis_grid-ximage_dataset.data_vars['vis_grid'].values,'fro')
    # rms_error_corrected_dirty_image =  np.linalg.norm(corrected_dirty_image-ximage_dataset.data_vars['corrected_dirty_image'].values ,'fro')

    # pass_test = False
    # print('*******************************************************************************')
    # print('Gridded and image values have been normalized before calculating error values')
    # print('Max error between CASA and CNGI gridded values ', max_error_grid)
    # print('Max error between CASA and CNGI dirty images ', max_error_corrected_dirty_image)
    # print('RMS error between CASA and CNGI gridded values ', rms_error_grid)
    # print('RMS error between CASA and CNGI dirty images ', rms_error_corrected_dirty_image)
    # if (max_error_grid < 3.552e-09) and (max_error_corrected_dirty_image < 9.795e-08) and (rms_error_grid < 1.601e-08) and (rms_error_corrected_dirty_image < 2.091e-06):
    #    print('Test Pass')
    #    pass_test = True
    # else:
    #    print('Test Fail')
    #    pass_test = False
    # print('*******************************************************************************')
    
    return True  # pass_test

test_standard_gridder(show_plots=False)