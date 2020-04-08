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

########################                                                                                                                                                                                    
def smooth(xds, dv='IMAGE', kernel='gaussian', size=(1., 1., 30.), current=None, scale=1.0, name='BEAM'):
    """                                                                                                                                                                                                     
    Smooth data along the spatial plane of the image cube.
    
    Computes a correcting beam to produce defined size when kernel=gaussia and current is defined.  Otherwise the size
    or existing beam is used directly.

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    dv : str
        name of data_var in xds to smooth. None will smooth all. Default is 'IMAGE'
    kernel : str
        Type of kernel to use:'boxcar', 'gaussian' or the name of a data var in this xds.  Default is 'gaussian'.
    size : tuple of floats
        tuple of three values corresponding to major and minor axes (in arcseconds) and position angle (in degrees).
    current : tuple of floats
        same structure as size, a tuple of three values corresponding to major and minor axes (in arcseconds) and position
        angle (in degrees) of the current beam applied to the image.  Default is None
    scale : float
        gain factor after convolution. Default is unity gain (1.0)
    name : str
        dataset variable name for kernel, overwrites if already present
        
    Returns                                                                                                                                                                                                 
    -------                                                                                                                                                                                                 
    xarray.core.dataset.Dataset                                                                                                                                                                             
        output Image                                                                                                                                                                                        
    """
    import xarray
    import dask.array as da
    import numpy as np
    import cngi._helper.beams as chb

    # compute kernel beam
    size_corr = None
    if kernel in xds.data_vars:
        beam = xds[kernel]
    elif kernel == 'gaussian':
        beam, parms_tar = chb.synthesizedbeam(size[0], size[1], size[2], len(xds.d0), len(xds.d1), xds.incr[:2])
        beam = xarray.DataArray(da.from_array(beam), dims=['d0', 'd1'], name=name)
        cf_tar = ((4 * np.pi ** 2) / (4 * parms_tar[0] * parms_tar[2] - parms_tar[1] ** 2)) * parms_tar  # equation 12
        size_corr = size
    else:  # boxcar
        incr = np.abs(xds.incr[:2]) * 180 / np.pi * 60 * 60
        xx, yy = np.mgrid[:int(np.round(size[0] / incr[0])), :int(np.round(size[1] / incr[1]))]
        box = np.array((xx.ravel() - np.max(xx) // 2, yy.ravel() - np.max(yy) // 2)) + np.array([len(xds.d0) // 2, len(xds.d1) // 2])[:, None]
        beam = np.zeros((len(xds.d0), len(xds.d1)))
        beam[box[0], box[1]] = 1.0
        beam = xarray.DataArray(da.from_array(beam), dims=['d0', 'd1'], name=name)
    
    # compute the correcting beam if necessary
    # this is done analytically using the parameters of the current beam, not the actual data
    # see equations 19 - 26 here:
    # https://casa.nrao.edu/casadocs-devel/stable/memo-series/casa-memos/casa_memo10_restoringbeam.pdf/view
    if (kernel == 'gaussian') and  (current is not None):
        parms_curr = chb.synthesizedbeam(current[0], current[1], current[2], len(xds.d0), len(xds.d1), xds.incr[:2])[1]
        cf_curr = ((4*np.pi**2) / (4*parms_curr[0]*parms_curr[2] - parms_curr[1]**2)) * parms_curr  # equation 12
        cf_corr = (cf_tar - cf_curr)  # equation 19
        c_corr = ((4*np.pi**2) / (4*cf_corr[0]*cf_corr[2] - cf_corr[1]**2)) * cf_corr  # equation 12
        # equations 21 - 23
        d1 = np.sqrt( 8*np.log(2) / ((c_corr[0] + c_corr[2]) - np.sqrt(c_corr[0]**2 - 2*c_corr[0]*c_corr[2] + c_corr[2]**2 + c_corr[1]**2)))
        d2 = np.sqrt( 8*np.log(2) / ((c_corr[0] + c_corr[2]) + np.sqrt(c_corr[0]**2 - 2*c_corr[0]*c_corr[2] + c_corr[2]**2 + c_corr[1]**2)))
        theta = 0.5*np.arctan2(-c_corr[1], c_corr[2] - c_corr[0])
        
        # make a beam out of the correcting size
        incr_arcsec = np.abs(xds.incr[:2]) * 180 / np.pi * 60 * 60
        size_corr = [d1 * incr_arcsec[0], d2 * incr_arcsec[1], theta * 180 / np.pi]
        scale_corr = 1.0 # not needed, rescaled below # (4*np.log(2) / (np.pi*d1*d2)) * (size[0]*size[1]/(current[0]*current[1])) # equation 20
        beam = scale_corr * chb.synthesizedbeam(size_corr[0], size_corr[1], size_corr[2], len(xds.d0), len(xds.d1), xds.incr[:2])[0]
        beam = xarray.DataArray(da.from_array(beam), dims=[xds[dv].dims[dd] for dd in range(beam.ndim)], name=name)
    
    # scale and FFT the kernel beam
    da_beam = da.atleast_3d(da.from_array(beam))
    if da_beam.ndim < 4: da_beam = da_beam[:, :, :, None]
    beam_norm = da.sum(da_beam, axis=[0, 1])
    beam = beam * scale
    ft_beam = da.fft.fft2(da_beam * scale, axes=[0, 1]) / beam_norm

    # FFT the image, multiply by the kernel beam FFT, then inverse FFT it back
    xda_image = da.from_array(xds[dv], chunks=(-1,-1,1,1))
    ft_image = da.fft.fft2(xda_image, axes=[0, 1])
    ft_smooth = ft_image * ft_beam
    ift_smooth = da.fft.fftshift(da.fft.ifft2(ft_smooth, axes=[0,1]), axes=[0,1])
    
    # store the smooth image and kernel beam back in the xds
    xda_smooth = xarray.DataArray(da.absolute(ift_smooth), dims=xds[dv].dims, coords=xds[dv].coords)
    new_xds = xds.assign({dv: xda_smooth, name: beam})
    if size_corr is not None:
        new_xds = new_xds.assign_attrs({name+'_params':tuple(size_corr)})
    return new_xds

