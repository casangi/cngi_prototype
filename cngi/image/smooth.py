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
def smooth(xds, dv='IMAGE', kernel='gaussian', size=(1., 1., 30.), ktype='target', current=None, scale=1.0, name='BEAM'):
    """                                                                                                                                                                                                     
    Smooth data along the spatial plane of the image cube

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
    ktype : str
        type of beam kernel being defined, either 'target' or 'correcting'.  Default is 'target'
    current : str
        name of data_var defining the current beam.  Default is None
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
    if kernel in xds.data_vars:
        beam = xds[kernel]
    elif kernel == 'gaussian':
        beam = chb.synthesizedbeam(size[0], size[1], size[2], len(xds.d0), len(xds.d1), xds.incr[:2])
        beam = xarray.DataArray(da.from_array(beam), dims=['d0', 'd1'], name=name)
    else:  # boxcar
        incr = np.abs(xds.incr[:2]) * 180 / np.pi * 60 * 60
        xx, yy = np.mgrid[:int(np.round(size[0] / incr[0])), :int(np.round(size[1] / incr[1]))]
        box = np.array((xx.ravel() - np.max(xx) // 2, yy.ravel() - np.max(yy) // 2)) + np.array([len(xds.d0) // 2, len(xds.d1) // 2])[:, None]
        beam = np.zeros((len(xds.d0), len(xds.d1)))
        beam[box[0], box[1]] = 1.0
        beam = xarray.DataArray(da.from_array(beam), dims=['d0', 'd1'], name=name)

    # scale and FFT the kernel beam
    da_beam = da.atleast_3d(da.from_array(beam))
    if da_beam.ndim < 4: da_beam = da_beam[:,:,:,None]
    beam_norm = da.sum(da_beam, axis=[0,1])
    beam = beam * scale
    ft_beam = da.fft.fft2(da_beam*scale, axes=[0, 1]) / beam_norm

    # compute the correcting beam if necessary
    if (kernel not in xds.data_vars) and (ktype == 'target') and (current is not None):
        beam_curr = xds[current]
        da_beam_curr = da.atleast_3d(da.from_array(beam_curr))
        if da_beam_curr.ndim < 4: da_beam_curr = da_beam_curr[:, :, :, None]
        beam_curr_norm = da.sum(da_beam_curr, axis=[0, 1])
        ft_beam_curr = da.fft.fft2(da_beam_curr, axes=[0, 1]) / beam_curr_norm
        ft_beam = ft_beam / ft_beam_curr
        beam = da.absolute(da.fft.fftshift(da.fft.ifft2(ft_beam, axes=[0,1]), axes=[0,1]))
        beam = xarray.DataArray(beam, dims=[xds[dv].dims[dd] for dd in range(beam.ndim)], name=name)
    
    # FFT the image, multiply by the kernel beam FFT, then inverse FFT it back
    xda_image = da.from_array(xds[dv], chunks=(-1,-1,1,1))
    ft_image = da.fft.fft2(xda_image, axes=[0, 1])
    ft_smooth = ft_image * ft_beam
    ift_smooth = da.fft.ifft2(ft_smooth, axes=[0,1])
    if (kernel in xds.data_vars) or (ktype == 'correcting') or (current is None):
        ift_smooth = da.fft.fftshift(ift_smooth, axes=[0,1])
    
    # store the smooth image and kernel beam back in the xds
    xda_smooth = xarray.DataArray(da.absolute(ift_smooth), dims=xds[dv].dims, coords=xds[dv].coords)
    new_xds = xds.assign({dv: xda_smooth, name: beam})
    return new_xds

