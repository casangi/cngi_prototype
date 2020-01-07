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
def smooth(ds):
    """                                                                                                                                                                                                     
    Smooth data along n-dimensions of the image cube

    .. todo::                                                                                                                                                                                               
        Verify
        
        Handle masking

        Handle polarization and channel selection

        Support more smoothing kernels


    Parameters                                                                                                                                                                                              
    ----------                                                                                                                                                                                              
    ds : xarray.core.dataset.Dataset                                                                                                                                                                        
        input Image                                                                                                                                                                                       
                                                                                                                                                                                                            
    Returns                                                                                                                                                                                                 
    -------                                                                                                                                                                                                 
    xarray.core.dataset.Dataset                                                                                                                                                                             
        output Image                                                                                                                                                                                        
    """
    import xarray as xr

    new_ds = xr.apply_ufunc(_filter_func, xds, dask='allowed',
                            keep_attrs=True)

    return new_ds

def _filter_func(da, kernel='gaussian'):
    """                                                                                                                                                                                                     
    .. warning::                                                                                                                                                                                               
        This function is still a test implementation
                                                                                                                                                                                                            
    Defines the kernel and smoothing ufunc to apply during smoothing
                                                                                                                                                                                                            
    Parameters                                                                                                                                                                                              
    ----------                                                                                                                                                                                              
    da : array_like

        input array of image data                                                                                                                                                                                         
    kernel : str

        Supported kernel to use for smoothing. See https://docs.astropy.org/en/stable/convolution/kernels.html#available-kernels
        .. note::
            Not all kernels are supported.

    Returns                                                                                                                                                                                                 
    -------                                                                                                                                                                                                 
    array_like

        output array of image data                                                                                                                                                                                        
    """
    from astropy.convolution import convolve, Gaussian2DKernel

    if kernel=='gaussian':
        pass
    else:
        print(f"kernel parameter {kernel} is not supported yet, defaulting to Gaussian")

    kernel = Gaussian2DKernel(1)

    return convolve(b.squeeze(), ker, boundary='extend')[np.newaxis,:,:]
