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
"""
this module will be included in the api
"""


########################
def spec_fit(xds, dv='IMAGE', pixel=(0.5,0.5), pol=0, sigma=2000, name='FIT'):
    """
    Perform gaussian spectral line fits in the image cube
    
    Adapted from https://github.com/emilyripka/BlogRepo/blob/master/181119_PeakFitting.ipynb Dave Mehringer 2021mar01
    
    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Image Dataset
    dv : str
        name of data_var in xds to smooth. Default is 'IMAGE'
    pixel : tuple of int or float
        tuple of integer or float coordinates of pixel to fit. If int, pixel index is used. If float, nearest pixel
        at that fractional location is used. Default is (0.5,0.5) corresponding to center pixel
    pol : int
        polarization index to use.  Default is 0
    sigma : float
        sigma of gaussian fit. Default is 1000
    name : str
        dataset variable name fit, overwrites if already present
    
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image with name added fit results in attributes
    """
    import scipy.optimize
    import scipy.constants
    import scipy.signal
    import numpy as np
    import xarray

    # find pixel index values to use
    pidx = [pp if type(pp) is int else int(pp * len(xds[dv][ii])) for ii, pp in enumerate(pixel)]
    
    restf = float(xds.rest_frequency.split(' ')[0])
    vel = (-(xds.chan - restf) / restf * scipy.constants.c).values
    max_amp = xds[dv][pidx[0], pidx[1], 0, :, pol].max().values.item()

    # define a function to pass to scipy optimize for fitting
    def _1gaussian(x, amp1, cen1, sigma1):
        return amp1 * (np.exp((-1.0 / 2.0) * (((vel - cen1) / sigma1) ** 2)))

    popt_gauss, pcov_gauss = scipy.optimize.curve_fit(_1gaussian, vel, xds[dv][pidx[0],pidx[1],0,:,pol], p0=[max_amp, vel[len(vel)//2], sigma])
    perr_gauss = np.sqrt(np.diag(pcov_gauss))

    gvec = max_amp * scipy.signal.gaussian(len(xds.chan), len(vel)*popt_gauss[2]/np.ptp(vel))
    gda = xarray.DataArray(gvec, dims=['chan']).chunk(xds.chunks['chan'])

    famp = f"{popt_gauss[0]:0.2f} (+/-) {perr_gauss[0]:0.2f} Jy/beam"
    fcenter = f"{popt_gauss[1]:0.2f} (+/-) {perr_gauss[1]:0.2f} km/s"
    fsigma = f"{popt_gauss[2]:0.2f} (+/-) {perr_gauss[2]:0.2f} km/s"
    nxds = xds.assign_attrs({name+'_amplitude':famp, name+'_center':fcenter, name+'_sigma':fsigma})
    nxds = nxds.assign({name: gda}).assign_coords({'vel':xarray.DataArray(vel, dims=['chan'])})
    
    return nxds

    
    