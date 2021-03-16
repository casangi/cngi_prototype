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

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

def cont_sub(xds, dv='IMAGE', fitorder=2, chans=None, linename='LINE', contname='CONTINUUM', compute=False):
    """
    Continuum subtraction of an image cube

    Perform a polynomial baseline fit to the specified channels from an image and subtract it from all channels
    
    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        image xarray dataset
    dv : str
        name of data_var in xds to polynomal fit. Default is 'IMAGE'
    chans : array of int
        Spectral channels to use for fitting a polynomial to determine continuum. Default is None, use all channels.
    fitorder : int
        Order of polynomial to fit to the specified spectral channels to determine the continuum. Default is 2.
    linename : str
        dataset variable name for output: name of image to which to save the result of subtracting the computed continuum
        from the input image. overwrites if already present.  Default is 'LINE'
    contname : str
        dataset variable name for output：the computed continuum image. overwrites if already present.  Default is 'CONTINUUM'
    compute : bool
       execute the DAG to compute the fit error. Default False returns lazy DAG
       (error can then be retrieved via xds.<name>.<key>.values)

    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """

    import xarray as xr
    import numpy as np
    from cngi._utils import _image_utility as iu

    xr.set_options(keep_attrs=True)

    # indices of channels to use for fitting
    if chans is None: chans = np.arange(xds.dims['chan'])
    includechans = np.atleast_1d(chans)

    if not isinstance(linename, str) or linename is None:
        linename = 'LINE'

    if not isinstance(contname, str) or contname is None:
        contname = 'CONTINUUM'

    # selected channel bin values serve as our training data X
    # expanding out polynomial combinations allows us to use linear regression for non-linear higher order fits
    # see: https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions
    chans = np.arange(xds.dims['chan']).reshape(-1, 1)
    xx = PolynomialFeatures(fitorder).fit_transform(chans)

    # define a function to fit a 1-D linear regression across the prescribed axis
    # see: https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
    # with the dask='parallelized' option in apply_ufunc, this function receives a straight numpy array of chunk size
    # but does not compute the dag, which is nice
    def lspolyfit(npa):
        yy = npa.swapaxes(0, 3).reshape(len(xx), -1)  # flatten to chans by (ra * dec*time*pol) features
        yy[:, np.all(np.isnan(yy), axis=0)] = 0  # fill ra/dec/time/pol cols that are all nan with 0's
        yy_r = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(np.real(yy))  # remove remaining nan's
        model_r = LinearRegression(fit_intercept=False).fit(xx[includechans], yy_r[includechans])
        model_vals = model_r.predict(xx)  # compute model values

        return model_vals.reshape(npa.swapaxes(0, 3).shape).swapaxes(0, 3)

    #local debug lspolyfit using below line code
    #model_data = lspolyfit(xds[dv].values)

    model_data = xr.apply_ufunc(lspolyfit, xds[dv].chunk({'chan': -1}), dask='parallelized',output_dtypes=[xds[dv].dtype])
    nxds = xds.assign({linename.upper(): model_data, contname.upper(): xds[dv] - model_data}).unify_chunks()
    
    error = nxds[linename][:, :, :,includechans, :] - nxds[dv][:, :, :,includechans, :]
    abs_error = (error.real ** 2 + error.imag ** 2) ** 0.5
    rms_error = (error ** 2).mean() ** 0.5
    min_max_error = [abs_error.min(), abs_error.max()]
    bw_frac = len(includechans) / len(chans)
    freq_frac = (nxds.chan[includechans].max() - nxds.chan[includechans].min()) / (nxds.chan.max() - nxds.chan.min())

    if compute:
        rms_error = rms_error.values.item()
        min_max_error = [min_max_error[0].values.item(), min_max_error[1].values.item()]
        freq_frac = freq_frac.values.item()

    nxds.attrs[linename.lower()] = {'rms_error':rms_error, 'min_max_error':min_max_error, 'bw_frac':bw_frac, 'freq_frac':freq_frac}
    
    return nxds
