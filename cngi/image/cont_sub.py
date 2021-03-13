#   Copyright 2020 AUI, Inc. Washington DC, USA
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
"""
this module will be included in the api
"""

########################

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

def cont_sub(xds, dv='IMAGE', fitOrder=2 , chans=None, polyfitCoefficiencs='CoefficiencesPoly', linename='LinePoly', continuumname = 'ContinuumPoly'):
    """
    Continuum subtraction of an image cube

    Perform a polynomial baseline fit to the specified channels from an image and subtract it from all channels
    
    .. note::
        chans = "" for all channels
        chans = "3~8,10~15"

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        image xarray dataset
    dv : str
        name of data_var in xds to polynomal fit. Default is 'IMAGE'
    chans : str
        Spectral channels to use for fitting a polynomial to determine continuum. Default is '', use all channels.
    fitOrder : int
        Order of polynomial to fit to the specified spectral channels to determine the continuum. Default is 2.
    polyfitCoefficiencs : str
        dataset attribute name for output: the polynomial fitting coffeiciences. Default is ''
    linename : str
        dataset variable name for output: name of image to which to save the result of subtracting the computed continuum from the input image. overwrites if already present.  Default is 'Line'
    continuumname : str
        dataset variable name for output：the computed continuum image. overwrites if already present.  Default is 'Continuum'

    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """

    import xarray as xr
    import dask.array as da
    from cngi._utils import _image_utility as iu

    xr.set_options(keep_attrs=True)

    if (xds.dims['pol'] > 1 ):
        raise NotImplementedError("The number of pol great than one is not yet supported")

    if chans is None:
       includechans = list(range(xds.dims['chan']))
    else:
       includechans = iu.selectedchannels(chans, xds.dims['chan'])

    if fitOrder is None:
       fitOrder = 2

    if not isinstance(linename, str) or linename is None:
       linename = 'Line'

    if not isinstance(continuumname, str) or continuumname is None:
        continuumname = 'Continuum'

    # selected channel bin values serve as our training data X
    # expanding out polynomial combinations allows us to use linear regression for non-linear higher order fits
    # see: https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions
    chans = np.arange(xds.dims['chan']).reshape(-1, 1)
    xx = PolynomialFeatures(fitOrder).fit_transform(chans)

    # indices of channels to use for fitting
    includechans = np.atleast_1d(includechans)

    # define a function to fit a 1-D linear regression across the prescribed axis
    # see: https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
    # with the dask='parallelized' option in apply_ufunc, this function receives a straight numpy array of chunk size
    # but does not compute the dag, which is nice
    def lspolyfit(npa):
        yy = npa.swapaxes(0, 3).reshape(len(xx), -1)  # flatten to chans by (ra * dec*time*pol) features
        yy[:, np.all(np.isnan(yy), axis=0)] = 0  # fill ra/dec/time/pol cols that are all nan with 0's
        yy_r = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(
            np.real(yy))  # remove remaining nan's
        model_r = LinearRegression(fit_intercept=False).fit(xx[includechans], yy_r[includechans])
        model_vals = model_r.predict(xx)  # compute model values

        return model_vals.reshape(npa.swapaxes(0, 3).shape).swapaxes(0, 3)

    #local debug lspolyfit using below line code
    #model_data = lspolyfit(xds[dv].values)

    model_data = xr.apply_ufunc(lspolyfit, xds[dv].chunk({'chan': -1}), dask='parallelized',output_dtypes=[xds[dv].dtype])
    xds[linename] = model_data
    xds[continuumname] = xds[dv] - model_data
    xds = xds.assign({linename: model_data}).unify_chunks()

    error = xds[linename][:, :, :,includechans, :] - xds[dv][:, :, :,includechans, :]
    abs_error = (error.real ** 2 + error.imag ** 2) ** 0.5
    rms_error = (error ** 2).mean() ** 0.5
    min_max_error = [abs_error.min(), abs_error.max()]
    bw_frac = len(includechans) / len(chans)
    freq_frac = (xds.chan[includechans].max() - xds.chan[includechans].min()) / (xds.chan.max() - xds.chan.min())

    xds.attrs[linename + '_rms_error'] = rms_error
    xds.attrs[linename + '_min_max_error'] = min_max_error
    xds.attrs[linename + '_bw_frac'] = bw_frac
    xds.attrs[linename + '_freq_frac'] = freq_frac
    
    return xds

