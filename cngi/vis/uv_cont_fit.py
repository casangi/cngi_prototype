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

##################################################
def uv_cont_fit(mxds, vis, source='DATA', target='CONTFIT', fitorder=1, excludechans=[]):
    """
    Fit a polynomial regression to source data and return model values to target

    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        input multi-xarray Dataset with global data
    vis : str
        visibility partition in the mxds to use
    source : str
        data variable in the dataset on which to fit the regression. Default is 'DATA'
    target : str
        new data variable to place the fit result, overwrites if already present. Default is 'CONTFIT'
    fitorder : int
        polynomial order for the fit, must be >= 1, but values larger than 1 will slow down rapidly.  Default is 1
    excludechans : list of ints
        indices of channels to exclude from the fit.  Default is empty (include all channels)
    
    Returns
    -------
    xarray.core.dataset.Dataset
        New output multi-xarray Dataset with global data
    """
    import numpy as np
    import xarray
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.impute import SimpleImputer
    from cngi._utils._io import mxds_copier

    xds = mxds.attrs[vis]
    
    # selected channel bin values serve as our training data X
    # expanding out polynomial combinations allows us to use linear regression for non-linear higher order fits
    # see: https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions
    chans = np.arange(xds.dims['chan']). reshape(-1,1)
    xx = PolynomialFeatures(fitorder).fit_transform(chans)
    
    # indices of channels to use for fitting
    includechans = np.setdiff1d(range(len(chans)), np.atleast_1d(excludechans))
    
    # define a function to fit a 1-D linear regression across the prescribed axis
    # see: https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
    # with the dask='parallelized' option in apply_ufunc, this function receives a straight numpy array of chunk size
    # but does not compute the dag, which is nice
    def lr1d(npa):
        #npa = xds.DATA[:100,:210,:,:1].values #.swapaxes(2,3)
        yy = npa.swapaxes(0,2).reshape(len(xx), -1)     # flatten to chans by (time * baseline * pol) features
        yy[:,np.all(np.isnan(yy), axis=0)] = 0          # fill baseline/time/pol cols that are all nan with 0's
        yy_r = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(np.real(yy))  # remove remaining nan's
        model_r = LinearRegression(fit_intercept=False).fit(xx[includechans], yy_r[includechans])
        model_vals = model_r.predict(xx)  # compute model values
        if npa.dtype == 'complex128':
            yy_i = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(np.imag(yy))
            model_i = LinearRegression(fit_intercept=False).fit(xx[includechans], yy_i[includechans])
            model_vals = model_vals + 1j*model_i.predict(xx)  # compute model values
        return model_vals.reshape(npa.swapaxes(0,2).shape).swapaxes(0,2)
    
    model_data = xarray.apply_ufunc(lr1d, xds[source].chunk({'chan':-1}), dask='parallelized', output_dtypes=[xds[source].dtype])
    
    new_xds = xds.assign({target: model_data}).unify_chunks()

    # compute some fit metrics to store in attributes section
    error = new_xds[target][:,:,includechans,:] - new_xds[source][:,:,includechans,:]
    abs_error = (error.real ** 2 + error.imag ** 2) ** 0.5
    rms_error = (error**2).mean()**0.5
    min_max_error = [abs_error.min(), abs_error.max()]
    bw_frac = len(includechans) / len(chans)
    freq_frac = (xds.chan[includechans].max() - xds.chan[includechans].min()) / (xds.chan.max()-xds.chan.min())
    
    new_xds = new_xds.assign_attrs({target+'_rms_error':rms_error,
                                    target+'_min_max_error':min_max_error,
                                    target+'_bw_frac':bw_frac,
                                    target+'_freq_frac':freq_frac})
    
    return mxds_copier(mxds, vis, new_xds)
