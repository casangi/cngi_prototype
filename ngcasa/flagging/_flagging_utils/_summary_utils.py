# C
import copy
import xarray as xr


def _cast_int(summary):
    """
    Take a dictionary and cast to int every xr.DataArray value included in it
    (assuming those arrays are return values from count(), sum() or similar xr
    aggregation functions).

    Parameters
    ----------
    summary: dict
        Dictionary including xr.DataArray values with counts (flag summaries)

    Returns:
    -------
    res: dict
        Dictionary copied from input 'summary' where all xrDataArray are casted to
        int
     """
    res = copy.deepcopy(summary)
    if isinstance(res, dict):
        for key, val in res.items():
            if isinstance(val, xr.DataArray):
                res[key] = int(val)
            else:
                _cast_int(val)
    return res

def _pol_id_to_corr_type_name(idx):
    """
    Produces a human readable name for stokes parameters, correlation products, etc. 
    from the integer IDs stored in the MS correlation type.

    Parameters
    ----------
    idx: int
        Polarization index as stored in the MS/dataset (CORR_TYPE)

    Returns:
    -------
    name: str
        String with stokes parameter or correlation product name (I, RR, YY, etc.)

    """
    # reproduce sequence in casacore/measures/Measures/Stokes.h, enum StokesTypes
    # (from which a subset is converted from enum idx to string in
    #  casa::FlagDataHandler::generatePolarizationsMap()
    # enum StokesTypes { 
    # // undefined value = 0
    # Undefined=0, 
    # // standard stokes parameters 
    # I, Q, U, V,
    # // circular correlation products
    # RR, RL, LR, LL,
    # // linear correlation products
    # XX, XY, YX, YY,
    # // mixed correlation products
    # RX, RY, LX, LY, XR, XL, YR, YL,
    # // general quasi-orthogonal correlation products
    # PP, PQ, QP, QQ,
    # // single dish polarization types
    # RCircular, LCircular, Linear,
    # // Polarized intensity ((Q^2+U^2+V^2)^(1/2))
    # Ptotal,
    # // Linearly Polarized intensity ((Q^2+U^2)^(1/2))
    # Plinear,
    # // Polarization Fraction (Ptotal/I)
    # PFtotal,
    # // Linear Polarization Fraction (Plinear/I)
    # PFlinear,
    # // Linear Polarization Angle (0.5 arctan(U/Q)) (in radians)
    # Pangle }
    corr_type_name = ['Undef', 'I', 'Q', 'U', 'V', 'RR', 'RL', 'LR', 'LL',
                      'XX', 'XY', 'YX', 'YY',
                      'RX', 'RY', 'LX', 'LY', 'XR', 'XL', 'YR', 'YL',
                      'PP', 'PQ', 'QP', 'QQ',
                      'RCircular', 'LCircular', 'Linear', 'Ptotal',
                      'Plinear', 'PFtotal', 'PFlinear', 'Pangle']
    min_idx = 1
    max_idx = len(corr_type_name)-1
    if idx < 1 or idx > max_idx:
        raise ValueError('Invalid corr type index: {}, should be in [{},{}]'.
                         format(idx, min_idx, max_idx))

    return corr_type_name[idx]

