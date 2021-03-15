#  CASA Next Generation Infrastructure
#  Copyright (C) 2021 AUI, Inc. Washington DC, USA
#  Copyright (C) 2021 European Southern Observatory, ALMA partnership
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
    dict
        A copy of the input 'summary' dictionary where all xrDataArray have
        been casted to int
     """
    casted_summary = copy.deepcopy(summary)

    def _cast_recursion(stats):
        if isinstance(stats, dict):
            for key, val in stats.items():
                if isinstance(val, xr.DataArray):
                    stats[key] = int(val)
                else:
                    _cast_recursion(val)
        return stats

    return _cast_recursion(casted_summary)


def _pol_id_to_corr_type_name(idx):
    """
    Produces a human readable name for stokes parameters, correlation products,
    etc. from the integer IDs stored in the MS correlation type.

    Parameters
    ----------
    idx: int
        Polarization index as stored in the MS/dataset (CORR_TYPE)

    Returns:
    -------
    str
        String with stokes parameter or correlation product (I, RR, YY, etc.)

    """
    # reproduce sequence in casacore/measures/Measures/Stokes.h,
    # enum StokesTypes, from which a subset is converted from enum idx to
    # string in casa::FlagDataHandler::generatePolarizationsMap()
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
