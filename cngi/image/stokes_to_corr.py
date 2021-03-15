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

def stokes_to_corr(ixds, basis="linear"):
    """Convert polarization data from Stokes parameters to the correlation basis.
    
     To be used as a converter during image reconstruction, and also stand-alone. 
     
     The sign convention used here should be the same as what CASA uses, (XX^YY)/2
     
     ..todo:
       Apply transformations to all data variables, not just IMAGE

     ..todo:
       Drop the basis and code_name dimensions from the product

     ..todo:
       Update pol dimension to match reassigned version across the rest of the xds when combining

     ..todo:
       Ensure that math to calculate linear basis products matches the CASA convention

    Parameters
    ----------
    ixds : xarray.core.dataset.Dataset
        Input image dataset (e.g., loaded from img.zarr file) with polarization data as Stokes (I,Q,U,V) parameters.
    basis : string, default='linear'
        Desired correlation basis 'linear' or 'circular'

    Returns
    -------
    xarray.core.dataset.Dataset
        Output image dataset with polarization data in the linear (XX,XY,YX,YY) or circular (RR,RL,LR,LL) correlation basis.

    See Also
    --------
    corr_to_stokes

    Notes
    -----
    Polarization codes from the MeasurementSet are preserved in vis.zarr:
    #. I
    #. Q
    #. U
    #. V
    #. RR
    #. RL
    #. LR
    #. LL
    #. XX
    #. XY
    #. YX
    #. YY
   
    Raises
    ______
    UserWarning
        If input pol dimension does not contain expected codes, or has the wrong shape.

    NotImplementedError
        If the input image has less than 4 Stokes parameters we still compute the possible results but cannot return them, so trap a ValueError with this until it is decided how to align pol dimension in the converted image with the pol dimension for all the other data variables of the input. Is it necessary to update the codes along the pol dimension for the whole dataset to match new basis?

    .. note::
        Support is presently limited for heterogeneous-feed arrays with elements expected to be missing in a given basis (e.g., very long baseline interferometry with the Event Horizon Telescope).
    """
    import xarray
    import numpy as np

    # unfortunately, the codes are not zero-indexed :(
    # but we can define a new dataset to keep track :)
    pol_codes = xarray.DataArray(
        data=np.arange(start=1, stop=13, step=1, dtype=float).reshape(3, 4),
        dims=["basis", "code_index"],
        coords={
            "basis": ["stokes", "circular", "linear"],
            "code_index": np.arange(start=0, stop=4, step=1, dtype=int),
            "code_name": (
                ("basis", "code_index"),
                [
                    ["I", "Q", "U", "V"],
                    ["RR", "RL", "LR", "LL"],
                    ["XX", "XY", "YX", "YY"],
                ],
            ),
        },
    )

    # from numpy.distutils.misc_util import is_sequence ?
    try:
        # first ensure expected input -- we have at least one Stokes parameter
        # normally this will mean that we don't have other codes (but we might)
        if ixds["pol"] in pol_codes.sel(basis="stokes"):
            # construct a temp object (view) for safety
            _ixds = ixds
            # next determine desired output basis
            if basis.upper().startswith("L"):
                # check size and convert all present parameters to LINEAR
                if ixds["pol"].size >= 2:
                    if ixds["pol"].size == 4:
                        # all correlations can be solved
                        # XX = I + Q
                        # XY = U + iV
                        # YX = U - iV
                        # YY = I - Q
                        XX = ixds["IMAGE"].sel(pol=1) + ixds["IMAGE"].sel(pol=2)
                        XY = ixds["IMAGE"].sel(pol=3) + ixds["IMAGE"].sel(pol=4).imag
                        YX = ixds["IMAGE"].sel(pol=3) - ixds["IMAGE"].sel(pol=4).imag
                        YY = ixds["IMAGE"].sel(pol=1) - ixds["IMAGE"].sel(pol=2)
                        # store results in new object with updated codes
                        NEW_IMAGE = xarray.concat([XX, XY, YX, YY], dim="pol")
                        NEW_IMAGE.assign_coords(
                            {
                                "pol": pol_codes.sel(basis="linear", drop=True).rename(
                                    {"code_index": "pol"}
                                )
                            }
                        )
                        # Update the codes on pol dimension of all data variables in temp dataset
                        _ixds = _ixds.assign_coords(
                            pol=pol_codes.sel(basis="linear", drop=True)
                            .rename({"code_index": "pol"})
                            .data
                        )
                    # without full Stokes parameterization, only 2/4 correlations can be solved
                    # check for IQ
                    elif [True, True] in (
                        pol_codes.sel(basis="stokes", code_index=[0, 1]).isin(
                            ixds["pol"].data
                        )
                    ):
                        print(
                            "Stokes I and Q are present in input but either U or V is absent"
                        )
                        print("Solving only for XX and YY")
                        # XX = I + Q
                        # YY = I - Q
                        XX = ixds["IMAGE"].sel(pol=1) + ixds["IMAGE"].sel(pol=2)
                        YY = ixds["IMAGE"].sel(pol=1) - ixds["IMAGE"].sel(pol=2)
                        # store results in new object with updated codes
                        NEW_IMAGE = xarray.concat([XX, YY], dim="pol")
                        # reassignment is necessary to preserve coord vals (vs. passing eval expression directly as in the ==4 case)
                        sub_codes = pol_codes.sel(
                            basis="linear", code_index=[0, 3]
                        ).data
                        NEW_IMAGE.assign_coords({"pol": sub_codes})
                        # Update the codes on pol dimension of all data variables in temp dataset
                        _ixds = (
                            _ixds.assign_coords(pol=sub_codes, drop=True)
                            .rename({"code_index": "pol"})
                            .data
                        )
                    # check for UV
                    elif [True, True] in (
                        pol_codes.sel(basis="stokes", code_index=[2, 3]).isin(
                            ixds["pol"].data
                        )
                    ):
                        print(
                            "Stokes U and V are present in input but either I or Q is absent"
                        )
                        print("Solving only for XY and YX")
                        # XY = U + iV
                        # YX = U - iV
                        XY = ixds["IMAGE"].sel(pol=3) + ixds["IMAGE"].sel(pol=4).imag
                        YX = ixds["IMAGE"].sel(pol=3) - ixds["IMAGE"].sel(pol=4).imag
                        # store results in new object with updated codes
                        NEW_IMAGE = xarray.concat([XX, YY], dim="pol")
                        # reassignment is necessary to preserve coord vals (vs. passing eval expression directly as in the ==4 case)
                        sub_codes = pol_codes.sel(
                            basis="linear", code_index=[1, 2]
                        ).data
                        NEW_IMAGE.assign_coords({"pol": sub_codes})
                        # Update the codes on pol dimension of all data variables in temp dataset
                        _ixds = (
                            _ixds.assign_coords(pol=sub_codes, drop=True)
                            .rename({"code_index": "pol"})
                            .data
                        )
                    try:
                        # After conditionals, add the output to this properly-coded temp object for later return
                        # At present this will only work for the full-polarization case!
                        _ixds.assign(
                            {
                                "IMAGE": NEW_IMAGE.transpose(
                                    "l", "m", "time", "chan", "pol"
                                )
                            }
                        )
                    except ValueError as E:
                        # arguments without labels along dimension 'pol' cannot be aligned because they have different dimension size(s)
                        # NEW_IMAGE has updated pol_codes with size == 2, but all other images in O.G. dataset have old codes
                        print(E.args)
                        raise NotImplementedError(
                            "How do we combine NEW_IMAGE with original dataset after transforming pol dimension?"
                        )
                else:
                    # Cannot form a linear correlation from a single Stokes parameter
                    raise UserWarning(
                        "Unexpected pol dimension shape, check input dataset"
                    )
            elif basis.upper().startswith("C"):
                # check size and convert all present parameters to CIRCULAR
                if ixds["pol"].size >= 2:
                    if ixds["pol"].size == 4:
                        # all correlations can be solved
                        # RR = I + V
                        # RL = Q + iU
                        # LR = Q - iU
                        # LL = I - V
                        RR = ixds["IMAGE"].sel(pol=1) + ixds["IMAGE"].sel(pol=4)
                        RL = ixds["IMAGE"].sel(pol=2) + ixds["IMAGE"].sel(pol=3).imag
                        LR = ixds["IMAGE"].sel(pol=2) - ixds["IMAGE"].sel(pol=3).imag
                        LL = ixds["IMAGE"].sel(pol=1) - ixds["IMAGE"].sel(pol=4)
                        # store results in new object with updated codes
                        NEW_IMAGE = xarray.concat([RR, RL, LR, LL], dim="pol")
                        NEW_IMAGE.assign_coords(
                            {
                                "pol": pol_codes.sel(
                                    basis="circular", drop=True
                                ).rename({"code_index": "pol"})
                            }
                        )
                        # Update the codes on pol dimension of all data variables in temp dataset
                        _ixds = _ixds.assign_coords(
                            pol=pol_codes.sel(basis="circular", drop=True)
                            .rename({"code_index": "pol"})
                            .data
                        )
                    # without full Stokes parameterization, only 2/4 correlations can be solved
                    # check for IV
                    elif [True, True] in (
                        pol_codes.sel(basis="stokes", code_index=[0, 3]).isin(
                            ixds["pol"].data
                        )
                    ):
                        print(
                            "Stokes I and V are present in input but either Q or U is absent"
                        )
                        print("Solving only for RR and LL")
                        # RR = I + V
                        # LL = I - V
                        RR = ixds["IMAGE"].sel(pol=1) + ixds["IMAGE"].sel(pol=4)
                        LL = ixds["IMAGE"].sel(pol=1) - ixds["IMAGE"].sel(pol=4)
                        # store results in new object with updated codes
                        NEW_IMAGE = xarray.concat([RR, LL], dim="pol")
                        # reassignment is necessary to preserve coord vals (vs. passing eval expression directly as in the ==4 case)
                        sub_codes = pol_codes.sel(
                            basis="circular", code_index=[0, 3]
                        ).data
                        NEW_IMAGE.assign_coords({"pol": sub_codes})
                        # Update the codes on pol dimension of all data variables in temp dataset
                        _ixds = (
                            _ixds.assign_coords(pol=sub_codes, drop=True)
                            .rename({"code_index": "pol"})
                            .data
                        )
                    # check for QU
                    elif [True, True] in (
                        pol_codes.sel(basis="stokes", code_index=[1, 2]).isin(
                            ixds["pol"].data
                        )
                    ):
                        print(
                            "Stokes Q and U are present in input but either I or V is absent"
                        )
                        print("Solving only for RL and LR")
                        # RL = Q + iU
                        # LR = Q - iU
                        RL = ixds["IMAGE"].sel(pol=2) + ixds["IMAGE"].sel(pol=3).imag
                        LR = ixds["IMAGE"].sel(pol=2) - ixds["IMAGE"].sel(pol=3).imag
                        # store results in new object with updated codes
                        NEW_IMAGE = xarray.concat([RL, LR], dim="pol")
                        # reassignment is necessary to preserve coord vals (vs. passing eval expression directly as in the ==4 case)
                        sub_codes = pol_codes.sel(
                            basis="circular", code_index=[1, 2]
                        ).data
                        NEW_IMAGE.assign_coords({"pol": sub_codes})
                        # Update the codes on pol dimension of all data variables in temp dataset
                        _ixds = (
                            _ixds.assign_coords(pol=sub_codes, drop=True)
                            .rename({"code_index": "pol"})
                            .data
                        )
                    try:
                        # After conditionals, add the output to our temp object for later return
                        # At present this will only work for the full-polarization case!
                        _ixds.assign(
                            {
                                "IMAGE": NEW_IMAGE.transpose(
                                    "l", "m", "time", "chan", "pol"
                                )
                            }
                        )
                    except ValueError as E:
                        # arguments without labels along dimension 'pol' cannot be aligned because they have different dimension size(s)
                        # NEW_IMAGE has updated pol_codes with size == 2, but all other images in O.G. dataset have old codes
                        print(E.args)
                        raise NotImplementedError(
                            "How do we combine NEW_IMAGE with original dataset after transforming pol dimension?"
                        )
                else:
                    # Cannot form a circular correlation from a single Stokes parameter
                    raise UserWarning(
                        "Unexpected pol dimension shape, check input dataset"
                    )
            else:
                # basis.startswith neither L nor R
                raise NotImplementedError(
                    f"Unsupported input for basis parameter: {basis}"
                )
        else:
            # input check failed OR there are some unexpected codes
            raise UserWarning(
                "Codes stored in pol dimension coordinate appear not to be in Stokes basis, check input dataset"
            )

    except (UserWarning, NotImplementedError) as E:
        print(E.args)  # Report the message
        print("Returning input dataset unchanged")
        return ixds

    except KeyboardInterrupt:
        print("Abort!")
        return ixds

    # if we reach this, we have calculated outputs in new basis (pol codes updated above) and stored as NEW_IMAGE
    # We have left IMAGE alone; if we replace it (or return a new dataset) info might be lost (e.g., size==3 case)
    # Q: Is it best to return as new data variable, new dataset, or same dataset with IMAGE replaced?
    new_ixds = _ixds
    return new_ixds
