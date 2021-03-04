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
def reframe(
    ixds,
    outframe=None,
    reference_time=None,
    observer_location=None,
    target_location=None,
    reference_frequency=None,
    **kwargs,
):
    """
    Change the velocity system of an image

    .. warn::
        This function is stil being implemented

    .. todo:
        Account for epoch and distance in target_location and observer_location

    .. todo:
        Account for velocity information during _change_frame (if present in the input)

    .. todo:
        call utility that will update some properties of global_xds after conversion

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Image
    outframe : str
        reference frame to which the input will be converted
    reference_time : numpy.datetime64 or datetime, optional
        any valid initializer of astropy.time.TimeBase should work
    observer_location : astropy.cordinates.SkyCoord, optional
        position and velocity of observer for frame transformation
    target_location : astropy.cordinates.SkyCoord, optional
        position and velocity of source for frame transformation
    reference_frequency : astropy.units.Quantity, optional
        value and unit to use when expressing the spectral value as a velocity, input to SpectralCoord doppler_rest parameter

    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    from astropy import units
    from astropy.coordinates import EarthLocation, SkyCoord, SpectralCoord
    from astropy.time import Time
    from datetime import datetime
    import xarray as xr
    import numpy as np

    # it would be better if the axis attributes were a single dict
    ixds.attrs["units_map"] = dict(
        zip(ixds.attrs["axisnames"], ixds.attrs["axisunits"])
    )

    # Assigning reference_time
    if reference_time is None:
        try:
            reference_time = ixds["time"].values[0]
        except:
            reference_time = datetime.now()
        finally:
            print(f"Adopted reference time: {reference_time}")

    # Assigning observer_location
    if observer_location is None:
        try:
            telpos = [
                xx.strip(",[]()") for xx in ixds.attrs["telescope_position"].split(" ")
            ]
            telcoord = [float(xyz.strip("m")) for xyz in telpos[:-1]]
            telpos[-1].lower() == "itrf"
            observer_location = SkyCoord(
                x=telcoord[0],
                y=telcoord[1],
                z=telcoord[2],
                unit=units.Unit("m"),
                frame="itrs",
                obstime=ixds["time"].values[0],
            )
        except ValueError:
            print("Failed to find valid telescope postition from dataset attributes")
            print("Defaulting to astropy-data telescope location")
            observer_location = EarthLocation.of_site(ixds.attrs["telescope"]).get_itrs(
                obstime=Time(ixds["time"].values[0])
            )
        finally:
            print(f"Adopted reference postion: {observer_location}")

    # Assigning target_location
    if target_location is None:
        try:
            target_location = SkyCoord.from_name(ixds.attrs["object_name"])
        except:  # NameResolveError
            print(
                f"Direction reference '{ixds.attrs['pointing_center']}' is not a valid SkyCoord initializer"
            )
            # if we can't get pointing center in terms of coordinates...
            target_location = SkyCoord(
                ixds.right_ascension.values.mean()
                * units.Unit(ixds.attrs["units_map"]["Right Ascension"]),
                ixds.declination.values.mean()
                * units.Unit(ixds.attrs["units_map"]["Declination"]),
                frame="icrs",
            )
            # note that we could include telescope dependent conditionals 
            # these could likely be based on ixds.attrs["direction_reference"]
        finally:
            print(f"Adopted target location: {target_location}")

    # Assigning reference_frequency
    if reference_frequency is None:
        try:
            reference_frequency = units.Quantity(ixds.attrs["rest_frequency"])
        except ValueError:
            print("Failed to assign valid rest frequency from dataset attributes")
            print("Attempting to parse and convert")
            try:
                malformed_quantity = ixds.attrs["rest_frequency"].split(" ")
                if isinstance(malformed_quantity[0], str):
                    value = float(malformed_quantity[0])
                # this is where pint-xarray would be useful
                # note that we'll have to extend this logic to handle a variety of inputs, e.g., km/s
                if malformed_quantity[1].lower() == "hz":
                    unit = "Hz"
                    reference_frequency = units.Quantity(value, unit)
            except (AttributeError, ValueError):
                # AttributeError: seems like the attribute is not a string
                # ValueError: seems like attribute can't be converted to float
                print("Failed to convert dataset attributes to valid rest frequency")
                print("Assuming 100 GHz, this may not be desired")
                value, unit = 100, "GHz"
                reference_frequency = units.Quantity(value, unit)
        finally:
            print(f"Adopted reference frequency: {reference_frequency}")

    # Defining a function to apply the transformation across the chan axis
    def _change_frame(
        input_array, observer, target, frequency, frame
    ):

        # the input_array will act on the relevant xr.DataArray
        sc = SpectralCoord(
            input_array,
            unit=ixds.attrs["units_map"]["Frequency"],
            observer=observer,
            target=target,
            doppler_rest=reference,
            doppler_convention=ixds.attrs["velocity__type"],
        )

        output_array = sc.with_observer_stationary_relative_to(frame, velocity=None).value

        return output_array(axis=-1, keepdims=True)

    # Apply the function to the data
    # This is where the parallelism happens
    output_ixds = xr.apply_ufunc(
        _change_frame,
        observer_location,
        target_location,
        reference_frequency,
        outframe,
        ixds.IMAGE.chunk({"chan": -1}),
        input_core_dims=[["chan"]],
        dask="parallelized",
        output_dtypes=[ixdsxds.IMAGE.dtype],
    )

    # ouptut_ixds.compute()
    return output_ixds
