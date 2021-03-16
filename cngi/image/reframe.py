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

def reframe(
    ixds,
    outframe=None,
    reference_time=None,
    observer_location=None,
    target_location=None,
    reference_frequency=None,
    # velocity=None,
):
    """
    Change the velocity system of an image

    .. warn::
        This function is stil being implemented

    .. todo:
        Use refpix attribute (if present) to improve accuracy of target_location assignment

    .. todo:
        Account for epoch, distance, and velocity (if present in input attributes) in target_location and observer_location

    .. todo:
        decide whether to drop velocity coordinate (if assigned)

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

    # create a working object
    _ixds = ixds

    # it would be better if the axis attributes were a single dict
    _ixds.attrs["units_map"] = dict(
        zip(_ixds.attrs["axisnames"], _ixds.attrs["axisunits"])
    )

    # Assigning reference_time
    if reference_time is None:
        try:
            reference_time = _ixds["time"].values[0]
        except:
            reference_time = datetime.now()
        finally:
            print(f"Adopted reference time: {reference_time}")

    # Assigning observer_location
    if observer_location is None:
        try:
            telpos = [
                xx.strip(",[]()") for xx in _ixds.attrs["telescope_position"].split(" ")
            ]
            telcoord = [float(xyz.strip("m")) for xyz in telpos[:-1]]
            if telpos[-1].lower() == "itrf":
                observer_location = SkyCoord(
                    x=telcoord[0],
                    y=telcoord[1],
                    z=telcoord[2],
                    unit=units.Unit("m"),
                    frame="itrs",
                    obstime=_ixds["time"].values[0],
                )
            else:
                raise NotImplementedError(
                    "Unsupported frame accompanying telescope_position attribute of input dataset"
                )
        except ValueError:
            print("Failed to find valid telescope postition from dataset attributes")
            print("Defaulting to astropy-data telescope location")
            observer_location = EarthLocation.of_site(
                _ixds.attrs["telescope"]
            ).get_itrs(obstime=Time(_ixds["time"].values[0]))
        finally:
            print(f"Adopted reference postion: {observer_location}")

    # Assigning reference_frequency
    if reference_frequency is None:
        try:
            reference_frequency = units.Quantity(_ixds.attrs["rest_frequency"])
        except ValueError:
            print("Failed to assign valid rest frequency from dataset attributes")
            print("Attempting to parse and convert")
            try:
                malformed_quantity = _ixds.attrs["rest_frequency"].split(" ")
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
            
    # Assigning target_location
    if target_location is None:
        try:
            target_location = SkyCoord.from_name(
                _ixds.attrs["object_name"], frame=_ixds.attrs["spectral__reference"]
            )
        except:  # NameResolveError
            # Here is where we could add another try block to check for velocity attributes in the ixds
            # Also check for refpix attribute in the input instead of having to calculate it
            # If not available and we couldn't get pointing center in terms of coordinates...
            print(
                f"Direction reference '{_ixds.attrs['pointing_center']}' is not a valid SkyCoord initializer"
            )
            target_location = SkyCoord(
                _ixds["right_ascension"][_ixds.l.size // 2][_ixds.m.size // 2].values
                * units.Unit(ixds.attrs["units_map"]["Right Ascension"]),
                _ixds["declination"][_ixds.l.size // 2][_ixds.m.size // 2].values
                * units.Unit(_ixds.attrs["units_map"]["Declination"]),
                frame=_ixds.attrs["spectral__reference"],
                # if target frame velocity reference is available, it would be easiest to assign it here
            )
            # note that we could include telescope dependent conditionals (e.g., FK5 vs ICRS)
            # these could likely be based on ixds.attrs["direction_reference"]
        finally:
            print(f"Adopted target location: {target_location}")
            if target_location.is_transformable_to(outframe):
                pass
            else:
                print("Input to outframe argument incompatible with target position")
                print(
                    "Adding velocity coordinate for per-channel assignment to target_location"
                )
                # if velocity information isn't present in the input
                # (and it won't be because that logic hasn't been encoded due to available test data)
                # then calculate a velocity for each channel and assign as dataset coordinate,
                # for selection inside the mapped function (_change_frame)
                measured_frequencies = _ixds.chan.values * units.Unit(
                    _ixds.attrs["units_map"]["Frequency"]
                )
                radio_equiv = units.doppler_radio(reference_frequency)
                measured_velocities = measured_frequencies.to(
                    units.km / units.s, equivalencies=radio_equiv
                )
                _ixds = _ixds.assign_coords(velocity=("chan", measured_velocities.value))
                _ixds.attrs["units_map"][
                    "Velocity"
                ] = measured_velocities.unit.to_string()

    # Defining a function to apply the transformation across the chan axis
    def _change_frame(ixds_block, observer, target, frequency, frame):
        """
        This function will be called using xr.map_blocks

        reference
        http://xarray.pydata.org/en/stable/generated/xarray.map_blocks.html
        and
        https://xarray-contrib.github.io/xarray-tutorial/scipy-tutorial/06_xarray_and_dask.html#map_blocks
        """

        if ("velocity" in ixds_block.coords) and (
            "Velocity" in ixds_block.attrs["units_map"]
        ):
            # update target to use velo coordinate value for this block
            # since this is inside map_blocks, the only velo/chan available is the one we want to use
            chan_vel = ixds_block.velocity.values * units.Unit(
                ixds_block.attrs["units_map"]["Velocity"]
            )
            # assignment via `SkyCoord(target, radial_velocity=chan_vel)` doesn't seem to work
            # see 
            # https://docs.astropy.org/en/stable/coordinates/velocities.html#adding-velocities-to-existing-frame-objects
            # https://docs.astropy.org/en/stable/coordinates/representations.html#attaching-differential-objects-to-representation-objects
            target = target.data.with_differentials({'s':RadialDifferential(chan_vel)})

        for xda in ixds_block.data_vars:
            # we don't want to transform boolean data arrays
            # we also don't want to if there is no chan dimension
            if ixds_block[xda].dtype != bool and "chan" in ixds_block[xda].dims:
                # ixds_block[xda].values is the computed ndarray for this block
                sc = SpectralCoord(
                    ixds_block[xda].values,
                    unit=ixds_block.attrs["units_map"]["Frequency"],
                    observer=observer,
                    target=target,
                    doppler_rest=frequency,
                    doppler_convention=ixds_block.attrs["velocity__type"],
                )
                # see https://docs.astropy.org/en/stable/coordinates/spectralcoord.html#specifying-an-observer-and-a-target-explicitly
                new_sc = sc.with_observer_stationary_relative_to(frame)
                ixds_block[xda] = new_sc.values

        return ixds_block

    # Apply the function to the data
    # The parallelism happens inside this function
    try:
        output_ixds = xr.map_blocks(
            obj=_ixds,
            func=_change_frame,
            args=[],  # no positional args
            kwargs={
                "observer": observer_location,
                "target": target_location,
                "frequency": reference_frequency,
                "frame": outframe,
            },
            template=_ixds,
        )
    except:
        print("Failed to change frame. Returning original dataset.")
        output_ixds = _ixds

    # ouptut_ixds.compute()
    return output_ixds
