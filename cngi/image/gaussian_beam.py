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
def gaussian_beam(xds, source='commonbeam', scale=1.0, name='BEAM'):
    """
    Construct a gaussian beam of image dimensions from specified size or beam attribute
    
    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Image Dataset
    source : str or list of floats
        Source xds attr name to find the beam information, or a list describing the major axis, minor axis, and position angle
        of the desired gaussian in (arcsec, arcsec, degrees), for example [1., 1., 30.].  Default is 'commonbeam'
    scale : float
        peak amplitude of beam. Default is unity (1.0)
    name : str
        dataset variable name for output beam(s), overwrites if already present.  Default is 'BEAM'

    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    import xarray
    import dask.array as da
    import numpy as np
    import cngi._utils._beams as chb

    if not isinstance(source, str):
        beams = np.array(source)
    else:
        beams = np.array(xds.attrs[source])

    # build beam from specified shape
    if beams.ndim == 1:
        beam = scale * chb.synthesizedbeam(beams[0], beams[1], beams[2], len(xds.l), len(xds.m), xds.incr[:2])[0]
        beam_xda = xarray.DataArray(da.from_array(beam), dims=['l', 'm'], name=name)

    # perplanebeams are stored chans x pols, each one needs to be processed individually
    else:
        beam = [scale*chb.synthesizedbeam(pp[0], pp[1], pp[2], len(xds.l), len(xds.m), xds.incr[:2])[0] for beam in beams for pp in beam]
        beam_xda = xarray.DataArray(da.from_array(beam).reshape(xds.dims['chan'], xds.dims['pol'], xds.dims['l'], xds.dims['m']),
                                    dims=['chan','pol','l','m'], name=name).chunk({'l':xds.chunks['l'], 'm':xds.chunks['m'],
                                                                                   'chan':xds.chunks['chan'], 'pol':xds.chunks['pol']})
        beam_xda = beam_xda.transpose('l', 'm', 'chan', 'pol')

    return xds.assign({name:beam_xda}).assign_attrs({name+'_params':beams.tolist()})