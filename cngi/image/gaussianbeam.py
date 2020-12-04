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
def gaussianbeam(xds, source='commonbeam', scale=1.0, name='BEAM'):
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
    import cngi._helper.beams as chb
    
    # build beam from specified shape
    if not isinstance(source, str):
        beam = scale * chb.synthesizedbeam(source[0], source[1], source[2], len(xds.d0), len(xds.d1), xds.incr[:2])[0]
        beam_xda = xarray.DataArray(da.from_array(beam), dims=['d0', 'd1'], name=name)
        beams = np.array(source)
        
    # otherwise put it together from attributes section
    else:
        beams = np.array(xds.attrs[source])
        
        if beams.ndim == 1:
            np_beam = scale * chb.synthesizedbeam(beams[0], beams[1], beams[2], len(xds.d0), len(xds.d1), xds.incr[:2])[0]
            da_beam = da.from_array(np_beam, chunks=[xds.chunks['d0'][0], xds.chunks['d1'][0]])
            beam_xda = xarray.DataArray(da_beam, dims=['d0','d1'])
            
        else:
            chan_list = []
            for cc in range(beams.shape[0]):
                pol_list = []
                for pp in range(beams.shape[1]):
                    np_beam = scale*chb.synthesizedbeam(beams[cc][pp][0], beams[cc][pp][1], beams[cc][pp][2],
                                                    len(xds.d0), len(xds.d1), xds.incr[:2])[0][:,:,None,None]
                    da_beam = da.from_array(np_beam, chunks=[xds.chunks['d0'][0], xds.chunks['d1'][0],
                                                             xds.chunks['chan'][0], xds.chunks['pol'][0]])
                    pol_list += [xarray.DataArray(da_beam, dims=['d0','d1','chan','pol']).assign_coords(
                                                                                {'chan':[xds.chan[cc]], 'pol':[xds.pol[pp]]})]
                
                chan_list += [xarray.concat(pol_list, dim='pol')]
        
            beam_xda = xarray.concat(chan_list, dim='chan')
    
    return xds.assign({name:beam_xda}).assign_attrs({name+'_params':beams.tolist()})