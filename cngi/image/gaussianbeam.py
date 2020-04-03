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

########################
def gaussianbeam(xds, source='perplanebeams', scale=1.0, name='BEAM'):
    """
    Construct a gaussian beam of image dimensions from specified size or per-plane beam parameters

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    source : str or tuple of floats
        Source xds attr name to find the per-plane beam information, or a tuple describing the major axis, minor axis, and position angle
        of the desired gaussian in (arcsec, arcsec, degrees), for example (1., 1., 30.).  Default is 'perplanebeams'
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
    import cngi._helper.beams as chb
    
    # build beam from specified shape
    if type(source) is tuple:
        beam = chb.synthesizedbeam(source[0], source[1], source[2], len(xds.d0), len(xds.d1), xds.incr[:2])
        beam_xda = xarray.DataArray(da.from_array(beam), dims=['d0', 'd1'], name=name)*scale
        
    # otherwise put it together from attributes section
    else:
        # note that attributes are just normal python lists, so this is not parallelized
        # parse per-plane beams in to a dictionary of parameters
        beams = {}
        for line in xds.attrs[source]:
            if not line[0].startswith('beam'): continue
            lp = line[0].split('.')
            if lp[1] not in beams: beams[lp[1]] = {}  # add channel to dict
            if lp[2] not in beams[lp[1]]: beams[lp[1]][lp[2]] = {}  # add pol dict
            beams[lp[1]][lp[2]][lp[3]+'.'+lp[4]] = line[1]
        
        # populate output xarray
        chan_list = []
        for cc in beams.keys():
            pol_list = []
            for pp in beams[cc].keys():
                beam = beams[cc][pp]
                chan, pol = int(cc[-1]), int(pp[-1])
                if (beam['major.unit'] != 'arcsec') or (beam['minor.unit'] != 'arcsec') or (beam['positionangle.unit'] != 'deg'):
                    print('ERROR: buildbeam unit conversion needed')
                
                np_beam = scale*chb.synthesizedbeam(beam['major.value'],
                                                    beam['minor.value'],
                                                    beam['positionangle.value'],
                                                    len(xds.d0), len(xds.d1), xds.incr[:2])[:,:,None,None]
                
                da_beam = da.from_array(np_beam, chunks=[xds.chunks['d0'][0], xds.chunks['d1'][0], xds.chunks['chan'][0], xds.chunks['pol'][0]])
                pol_list += [xarray.DataArray(da_beam, dims=['d0','d1','chan','pol']).assign_coords(
                                                                            {'chan':[xds.chan[chan]], 'pol':[xds.pol[pol]]})]
                
            chan_list += [xarray.concat(pol_list, dim='pol')]
        
        beam_xda = xarray.concat(chan_list, dim='chan')
    
    return xds.assign({name:beam_xda})