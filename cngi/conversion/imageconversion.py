#   Copyright 2019 AUI, Inc. Washington DC, USA
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



##########################################
def image_to_zarr(infile, outfile=None):
    """
    Convert legacy format Image to xarray compatible zarr format image

    This function requires CASA6 casatools module. 

    Parameters
    ----------
    infile : str
        Input image filename
    outfile : str
        Output zarr filename. If None, will use infile name with .zarr extension
    
    Returns
    -------
    """    
    from casatools import image as ia
    import numpy as np
    import os, time
    from itertools import cycle
    from xarray import Dataset as xd
    from xarray import DataArray as xa
    from numcodecs import Blosc
    
    print("converting Image...")
    
    infile = os.path.expanduser(infile)
    prefix = infile[:infile.rindex('.')]
    if outfile==None: outfile = prefix + '.zarr'
    tmp = os.system("rm -fr " + outfile)
    tmp = os.system("mkdir " + outfile)
    
    compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
    
    IA = ia()
    IA.open(infile)
    begin = time.time()
    
    # compute image coordinates
    summary = IA.summary(list=False)
    start = summary['refval'] - summary['refpix']*summary['incr']
    stop = start + summary['shape']*summary['incr']
    dsize = summary['shape']
    coords = [np.linspace(start[xx], stop[xx], dsize[xx], endpoint=False) for xx in range(len(dsize))]
    coords = dict(zip(summary['axisnames'], coords))
    freq_coords = coords.pop('Frequency', None)
    
    # figure out where freq dimension is
    if len(np.where(summary['axisnames'] == 'Frequency')[0]) != 1:
        print("#### ERROR: can't find channel axis")
    chan_dim = np.where(summary['axisnames'] == 'Frequency')[0][0]
    
    # partition by channel
    pt = [-1 for _ in range(summary['ndim'])]
    for chan in range(dsize[chan_dim]):
      print('processing channel ' + str(chan) + ' of ' + str(dsize[chan_dim]))
      pt[chan_dim] = chan
      coords['Frequency'] = [freq_coords[chan]]
      imchunk = IA.getchunk(pt, pt)
      imxa = xa(imchunk, dims=summary['axisnames'])
      xds = xd({'image':imxa}, coords=coords)
      if chan == 0:
        xds.to_zarr(outfile, mode='a', append_dim='Frequency', encoding={'image':{'compressor':compressor}})
      else: xds.to_zarr(outfile, mode='a', append_dim='Frequency')
    
    IA.close()
    
    print("processed image size " + str(dsize) + " in " + str(np.float32(time.time()-begin)) + " seconds")
    print('complete')




###########################################
def zarr_to_image(infile, outfile=None):
    """
    .. todo::
        This function is not yet implemented

    Convert xarray compatible zarr format to legacy CASA Image format

    Parameters
    ----------
    infile : str
        Input zarr image filename
    outfile : str
        Output image filename. If None, will use infile name with .image extension

    Returns
    -------
    """
    return True




###########################################
def fits_to_zarr(infile, outfile=None):
    """
    .. todo::
        This function is not yet implemented

    Convert FITS format Image to xarray compatible zarr format image (future)
    
    Parameters
    ----------
    infile : str
        Input FITS filename
    outfile : str
        Output zarr filename. If None, will use infile name with .zarr extension

    Returns
    -------
    """
    return True



############################################
def zarr_to_fits(infile, outfile=None):
    """
    .. todo::
        This function is not yet implemented

    Convert xarray compatible zarr image to FITS format image (future)
    
    Parameters
    ----------
    infile : str
        Input zarr filename
    outfile : str
        Output FITS filename. If None, will use infile name with .fits extension

    Returns
    -------
    """
    return True
