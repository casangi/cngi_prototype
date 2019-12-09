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
def image_to_zarr(infile, outfile=None, artifacts=None):
  """
  Convert legacy format Image to xarray compatible zarr format image
  
  This function requires CASA6 casatools module. 
  
  Parameters
  ----------
  infile : str
      Input image filename
  outfile : str
      Output zarr filename. If None, will use infile name with .zarr extension
  artifacts : list of strings
      List of other image artifacts to include if present with infile. Default None = ['mask','model','pb','psf','residual','sumwt']
  
  Returns
  -------
  """    
  from casatools import image as ia
  import numpy as np
  import os
  import time
  from itertools import cycle
  from pandas.io.json._normalize import nested_to_record
  from xarray import Dataset as xd
  from xarray import DataArray as xa
  from numcodecs import Blosc
  
  print("converting Image...")
  
  infile = os.path.expanduser(infile)
  prefix = infile[:infile.rindex('.')]
  suffix = infile[infile.rindex('.')+1:]
  if outfile == None: 
    outfile = prefix + '.zarr'
  else:
    outfile = os.path.expanduser(outfile)
  tmp = os.system("rm -fr " + outfile)
  begin = time.time()
  
  compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
  IA = ia()
  
  # all image artifacts will go in same zarr file and share common dimensions if possible
  # check for meta data compatibility
  # store necessary coordinate conversion data
  if artifacts == None: 
    imtypes = [suffix] + ['mask','model','pb','psf','residual','sumwt']
  else: 
    imtypes = [suffix] + artifacts
  meta, tm, diftypes, difmeta = {}, {}, [], []
  for imtype in imtypes:
    if os.path.exists(prefix + '.' + imtype): 
      rc = IA.open(prefix + '.' + imtype) 
      
      summary = IA.summary(list=False)
      tm['start'] = np.array(summary['refval'] - summary['refpix']*summary['incr'])
      tm['stop'] = np.array(tm['start'] + summary['shape']*summary['incr'])
      tm['dsize'] = np.array(summary['shape'])
      coord_names = [ss.replace(' ','_') for ss in summary['axisnames']]
      coords = [np.linspace(tm['start'][xx], tm['stop'][xx], tm['dsize'][xx], endpoint=False) 
                for xx in range(len(tm['dsize']))]
      coords = dict(zip(coord_names, coords))
      tm['freq_coords'] = coords.pop('Frequency', None)
      tm['coords'] = coords
      if len(np.where(summary['axisnames'] == 'Frequency')[0]) != 1:
        print("#### ERROR: can't find channel axis")
      tm['chan_dim'] = np.where(summary['axisnames'] == 'Frequency')[0][0]
      omitkeys = ['axisnames','incr','ndim','refpix','refval','shape','tileshape','messages']
      for key in [key for key in summary.keys() if key not in omitkeys]:
        tm[key] = summary[key]
      if meta == {}: 
        meta = dict(tm)
      elif np.any([np.any(meta[kk] != tm[kk]) for kk in ['start','stop','dsize']]):
        diftypes += [imtype]
        difmeta += [tm]
        imtypes = [_ for _ in imtypes if _ != imtype]
      
      rc = IA.close()
    else:
      imtypes = [_ for _ in imtypes if _ != imtype]
  
  # process all image artifacts with compatible metadata to same zarr file
  # partition by channel, read each image artifact for each channel
  dsize, chan_dim = meta['dsize'], meta['chan_dim']
  for chan in range(dsize[chan_dim]):
    print('processing channel ' + str(chan+1) + ' of ' + str(dsize[chan_dim]))
    pt = [-1 for _ in range(len(dsize))]
    xdas = {}
    for imtype in imtypes:
      rc = IA.open(prefix + '.' + imtype) 
      pt[chan_dim] = chan
      imchunk = IA.getchunk(pt, pt)
      rc = IA.close()
      
      coords = dict(meta['coords'])
      coords['Frequency'] = [meta['freq_coords'][chan]]
      if imtype == 'fits': imtype = 'image'
      xdas[imtype] = xa(imchunk, dims=list(coords.keys()))
    
    omitkeys = ['start','stop','dsize','freq_coords','coords','chan_dim']
    attrs = dict([(kk, meta[kk]) for kk in meta.keys() if kk not in omitkeys])
    if chan == 0:
      xds = xd(xdas, coords=coords, attrs=nested_to_record(attrs, sep='_'))
      encoding = dict(zip(list(xds.data_vars), cycle([{'compressor':compressor}])))
      xds.to_zarr(outfile, mode='w', encoding=encoding)
    else: 
      xds = xd(xdas, coords=coords, attrs=nested_to_record(attrs, sep='_'))
      xds.to_zarr(outfile, mode='a', append_dim='Frequency')
  
  print("processed image size " + str(dsize) + " in " + str(np.float32(time.time()-begin)) + " seconds")
  
  
  # process remaining image artifacts with different sizes to different zarr files
  # partition by channel, read each channel for each image artifact
  for nn,imtype in enumerate(diftypes):
    outfile = prefix + '.' + imtype + '.zarr'
    tmp = os.system("rm -fr " + outfile)
    dsize, chan_dim = difmeta[nn]['dsize'], difmeta[nn]['chan_dim']
    pt = [-1 for _ in range(len(dsize))]
    rc = IA.open(prefix + '.' + imtype) 
    for chan in range(dsize[chan_dim]):
      print('processing ' + imtype + ' channel ' + str(chan+1) + ' of ' + str(dsize[chan_dim]))  
      pt[chan_dim] = chan
      imchunk = IA.getchunk(pt, pt)
      coords = dict(difmeta[nn]['coords'])
      coords['Frequency'] = [difmeta[nn]['freq_coords'][chan]]
      xda = xa(imchunk, dims=list(coords.keys()))
      
      omitkeys = ['start','stop','dsize','freq_coords','coords','chan_dim']
      attrs = dict([(kk, difmeta[nn][kk]) for kk in difmeta[nn].keys() if kk not in omitkeys])
      if chan == 0:
        xds = xd({imtype:xda}, coords=coords, attrs=nested_to_record(attrs, sep='_'))
        encoding = dict(zip(list(xds.data_vars), cycle([{'compressor':compressor}])))
        xds.to_zarr(outfile, mode='w', encoding=encoding)
      else: 
        xds = xd({imtype:xda}, coords=coords, attrs=nested_to_record(attrs, sep='_'))
        xds.to_zarr(outfile, mode='a', append_dim='Frequency')
    
    rc = IA.close()
  
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
