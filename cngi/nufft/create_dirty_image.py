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

from __future__ import print_function

def create_dirty_image(vis_dataset, grid_parms):
      """
      Parameters
      ----------
      vis_xarray : xarray

      Returns
      -------
      """
      
      import numpy as np
      from numba import jit
      import time
      import math
      from cngi.nufft.grid import grid
      import dask.array.fft as dafft
      import xarray as xr
      import dask.array as da
      import matplotlib.pylab as plt
      
      
      grid_xdataset = grid(vis_dataset,grid_parms)
      
      
      
      plt.figure()
      plt.imshow(np.real(grid_xdataset.VIS_GRID[150,0,:,:]))
      plt.title('Grid')
      plt.colorbar()
      plt.show()
      
      print(grid_parms['imsize'][1])
      
      uncorrected_dirty_image = dafft.fftshift(dafft.ifft2(dafft.ifftshift(grid_xdataset.VIS_GRID, axes=(2,3)), axes=(2,3)), axes=(2,3))
      uncorrected_dirty_image = uncorrected_dirty_image * ((grid_parms['imsize'][0] * grid_parms['imsize'][1]))
      
      def my_divide(x,y):
        print(x.shape)
        print(y.shape)
        return x/y
      
      uncorrected_dirty_image = da.map_blocks(my_divide, uncorrected_dirty_image, grid_xdataset.SUM_WEIGHT, dtype=uncorrected_dirty_image.dtype)
      #uncorrected_dirty_image = np.divide(uncorrected_dirty_image,grid_xdataset.SUM_WEIGHT)
      
      #uncorrected_dirty_image = uncorrected_dirty_image * ((grid_parms['imsize'][0] * grid_parms['imsize'][1])) #/ grid_xdataset.SUM_WEIGHT)
      #corrected_dirty_image = uncorrected_dirty_image.real / grid_xdataset.CORRECTING_CGK 
      #corrected_dirty_image = corrected_dirty_image.compute()
      uncorrected_dirty_image.compute()
      print(uncorrected_dirty_image.shape)
      
      
      
      #img_xdataset = grid_xdataset
      
      #Corrected dirty image should be l,m coordinates
      #img_xdataset = img_xdataset.assign({'CORRECTED_DIRTY_IMAGE':xr.DataArray(da.array(corrected_dirty_image),dims=['chan','corr','u', 'v'])})
      
      #return img_xdataset
      
      '''
      #print(corrected_dirty_image.compute())
      print(uncorrected_dirty_image)
      grid_dict = {}
      coords = dict(zip(['chan', 'corr','u','v'], [list(range(ss)) for ss in vis_grid.shape]))
      grid_dict['VIS_GRID'] = xr.DataArray(da.array(vis_grid), dims=['chan','corr','u', 'v'])
      grid_dict['SUM_WEIGHT'] = xr.DataArray(da.array(sum_weight), dims=['chan','corr'])
      grid_dict['CORRECTING_CGK'] = xr.DataArray(da.array(correcting_cgk_image), dims=['u','v'])
      grid_dataset = xr.Dataset(grid_dict, coords=coords)
      '''
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      