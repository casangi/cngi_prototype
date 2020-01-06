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

def test_ms_to_zarr_numba():
    import cngi
    from cngi.conversion.ms_to_zarr_numba import ms_to_zarr_numba
    import os
    import xarray as xr
    
    cngi_path = os.path.dirname(cngi.__file__)
    cngi_prototype_path = cngi_path[:cngi_path.rfind('/')]
    
    infile = cngi_prototype_path +  '/data/sis14_twhya_field5_mstrans_lsrk.ms'
    outfile = cngi_prototype_path + '/data/test_ms_to_numba_zarr.ms.zarr'
    ms_to_zarr_numba(infile,outfile)
    
    x_dataset = xr.open_zarr(outfile + '/0')
    print(x_dataset)
    
    tmp = os.system("rm -fr " + outfile)
  
if __name__ == '__main__':
    test_ms_to_zarr_numba()
    