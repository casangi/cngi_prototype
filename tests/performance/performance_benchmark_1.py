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


###############################################################################
# End-to-end performance benchmarking

# 1. Convert a legacy MS to visibility zarr file
# 2. Grid visibilities and produce dirty image
# 3. Save dirty image to image zarr file
# 4. display preview of dirty image
###############################################################################

from cngi.conversion import ms_to_zarr
from cngi.direct import InitializeFramework
from cngi.dio import read_vis, read_image, write_image
from cngi.gridding import dirty_image
from cngi.image import preview
import os
import time

if not os.path.isdir("data"): os.system("mkdir data >/dev/null")

if __name__ == '__main__':
    client = InitializeFramework(workers=4, memory='4GB')
    
    ##########
    ## RUN THIS or CHANGE to desired input MS
    ##########
    print("downloading MeasurementSet from CASAguide First Look at Imaging...")
    os.system("wget -q https://bulk.cv.nrao.edu/almadata/public/working/sis14_twhya_calibrated_flagged.ms.tar")
    os.system("tar -C data/ -xvf sis14_twhya_calibrated_flagged.ms.tar >/dev/null")
    
    infile = 'data/sis14_twhya_calibrated_flagged.ms'
    
    print('Converting MS to Zarr...')
    ms_to_zarr(infile, outfile='data/benchmark.vis.zarr')
    
    print('Gridding Visibilities and saving image...')
    start = time.time()
    v_xds = read_vis('data/benchmark.vis.zarr')
    i_xds = dirty_image(v_xds, field_id=None, imsize=[200,400], cell=[0.08, 0.08], nchan=v_xds.dims['chan'])
    write_image(i_xds, 'data/benchmark.img.zarr')
    print('time to grid and save data : ', time.time() - start, ' seconds')
    
    start = time.time()
    xds = read_image('data/benchmark.img.zarr')
    preview(xds)
    print('time to preview output image : ', time.time() - start, ' seconds')
    print('\nbenchmark image summary : \n', xds)
    
    client.close()
