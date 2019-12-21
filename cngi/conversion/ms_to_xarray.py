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

import os
import numpy as np
from numcodecs import Blosc
import time
from .xarray_conversion_tools import *
print("numpy version: %s" % (np.__version__))

def ms_to_xarray(infile, outfile=None, ddi=None, maxchunksize=1000, compressor = Blosc(cname='zstd', clevel=2, shuffle=0)):
    """
    Convert legacy format MS to a xarray dataset which is saved to disk using zarr

    This function requires CASA6 casatools module. 

    Parameters
    ----------
    infile : str
        Input MS filename
    outfile : str
        Output zarr filename. If None, will use infile name with .zarr extension
    ddi : int
        Specific ddi to convert. Leave as None to convert entire MS
    compressor : blosc
        The blosc compressor to use when saving the converted data to disk using zarr. 
        If None the zstd compression algorithm used with compression level 2. 

    Returns
    -------
    """
    
    from casatools import table as tb
    from casatools import ms
    from casatools import msmetadata as msmd

    # Parse filename to use
    prefix = infile[:infile.rindex('.')]
    if outfile == None: outfile = prefix + '.zarr'
    
    # Need to manually remove existing directory (if any)
    tmp = os.system("rm -fr " + outfile)
    tmp = os.system("mkdir " + outfile)
    
    
    # Get each DATA_DESC_ID (ddi). Visbility data for each ddi is assumed to have fixed shape.
    tb_tool = tb(infile)
    tb_tool.open(infile, nomodify=True, lockoptions={'option':'usernoread'})
    ddis = tb_tool.taql('select distinct DATA_DESC_ID from %s' % prefix+'.ms').getcol('DATA_DESC_ID')
    tb_tool.close()
    
    # Parallelize with direct interface
    client = None#GetFrameworkClient()
    if ddi != None:
        processDDI(ddi,infile,outfile,compressor)
    elif client == None:
        for ddi in ddis:
            processDDI(ddi,infile,outfile,compressor)
    else:
        jobs = client.map(processDDI, ddis, 
                          np.repeat(infile, len(ddis)), 
                          np.repeat(outfile, len(ddis)),
                          np.repeat(compressor, len(ddis)))
        
        # block until complete
        for job in jobs: job.result()
        print('Complete.')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
