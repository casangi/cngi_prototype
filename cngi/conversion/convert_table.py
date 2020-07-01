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


def convert_table(infile, outfile=None, compressor=None, nofile=False):
    """
    Convert casacore table format to xarray Dataset and zarr storage format

    This function requires CASA6 casatools module.

    Parameters
    ----------
    infile : str
        Input table filename
    outfile : str
        Output zarr filename. If None, will use infile name with .tbl.zarr extension
    compressor : numcodecs.blosc.Blosc
        The blosc compressor to use when saving the converted data to disk using zarr.
        If None the zstd compression algorithm used with compression level 2.
    nofile : bool
        Allows legacy MS to be directly read without file conversion. If set to true, no output file will be written and entire MS will be held in memory.
        Requires ~4x the memory of the MS size.  Default is False
    Returns
    -------
    New xarray.core.dataset.Dataset
      New xarray Dataset of table data contents. One element in list per DDI plus the metadata global.
    """
    import os
    from casatools import table as tb
    from numcodecs import Blosc
    import xarray
    import numpy as np
    from itertools import cycle
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)

    # parse filename to use
    infile = os.path.expanduser(infile)
    prefix = infile[:infile.rindex('.')]
    if outfile is None:
        outfile = prefix + '.tbl.zarr'
    else:
        outfile = os.path.expanduser(outfile)

    # need to manually remove existing zarr file (if any)
    print('processing %s ' % infile)
    if not nofile:
        os.system("rm -fr " + outfile)
        os.system("mkdir " + outfile)
    
    tb_tool = tb()
    tb_tool.open(infile, nomodify=True, lockoptions={'option': 'usernoread'})  # allow concurrent reads
    
    mvars = {}
    print('processing table...')
    for col in tb_tool.colnames():
        if not tb_tool.iscelldefined(col, 0):
            print("##### ERROR processing column %s, skipping..." % col)
            continue
        if tb_tool.isvarcol(col):
            data = tb_tool.getvarcol(col)
            tshape = np.max([list(data['r' + str(kk)].shape) for kk in np.arange(len(data)) + 1], axis=0)
            # expand the variable shaped column to the maximum size of each dimension
            # blame the person who devised the tb.getvarcol return structure
            data = [np.pad(data['r'+str(kk)], np.stack((tshape*0, tshape-data['r'+str(kk)].shape),-1)) for kk in np.arange(len(data))+1]
            data = np.array(data)
        else:
            data = tb_tool.getcol(col).transpose()
        dims = ['d0'] + ['d%s_%s' % (str(ii), str(data.shape[ii])) for ii in range(1,data.ndim)]
        mvars[col.upper()] = xarray.DataArray(data, dims=dims)
    
    tb_tool.close()

    xds = xarray.Dataset(mvars)
    
    if not nofile:
        print('writing...')
        encoding = dict(zip(list(xds.data_vars), cycle([{'compressor': compressor}])))
        xds.to_zarr(outfile, mode='w', encoding=encoding, consolidated=True)
    
    return xds
