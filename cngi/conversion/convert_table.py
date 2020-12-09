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
"""
this module will be included in the api
"""



def convert_table(infile, outfile=None, subtable=None, keys=None, timecols=None, ignorecols=None, compressor=None, chunk_shape=(40000, 20, 1), append=False, nofile=False):
    """
    Convert casacore table format to xarray Dataset and zarr storage format.

    This function requires CASA6 casatools module. Table rows may be renamed or expanded to n-dim arrays based on column values specified in keys.

    Parameters
    ----------
    infile : str
        Input table filename
    outfile : str
        Output zarr filename. If None, will use infile name with .tbl.zarr extension
    subtable : str
        Name of the subtable to process. If None, main table will be used
    keys : dict or str
        Source column mappings to dimensions. Can be a dict mapping source columns to target dims, use a tuple when combining cols
        (ie {('ANTENNA1','ANTENNA2'):'baseline'} or a string to rename the row axis dimension to the specified value.  Default of None
    timecols : list
        list of strings specifying column names to convert to datetime format from casacore time.  Default is None
    ignorecols : list
        list of column names to ignore. This is useful if a particular column is causing errors.  Default is None
    compressor : numcodecs.blosc.Blosc
        The blosc compressor to use when saving the converted data to disk using zarr.
        If None the zstd compression algorithm used with compression level 2.
    chunk_shape : int
        Shape of desired chunking in the form of (dim0, dim1, ..., dimN), use -1 for entire axis in one chunk. Default is (80000, 10).
        Chunking is applied per column / data variable.  If too few dimensions are specified, last chunk size is reused as necessary.
        Note: chunk size is the product of the four numbers, and data is batch processed by the first axis, so that will drive memory needed for conversion.
    append : bool
        Append an xarray dataset as a new partition to an existing zarr directory.  False will overwrite zarr directory with a single new partition
    nofile : bool
        Allows legacy table to be directly read without file conversion. If set to true, no output file will be written and entire table will be held in memory.
        Requires ~4x the memory of the table size.  Default is False
    Returns
    -------
    New xarray.core.dataset.Dataset
      New xarray Dataset of table data contents. One element in list per DDI plus the metadata global.
    """
    import os
    from numcodecs import Blosc
    import cngi._helper.table_conversion as tblconv

    # parse filename to use
    infile = os.path.expanduser(infile)
    prefix = infile[:infile.rindex('.')]
    if outfile is None: outfile = prefix + '.tbl.zarr'
    outfile = os.path.expanduser(outfile)
    if not infile.endswith('/'): infile = infile + '/'
    if not outfile.endswith('/'): outfile = outfile + '/'
    if subtable is None: subtable = ''
    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
        
    print('processing %s to %s' % (infile+subtable, outfile+subtable))

    # need to manually remove existing zarr file (if any)
    if (not nofile) and (not append):
        os.system("rm -fr " + outfile)
        os.system("mkdir " + outfile)
    
    if (keys is None) or (type(keys) is str):
        xds = tblconv.convert_simple_table(infile, outfile,
                                           subtable=subtable,
                                           rowdim='d0' if keys is None else keys,
                                           timecols=[] if timecols is None else timecols,
                                           ignore= [] if ignorecols is None else ignorecols,
                                           compressor=compressor,
                                           chunk_shape=chunk_shape,
                                           nofile=nofile)
    else:
        xds = tblconv.convert_expanded_table(infile, outfile,
                                             keys=keys,
                                             subtable=subtable,
                                             subsel=None,
                                             timecols=[] if timecols is None else timecols,
                                             dimnames={},
                                             ignore=[] if ignorecols is None else ignorecols,
                                             compressor=compressor,
                                             chunk_shape=chunk_shape,
                                             nofile=nofile)
    
    return xds
