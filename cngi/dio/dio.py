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


#############################################
def read_ms(infile, ddi=0, columns=None):
    """
    Read Apache Parquet format MS from disk

    Parameters
    ----------
    infile : str
        input MS filename
    ddi : int
        Data Description ID in MS to read. Defaults to 0
    columns : list
        List of column names (as strings) to read. If None, read all (bad idea!)

    Returns
    -------
    Dask Dataframe
        New Dataframe of MS contents
    """
    import os
    import dask.dataframe as dd
    
    infile = os.path.expanduser(infile)
    ddf = dd.read_parquet(infile+'/'+str(ddi), engine='pyarrow', columns=columns, gather_statistics=False)
    return ddf



#############################################
def write_ms(df, outfile='ms.pq', ddi=0, append=False):
    """
    Write MS dataframe to Apache Parquet format on disk
    
    Parameters
    ----------
    df : Dask Dataframe
        MS dataframe to write to disk
    outfile : str
        output filename, generally ends in .pq
    ddi = int
        Data Description ID in MS to write. Defaults to 0
    append = bool
        Append this DDI in to an existing Parquet MS. Default=False will erase old pq file

    Returns
    -------
    """
    import os
    import dask.dataframe as dd
    
    outfile = os.path.expanduser(outfile)
    
    # need to manually remove existing parquet file (if any)
    if not append:
        tmp = os.system("rm -fr " + outfile)
    else: # still need to remove existing ddi (if any)
        tmp = os.system("rm -fr " + outfile+'/'+str(ddi))
    
    dd.to_parquet(df, outfile+'/'+str(ddi), engine='pyarrow', compression='snappy', 
                  write_metadata_file=True, compute=True)




#############################################
def read_zarr_ms(infile, ddi=0):
    """
    Read xarray zarr format MS from disk
    
    Parameters
    ----------
    infile : str
        input MS filename
    ddi : int
        Data Description ID in MS to read. Defaults to 0
    
    Returns
    -------
    xarray Dataset
        New xarray Dataset of MS contents
    """
    import os
    from xarray import open_zarr
    
    infile = os.path.expanduser(infile)
    xds = open_zarr(infile+'/'+str(ddi))
    return xds




#############################################
def read_image(infile):
    """
    Read xarray zarr format image from disk
    
    Parameters
    ----------
    infile : str
        input zarr image filename
    
    Returns
    -------
    xarray Dataset
        New xarray Dataset of image contents
    """
    import os
    from xarray import open_zarr
    
    infile = os.path.expanduser(infile)
    xds = open_zarr(infile)
    return xds




#############################################
def write_image(ds, outfile='image.zarr'):
    """
    Write image dataset to xarray zarr format on disk
    
    Parameters
    ----------
    ds : xarray Dataset
        image Dataset to write to disk
    outfile : str
        output filename, generally ends in .zarr
    
    Returns
    -------
    """
    from numcodecs import Blosc
    from itertools import cycle
    
    outfile = os.path.expanduser(outfile)
    compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
    encoding = dict(zip(list(ds.data_vars), cycle([{'compressor':compressor}])))
    
    ds.to_zarr(outfile, mode='w', encoding=encoding)


