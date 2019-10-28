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
def read_pq(infile, ddi=0, columns=None):
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
    import dask.dataframe as dd
    from cngi.direct import GetFrameworkClient
    
    if GetFrameworkClient() == None:
      print("*****Processing Framework is not initialized, call cngi.direct.InitializeFramework first!")
      return None
    
    ddf = dd.read_parquet(infile+'/'+str(ddi), engine='pyarrow', columns=columns, gather_statistics=False)
    return ddf



#############################################
def write_pq(df, outfile='ms.pq', ddi=0, append=False):
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
    from cngi.direct import GetFrameworkClient
    
    if GetFrameworkClient() == None:
      print("*****Processing Framework is not initialized, call cngi.direct.InitializeFramework first!")
      return None
    
    # need to manually remove existing parquet file (if any)
    if not append:
        tmp = os.system("rm -fr " + outfile)
    else: # still need to remove existing ddi (if any)
        tmp = os.system("rm -fr " + outfile+'/'+str(ddi))
    
    dd.to_parquet(df, outfile+'/'+str(ddi), engine='pyarrow', compression='snappy', 
                  write_metadata_file=True, compute=True)




#############################################
def read_ncdf(infile, ddi=0):
    """
    Read xarray NetCDF format MS from disk
    
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
    from xarray import open_mfdataset
    from cngi.direct import GetFrameworkClient
    
    if GetFrameworkClient() == None:
      print("*****Processing Framework is not initialized, call cngi.direct.InitializeFramework first!")
      return None
    
    xdf = open_mfdataset(infile+'/'+str(ddi)+'/*.nc', combine='by_coords', parallel=True)
    return xdf


