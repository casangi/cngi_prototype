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
    
    ddf = dd.read_parquet(infile+'/'+str(ddi), engine='pyarrow', columns=columns)
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
    
    # need to manually remove existing parquet file (if any)
    if not append:
        tmp = os.system("rm -fr " + outfile)
    else: # still need to remove existing ddi (if any)
        tmp = os.system("rm -fr " + outfile+'/'+str(ddi))
    
    dd.to_parquet(df, outfile+'/'+str(ddi), engine='pyarrow', compression='snappy', 
                  write_metadata_file=True, compute=True)
