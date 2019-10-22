#############################################
def read_pq(infile, ddi=None):
    """
    Read Apache Parquet format MS from disk

    .. todo::
        This function is not yet implemented 

    Parameters
    ----------
    infile : str
        input MS filename
    ddi : int
        Data Description ID in MS to read. If None, defaults to 0

    Returns
    -------
    Dask Dataframe
        New Dataframe of MS contents
    """
    return {}



#############################################
def write_pq(df, outfile):
    """
    Write MS dataframe to Apache Parquet format on disk

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    df : Dask Dataframe
        MS dataframe to write to disk
    outfile : str
        output filename

    Returns
    -------
    bool
        Success
    """
    return True
