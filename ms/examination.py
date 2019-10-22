###############
def setmeta(df, fields={}):
    """
    Set the metadata contents of an MS Dataframe

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    df : Dask Dataframe
        input MS to view
    fields : dict
        dictionary of column:value pairs to set in the metadata

    Returns
    -------
    Dask Dataframe
        New Dataframe with modified metadata
    """
    return {}


#############
def viewmeta(df):
    """
    View the metadata of an MS Dataframe

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    df : Dask Dataframe
        input MS to view

    Returns
    -------
    dict
        Summary information
    """
    return {}


#############
def summarizeDF(df, field=None, spw=None, timerange=None, uvrange=None, antenna=None, scan=None):
    """
    Summarize the contents of an MS Dataframe

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    df : Dask Dataframe
        input MS to summarize
    field : int
        field selection. If None, use all fields
    spw : int
        spw selection. If None, use all spws
    timerange : int
        time selection. If None, use all times
    uvrange : int
        uvrange selection. If None, use all uvranges
    antenna : int
        antenna selection. If None, use all antennas
    scan : int
        scan selection. If None, use all scans

    Returns
    -------
    dict
        Summary information
    """
    return {}


###################
def summarizeFile(infile):
    """
    Summarize the contents of an Apache Parquet MS

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    infile : Dask Dataframe
        input filename of Parquet MS 

    Returns
    -------
    dict
        Summary information
    """
    return {}
