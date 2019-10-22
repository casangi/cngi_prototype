#########################################
def read_ms(infile, ddi=None):
    """
    Read MS format data in to Dask Dataframe

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    infile : str
        Input MS filename
    ddi : int
        Data Description ID in MS to read. If None, defaults to 0

    Returns
    -------
    Dask Dataframe
        New Dataframe of MS contents
    """
    return True



##########################################
def ms_to_pq(infile, outfile=None):
    """
    Convert Legacy MS to Apache Parquet format

    .. todo::
        This function is not yet implemented

    Parameters
    ----------
    infile : str
        Input MS filename
    outfile : str
        Output Parquet filename. If None, will use infile name with .pq extension

    Returns
    -------
    bool
        Success status
    """
    return True




###########################################
def pq_to_ms(infile, outfile=None):
    """
    Convert Apache Parquet MS to Legacy CASA MS format

    .. todo::
        This function is not yet implemented

    Parameters
    ----------
    infile : str
        Input Parquet filename
    outfile : str
        Output MS filename. If None, will use infile name with .ms extension

    Returns
    -------
    bool
        Success status
    """
    return True



###########################################
def asdm_to_pq(infile, outfile=None):
    """
    Convert ASDM format to Apache Parquet format (future)

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    infile : str
        Input ASDM filename
    outfile : str
        Output Parquet filename. If None, will use infile name with .pq extension

    Returns
    -------
    bool
        Success status
    """
    return True



###########################################
def pq_to_asdm(infile, outfile=None):
    """
    Convert Apache Parquet MS to ASDM format (future)

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    infile : str
        Input Parquet filename
    outfile : str
        Output ASDM filename. If None, will use infile name with .asdm extension

    Returns
    -------
    bool
        Success status
    """
    return True



###########################################
def fits_to_pq(infile, outfile=None):
    """
    Convert FITS format MS to Apache Parquet format (future)

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    infile : str
        Input FITS filename
    outfile : str
        Output Parquet filename. If None, will use infile name with .pq extension

    Returns
    -------
    bool
        Success status
    """
    return True



############################################
def pq_to_fits(infile, outfile=None):
    """
    Convert Apache Parquet MS to FITS format (future)

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    infile : str
        Input Parquet filename
    outfile : str
        Output FITS filename. If None, will use infile name with .fits extension

    Returns
    -------
    bool
        Success status
    """
    return True
