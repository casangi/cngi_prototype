def ms_to_pq(infile, outfile=None)
    """
    Convert Legacy MS to Apache Parquet format

    Parameters
    ----------
    infile : str
        Input MS filename
    outfile : str
        Output Parquet filename. If None, will use infile name with .pq extension

    Returns
    -------
    None
    """
    tmp = infile

