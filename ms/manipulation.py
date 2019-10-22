########################
def chanaverage(df, width=1):
    """
    Average data across channels

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    df : Dask Dataframe
        input MS
    width : int
        number of adjacent channels to average. Default=1 (no change)

    Returns
    -------
    Dask Dataframe
        New Dataframe with averaged data
    """
    return {}



########################
def joinspw(df1, df2):
    """
    Set the metadata contents of an MS Dataframe

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    df1 : Dask Dataframe
        first MS DF to join
    df2 : Dask Dataframe
        second MS DF to join

    Returns
    -------
    Dask Dataframe
        New Dataframe with combined contents
    """
    return {}



##########################
def hanningsmooth(df, field=None, spw=None, timerange=None, uvrange=None, antenna=None, scan=None):
    """
    Perform a running mean across the spectral axis with a triangle as a smoothing kernel

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    df : Dask Dataframe
        input MS
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
    Dask Dataframe
        New Dataframe with updated data
    """
    return {}



#############################################
def recalculateUVW(df, field=None, refcode=None, reuse=True, phasecenter=None):
    """
    Recalulate UVW and shift data to new phase center

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    df : Dask Dataframe
        input MS
    field : int
        fields to operate on. None = all
    refcode : str
        reference frame to convert UVW coordinates to
    reuse : bool
        base UVW calculation on the old values
    phasecenter : float
        direction of new phase center. None = no change

    Returns
    -------
    Dask Dataframe
        New Dataframe with updated data
    """
    return {}



###############################################
def regridSPW(df, field=None, spw=None, timerange=None, uvrange=None, antenna=None, scan=None, mode='channel', nchan=None, start=0, width=1, interpolation='linear', phasecenter=None, restfreq=None, outframe=None, veltype='radio'):
    """
    Transform channel labels and visibilities to a spectral reference frame which is appropriate for analysis, e.g. from TOPO to LSRK or to correct for doppler shifts throughout the time of observation
    
    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    df : Dask Dataframe
        input MS
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
    mode : str
        regridding mode
    nchan : int
        number of channels in output spw. None=all
    start : int
        first input channel to use
    width : int
        number of input channels to average
    interpolation : str
        spectral interpolation method
    phasecenter : int
        image phase center position or field index
    restfreq : float
        rest frequency
    outframe : str
        output frame, None=keep input frame
    veltype : str
        velocity definition

    Returns
    -------
    Dask Dataframe
        New Dataframe with updated data
    """
    return {}



################################################
def timeaverage(df, timebin=0.0, timespan=None, maxuvdistance=0.0):
    """
    Average data across time axis

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    df : Dask Dataframe
        input MS
    timebin : float
        Bin width for time averaging (in seconds). Default 0.0
    timespan : str
        Span the timebin. Allowed values are None, 'scan', 'state' or 'both'

    Returns
    -------
    Dask Dataframe
        New Dataframe with averaged data
    """
    return {}




##################################################
def uvcontsub(df, field=None, fitspw=None, combine=None, solint='int', fitorder=0):
    """
    Estimate continuum emission and subtract it from visibilities

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    df : Dask Dataframe
        input MS
    field : int
        field selection. If None, use all fields
    fitspw : int
        spw:channel selection for fitting
    combine : str
        data axis to combine for the continuum
    solint : str
        continuum fit timescale
    fitorder : int
        polynomial order for the fits

    Returns
    -------
    Dask Dataframe
        New Dataframe with updated data
    """
    return {}




#####################################################
def uvmodelfit(df, field=None, spw=None, timerange=None, uvrange=None, antenna=None, scan=None, niter=5, comptype='p', sourcepar=[1,0,0], varypar=[]):
    """
    Fit simple analytic source component models directly to visibility data

    .. todo::
        This function is not yet implemented
    
    Parameters
    ----------
    df : Dask Dataframe
        input MS
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
    niter : int
        number of fitting iteractions to execute
    comptype : str
        component type (p=point source, g=ell. gauss. d=ell. disk)
    sourcepar : list
        starting fuess (flux, xoff, yoff, bmajaxrat, bpa)
    varypar : list
        parameters that may vary in the fit

    Returns
    -------
    Dask Dataframe
        New Dataframe with updated data
    """
    return {}
