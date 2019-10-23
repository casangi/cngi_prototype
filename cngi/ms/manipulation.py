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

########################
def chanaverage(df, width=1):
    """
    .. todo::
        This function is not yet implemented

    Average data across channels
    
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
    .. todo::
        This function is not yet implemented

    Set the metadata contents of an MS Dataframe
    
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
    .. todo::
        This function is not yet implemented

    Perform a running mean across the spectral axis with a triangle as a smoothing kernel
    
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
    .. todo::
        This function is not yet implemented

    Recalulate UVW and shift data to new phase center
    
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
    .. todo::
        This function is not yet implemented

    Transform channel labels and visibilities to a spectral reference frame which is appropriate for analysis, e.g. from TOPO to LSRK or to correct for doppler shifts throughout the time of observation
        
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
    .. todo::
        This function is not yet implemented

    Average data across time axis
    
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
    .. todo::
        This function is not yet implemented

    Estimate continuum emission and subtract it from visibilities
    
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
    .. todo::
        This function is not yet implemented

    Fit simple analytic source component models directly to visibility data
    
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
