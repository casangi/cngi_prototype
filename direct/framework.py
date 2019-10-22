from dask.distributed import Client

global_framework_client = None

########################
def InitializeFramework(workers=2, memory='8GB'):
    """
    Initialize the CNGI framework

    This sets up the processing environment such that all future calls to Dask
    Dataframes, arrays, etc will automatically use the scheduler that is configured
    here.

    Parameters
    ----------
    workers : int
        Number of processor cores to use
    memory : str
        Max memory per core to use. String format eg '8GB', default is '2GB'

    Returns
    -------
    Dask Distributed Client
        Client from Dask Distributed for use by Dask objects
    """
    global global_framework_client
    
    if global_framework_client != None:
        global_framework_client.close()
    
    global_framework_client = Client(n_workers=workers, threads_per_worker=1, memory_limit=memory)
    
    return(global_framework_client)


#########################
def GetFrameworkClient():
    """
    Return the CNGI framework scheduler client

    Parameters
    ----------
    
    Returns
    -------
    Dask Distributed Client
        Client from Dask Distributed for use by Dask objects
    """
    return (global_framework_client)
