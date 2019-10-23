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

from dask.distributed import Client

global_framework_client = None

########################
def InitializeFramework(workers=2, memory='8GB', processes=True):
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
        Max memory per core to use. String format eg '8GB'
    processes : bool
        Whether to use processes (True) or threads (False), Default=True

    Returns
    -------
    Dask Distributed Client
        Client from Dask Distributed for use by Dask objects
    """
    global global_framework_client
    
    if global_framework_client != None:
        global_framework_client.close()
    
    global_framework_client = Client(processes=processes, n_workers=workers, 
                                     threads_per_worker=1, memory_limit=memory)
    
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
