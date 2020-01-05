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

from dask.distributed import Client, LocalCluster

global_framework_client = None

########################
def InitializeFramework(workers=2, memory='8GB', processes=True, **kwargs):
    """
    Initialize the CNGI framework

    This sets up the processing environment such that all future calls to Dask
    Dataframes, arrays, etc will automatically use the scheduler that is configured
    here.

    Parameters
    ----------
    workers : int
        Number of processor cores to use, Default=2
    memory : str
        Max memory allocated to each worker in string format. Default='8GB'
    processes : bool
        Whether to use processes (True) or threads (False), Default=True
    threads_per_worker : int
        Only used if processes = True. Number of threads per python worker process, Default=1

    Returns
    -------
    distributed.client.Client
        Client from Dask Distributed for use by Dask objects
    """

    import socket

    if 'threads_per_worker' in kwargs.keys():
        tpw = kwargs['threads_per_worker']
    else: # enforce default of 1 thread per process
        tpw = 1
    
    global global_framework_client
    
    if global_framework_client != None:
        global_framework_client.close()
    
    # set up a cluster object to pass into Client
    # for now, only supporting single machine
    cluster = LocalCluster(n_workers=workers, threads_per_worker=tpw,
                           processes=processes, memory_limit=memory)

    global_framework_client = Client(cluster)
    
    print("Dask dashboard hosted at:")
    if processes == True:
        print(global_framework_client.cluster.dashboard_link)
    else:
        print(socket.gethostbyname(socket.gethostname()) + '/' + 
              str(global_framework_client.scheduler_info()['services']['dashboard']))
    
    return(global_framework_client)


#########################
def GetFrameworkClient():
    """
    Return the CNGI framework scheduler client

    Parameters
    ----------
    
    Returns
    -------
    distributed.client.Client
        Client from Dask Distributed for use by Dask objects
    """
    return (global_framework_client)
