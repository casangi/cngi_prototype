#  CASA Next Generation Infrastructure
#  Copyright (C) 2021 AUI, Inc. Washington DC, USA
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
this module will be included in the api
"""
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
