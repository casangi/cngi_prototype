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
"""
this module will be included in the api
"""

def flag_summary(vis_dataset, summary_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
        
    Return the vis_dataset with a attribute dictionary containing metrics to assess flagging quality
    
    Type 1 - Flag counts

    Flag ratio = n_flags/n_total along multiple axes and different levels of granularity.
    
    Type 2 - Flag validity
    
    Compare statistics of flagged versus unflagged visibility data
    and define a metric that quantifies the following.
     
    - Flagged data must have a higher mean than unflagged data
    
    - Unflagged data should follow Gaussian stats
    
    - Protect against under-flagging (less than 10%) or over-flagging (more than 70%)

    This option is for pipelines or applications that need to auto-tune autoflag parameters
    (An 'autotune' algorithm prototype exists).

    Example (this is very rudimentary) :
    
    score1 = (mean(flagged_data) - mean(unflagged_data))/mean(unflagged_data)
    
    score2 = ( max(unflagged_data)/ mean(unflagged_data) - 3.0 )
    
    score3 = (count(flagged)/count(total) - 0.1)*2 +  (count(flagged)/count(total) - 0.7)*2

    Inputs :
    
    (1) list of metrics to evaluate and return
    (2) array name for input flags. Default = FLAG

    Returns
    -------
    vis_dataset : xarray.core.dataset.Dataset
    """

