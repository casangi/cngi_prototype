#   Copyright 2020 AUI, Inc. Washington DC, USA
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

def add_meta_data():
    """
    .. todo::
        This function is not yet implemented
    
    Make an empty dataset, or append to an existing vis_dataset.
    Each call to this method will construct meta-data (UVW values, time values, etc) for one scan/spw_pol/field.

    A dataset containing multiple scans, fields(pointings) or spectral window and pol settings may be
    constructed by calling this method in a sequence.

    Inputs :
    
    - Observation phase center
    - Spectral window and polarization setup parameters (chanwidth, nchan, startchan, etc...)
    - Array configuration
    - Integration length and Time-range of one scan.

    Output :
    
    - A new or appended XDS with new/appended meta-data information

   TBD - Split this into smaller methods as needed.
    
    Can follow the casa6.casatools.simulator tool interface for all methods required before the actual calculation of visibility values.
    May be best to make them simulation_utils methods though.

    """