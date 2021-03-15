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