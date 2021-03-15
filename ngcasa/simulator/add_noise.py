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

def add_noise():
    """
    .. todo::
        This function is not yet implemented
    
    Add noise to the DATA column of the MS

    Options :
    
    - Gaussian random noise of specified standard-deviation
    - Noise that accounts for delta-T and delta-NU of the observation.
    - Noise that accounts for different antenna collecting areas (from dish diameter meta-data)

    """