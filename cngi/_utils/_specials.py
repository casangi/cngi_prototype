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

#################################
# Helper File
#
# Not exposed in API
#
#################################

""" Describe anything in the standard ngCASA dataset format that needs special treatment.

General rules, to which this file provides exceptions:
1. Attributes are fleeting. Any cngi/ngcasa function should expect any attribute to be overwritten at any time."""

import copy

class specials:
    """ Don't access these variables directly. Use one of the associated functions instead in order to avoid accidentally changing these variables. """
    _attributes = {
        'ddi': {
            'desc': 'Data Description Identifier for a dataset. Should be unique to each dataset. Can be a number or a string.'
        },
        # 'units': {},
        # 'topo': {}
    }

def attrs():
    """ Retrieves a copy of the attribute names that require special treatment and should not be overwritten without careful consideration. """
    return copy.deepcopy(specials._attributes)