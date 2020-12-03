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