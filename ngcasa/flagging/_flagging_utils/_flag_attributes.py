#   Copyright 2020-21 European Southern Observatory, ALMA partnership
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


def _ensure_flags_attr(xds):
    """
    Returns the name of the dataset attribute that holds info on flag variables.
    """
    flag_var = 'FLAG'
    flags_attr = 'flag_variables'
    if flags_attr not in xds.attrs:
        xds.attrs[flags_attr] = {flag_var: 'Default flags variable'}
    return flags_attr


def _add_descr(xds, add_name=None, descr=None):
    """
    Adds into the dict of flag variables a new flag variable and its description.
    """
    flags_attr = _ensure_flags_attr(xds)
    if add_name:
        xds.attrs[flags_attr][add_name] = descr
