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

def _check_parms(parm_dict, string_key, acceptable_data_types, acceptable_data = None, acceptable_range = None, list_acceptable_data_types=None, list_len=None, default=None):
    """

    Parameters
    ----------
    parm_dict: dict
        The dictionary in which the a parameter will be checked
    string_key :
    acceptable_data_types : list
    acceptable_data : list
    acceptable_range : list (length of 2)
    list_acceptable_data_types : list
    list_len : int
        If list_len is -1 than the list can be any length.
    default :
    Returns
    -------
    parm_passed : bool
        
    """
    
    import numpy as np

    if string_key in parm_dict:
        if (list in acceptable_data_types) or (np.array in acceptable_data_types):
            if (len(parm_dict[string_key]) != list_len) and (list_len != -1):
                print('######### ERROR:Parameter ', string_key, 'must be a list of ', list_acceptable_data_types, ' and length', list_len , '. Wrong length.')
                return False
            else:
                list_len = len(parm_dict[string_key])
                for i in range(list_len):
                    type_check = False
                    for lt in list_acceptable_data_types:
                        if isinstance(parm_dict[string_key][i], lt):
                            type_check = True
                    if not(type_check):
                          print('######### ERROR:Parameter ', string_key, 'must be a list of ', list_acceptable_data_types, ' and length', list_len, '. Wrong type.')
                          return False
                          
                    if acceptable_data is not None:
                        if not(parm_dict[string_key][i] in acceptable_data):
                            print('######### ERROR: Invalid', string_key,'. Can only be one of ',acceptable_data,'.')
                            return False
                            
                    if acceptable_range is not None:
                        if (parm_dict[string_key][i] < acceptable_range[0]) or (parm_dict[string_key][i] > acceptable_range[1]):
                            print('######### ERROR: Invalid', string_key,'. Must be within the range ',acceptable_range,'.')
                            return False
        else:
            type_check = False
            for adt in acceptable_data_types:
                if isinstance(parm_dict[string_key], adt):
                    type_check = True
            if not(type_check):
                print('######### ERROR:Parameter ', string_key, 'must be of type ', acceptable_data_types)
                return False
                    
            if acceptable_data is not None:
                if not(parm_dict[string_key] in acceptable_data):
                    print('######### ERROR: Invalid', string_key,'. Can only be one of ',acceptable_data,'.')
                    return False
                    
            if acceptable_range is not None:
                if (parm_dict[string_key] < acceptable_range[0]) or (parm_dict[string_key] > acceptable_range[1]):
                    print('######### ERROR: Invalid', string_key,'. Must be within the range ',acceptable_range,'.')
                    return False
    else:
        if default is not None:
            parm_dict[string_key] =  default
            print ('Setting default', string_key, ' to ', parm_dict[string_key])
        else:
            print('######### ERROR:Parameter ', string_key, 'must be specified')
            return False
            
    return True

def _check_dataset(vis_dataset,data_variable_name):
    try:
        temp = vis_dataset[data_variable_name]
    except:
        print ('######### ERROR Data array ', data_variable_name,'can not be found in dataset.')
        return False
    return True
    
    
def _check_storage_parms(storage_parms,default_outfile,graph_name):
    from numcodecs import Blosc
    parms_passed = True
    
    if not(_check_parms(storage_parms, 'to_disk', [bool], default=False)): parms_passed = False
    if not(_check_parms(storage_parms, 'graph_name', [str],default=graph_name)): parms_passed = False
    
    if storage_parms['to_disk'] == True:
        if not(_check_parms(storage_parms, 'outfile', [str],default=default_outfile)): parms_passed = False
        if not(_check_parms(storage_parms, 'append', [bool],default=False)): parms_passed = False
        if not(_check_parms(storage_parms, 'compressor', [Blosc],default=Blosc(cname='zstd', clevel=2, shuffle=0))): parms_passed = False
        if not(_check_parms(storage_parms, 'chunks_on_disk', [dict],default={})): parms_passed = False
        if not(_check_parms(storage_parms, 'chunks_return', [dict],default={})): parms_passed = False
        
    return parms_passed



def _check_sel_parms(sel_parms,select_defaults):
    parms_passed = True
    for sel in select_defaults:
        if not(_check_parms(sel_parms, sel, [str], default=select_defaults[sel])): parms_passed = False
    return parms_passed


def _check_existence_sel_parms(dataset, sel_parms):
    parms_passed = True
    for sel in sel_parms:
        if not(_check_dataset(dataset,sel_parms[sel])): parms_passed = False
    return parms_passed
