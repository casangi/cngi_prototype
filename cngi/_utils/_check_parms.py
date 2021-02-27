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
import numpy as np

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
        elif (dict in acceptable_data_types):
            parms_passed = True

            if default is None:
                print('######### ERROR:Dictionary parameters must have a default. Please report bug.')
                return False
            #print('is a dict',default)
            for default_element in default:
                if default_element in parm_dict[string_key]:
                    #print('1.*******')
                    #print(parm_dict[string_key], default_element, [type(default[default_element])], default[default_element])
                    if not(_check_parms(parm_dict[string_key], default_element, [type(default[default_element])], default=default[default_element])): parms_passed = False
                    #print('2.*******')
                else:
                    #print('parm_dict',default_element,string_key)
                    parm_dict[string_key][default_element] = default[default_element]
                    print('Setting default', string_key, '[\'',default_element,'\']', ' to ', default[default_element])
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
            #print(parm_dict, string_key,  default)
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
    
    

def _check_sel_parms(xds,sel_parms,new_or_modified_data_variables={},required_data_variables={},drop_description_out_keys=[],append_to_in_id=False):
    """
    
    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        Input vis.zarr multi dataset.
    sel_parms : dictionary
    Returns
    -------
    psf_dataset : xarray.core.dataset.Dataset
    """
    import copy
    
    #If an empty xds is supplied
    if not('data_groups' in xds.attrs):
        if 'data_group_out_id' in sel_parms:
            xds.attrs['data_groups'] = [{str(sel_parms['data_group_out_id']):{'id':str(sel_parms['data_group_out_id'])}}]
        else:
            xds.attrs['data_groups'] = [{'1':{'id':'1'}}]
        append_to_in_id = True
    ######################
    
    data_group_ids = np.array(list(xds.data_groups[0])).astype(int)
    
    if 'data_group_in_id' in sel_parms:
        sel_parms['data_group_in'] = {'id':str(sel_parms['data_group_in_id'])}
    
    if 'data_group_out_id' in sel_parms:
        sel_parms['data_group_out'] = {'id':str(sel_parms['data_group_out_id'])}
        
    for mdv in new_or_modified_data_variables:
        if mdv in sel_parms:
            if 'data_group_out' in sel_parms:
                sel_parms['data_group_out'] = {**sel_parms['data_group_out'],**{mdv:sel_parms[mdv]}}
            else:
                sel_parms['data_group_out'] = {mdv:sel_parms[mdv]}
                
    if 'data_group_in' in sel_parms:
        if 'id' in sel_parms['data_group_in']:
            assert (int(sel_parms['data_group_in']['id']) in data_group_ids), "######### ERROR: data_group_in id does not exist in " + sel_parms['xds']
            sel_parms['data_group_in'] = copy.deepcopy(xds.data_groups[0][str(sel_parms['data_group_in']['id'])])
            print("Setting data_group_in  to ",sel_parms['data_group_in'])
            data_group_in_default = sel_parms['data_group_in']
            
            if (append_to_in_id==True):
                new_data_group_id = sel_parms['data_group_in']['id']
            else:
                new_data_group_id = np.max(data_group_ids) + 1
        else:
            data_group_in_default = list(xds.data_groups[0])[0]
            data_group_in_default['id'] = -1 #Custom
    else:
        data_group_in_default = list(xds.data_groups[0].values())[0] #Choose id 1
        
        if (append_to_in_id==True):
            new_data_group_id = data_group_in_default['id']
        else:
            new_data_group_id = np.max(data_group_ids) + 1
            
    if 'data_group_out_id' in sel_parms:
        new_data_group_id = sel_parms['data_group_out_id']
        
    #If new coord is created new_or_modified_data_variables should contain all data variables that will be present in the output
    if 'data_group_out' in sel_parms:
        temp_sel_parms = copy.deepcopy(sel_parms)
        for sel in temp_sel_parms['data_group_out']:
            if not(sel in new_or_modified_data_variables) and (sel!='id'):
                sel_parms['data_group_out'].pop(sel,None)
                
    data_group_in_default = {**required_data_variables,**data_group_in_default}
    
    data_group_out_defaults = {**{'id':str(new_data_group_id)} , **new_or_modified_data_variables}
    #This is for type add_to_xds
    #print(data_group_in_default,data_group_out_defaults)
    data_group_out_defaults = {**data_group_in_default,**data_group_out_defaults} #{**x, **y} merges dicts with y taking precedence for repeated entries.
    
    sel_defaults = {'data_group_in':data_group_in_default,'data_group_out':data_group_out_defaults}
    assert(_check_sub_sel_parms(sel_parms,sel_defaults)), "######### ERROR: sel_parms checking failed"
    
    sel_check = {'data_group_in':sel_parms['data_group_in']}
    assert(_check_existence_sel_parms(xds,sel_check)), "######### ERROR: sel_parms checking failed"
    
    #data_group_id = sel_parms['data_group_out']['id']
    
    
    #print('%%%%%%%%%%')
    #print(sel_parms['data_group_in'], data_group_in_default)
    
    
    for mdv in new_or_modified_data_variables:
        for d_id in xds.data_groups[0]:
            #print(xds.data_groups[0][d_id][modified_data_variable])
            
            if mdv in xds.data_groups[0][d_id]:
                assert (sel_parms['data_group_out']['id'] == d_id) or (xds.data_groups[0][d_id][mdv] != sel_parms['data_group_out'][mdv]), "Data variables, that are modified by the function, can not be replaced if they are referenced in another data_group"
            #if (sel_parms['data_group_out']['id'] != d_id) and (xds.data_groups[0][d_id][modified_data_variable] == sel_parms['data_group_out'][modified_data_variable]):
            #    print(sel_parms['data_group_out']['id'], d_id)
        
    #for drop_description_out_keys
    #print('AAAAA',sel_parms)
    return True
    
    
    
def _check_sub_sel_parms(sel_parms,select_defaults):
    parms_passed = True
    for sel in select_defaults:
        if not(_check_parms(sel_parms, sel, [type(select_defaults[sel])], default=select_defaults[sel])): parms_passed = False
    return parms_passed

'''
def _check_sel_parms(sel_parms,select_defaults):
    parms_passed = True
    for sel_def in select_defaults:
        if isinstance(select_defaults[sel_def], dict):
            if sel_def in sel_parms:
                for sub_sel_def in select_defaults[sel_def]:
                        #print(sub_sel_def,select_defaults[sel_def])
                        #print(sel_parms[sel_def], sub_sel_def, select_defaults[sel_def][sub_sel_def])
                        if not(_check_parms(sel_parms[sel_def], sub_sel_def, [str], default=select_defaults[sel_def][sub_sel_def])): parms_passed = False
            else:
                sel_parms[sel_def] = select_defaults[sel_def]
                print ('Setting default', string_key, ' to ', parm_dict[string_key])
        else:
            if not(_check_parms(sel_parms, sel_def, [str], default=select_defaults[sel_def])): parms_passed = False
    return parms_passed
'''

def _check_existence_sel_parms(dataset, sel_parms):
    parms_passed = True
    for sel in sel_parms:
        if isinstance(sel_parms[sel], dict):
            if sel != 'properties':
                _check_existence_sel_parms(dataset, sel_parms[sel])
        else:
            if (sel != 'id') and (sel != 'properties'):
                if not(_check_dataset(dataset,sel_parms[sel])): parms_passed = False
    return parms_passed
