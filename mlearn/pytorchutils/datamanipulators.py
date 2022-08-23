import json

import datetime
import os
import shutil

def create_folder(prefix, add_time=True, add_str=None, delete=False):
    additional_str = ''
    if delete is True:
        if os.path.exists(prefix):
            shutil.rmtree(prefix)
        os.makedirs(prefix)
    folder = prefix
    if add_time is True:
        # additional_str has a form such as '170903_220351'
        additional_str += datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        if add_str is not None:
            folder += '/' + additional_str + '_' + add_str
        else:
            folder += '/' + additional_str
    if delete is True:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    os.makedirs(folder)
    return folder

    
def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + k)
        else:
            flatted[start + sep + k] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, k)
        else:
            flatted[k] = v
    return flatted