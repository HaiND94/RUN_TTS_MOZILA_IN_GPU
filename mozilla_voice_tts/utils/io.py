import re
import json
import traceback
from shutil import copyfile

class AttrDict(dict):
    """A custom dict which converts dict keys
    to class attributes"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(config_path):
    """Load config files and discard comments

    Args:
        config_path (str): path to config file.
    """
    config = AttrDict()
    with open(config_path, "r") as f:
        input_str = f.read()
    # handle comments
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    try:
        data = json.loads(input_str)
    except:
        traceback.print_exc()
    config.update(data)
    return config


def copy_config_file(config_file, out_path, new_fields):
    """Copy config.json to training folder and add
    new fields.

    Args:
        config_file (str): path to config file.
        out_path (str): output path to copy the file.
        new_fields (dict): new fileds to be added or edited
            in the config file.
    """
    config_lines = open(config_file, "r").readlines()
    # add extra information fields
    for key, value in new_fields.items():
        if isinstance(value, str):
            new_line = '"{}":"{}",\n'.format(key, value)
        else:
            new_line = '"{}":{},\n'.format(key, value)
        config_lines.insert(1, new_line)
    config_out_file = open(out_path, "w")
    config_out_file.writelines(config_lines)
    config_out_file.close()
