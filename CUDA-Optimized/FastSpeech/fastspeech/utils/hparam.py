# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import yaml


def load_hparam(filepath):
    hparam_dict = dict()

    if not filepath:
        return hparam_dict

    stream = open(filepath, 'r')
    docs = yaml.load_all(stream)
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict


def merge_dict(new, default):
    if isinstance(new, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in new:
                new[k] = v
            else:
                new[k] = merge_dict(new[k], v)
    return new


class Dotdict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError as e:
            return None


class Hparam(Dotdict):

    __getattr__ = Dotdict.__getattr__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__

    def __init__(self, root_path):
        self.hp_root_path = root_path
        super(Hparam, self).__init__()

    def set_hparam(self, filename, hp_commandline=dict()):

        def get_hp(file_path):
            """
            It merges parent_yaml in yaml recursively.
            :param file_rel_path: relative hparam file path.
            :return: merged hparam dict.
            """
            hp = load_hparam(file_path)
            if 'parent_yaml' not in hp:
                return hp
            parent_path = os.path.join(self.hp_root_path, hp['parent_yaml'])

            if parent_path == file_path:
                raise Exception('To set myself({}) on parent_yaml is not allowed.'.format(file_path))

            base_hp = get_hp(parent_path)
            hp = merge_dict(hp, base_hp)
            
            return hp

        hparam_path = os.path.join(self.hp_root_path, filename)

        hp = get_hp(hparam_path)
        hp = merge_dict(hp_commandline, hp)

        hp = Dotdict(hp)

        for k, v in hp.items():
            setattr(self, k, v)