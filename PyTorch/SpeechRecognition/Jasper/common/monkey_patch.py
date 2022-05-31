import re
import warnings

import torch


major, minor, *_ = re.search('(\d+)\.(\d+)', torch.__version__).groups()

if int(major) >= 1 and int(minor) >= 12:
    # Mutes 'UserWarning: positional arguments and argument "destination"
    # are deprecated. nn.Module.state_dict will not accept them in the future.'

    def state_dict(self, *args, **kwargs):
        warnings.filterwarnings("ignore")
        ret = self._state_dict(*args, **kwargs)
        warnings.filterwarnings("default")
        return ret

    torch.nn.Module._state_dict = torch.nn.Module.state_dict
    torch.nn.Module.state_dict = state_dict
