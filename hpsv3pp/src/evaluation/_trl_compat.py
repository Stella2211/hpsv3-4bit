"""Compatibility shim: trl>=1.10 removed `get_kbit_device_map` (still imported,
unconditionally, by the upstream HPSv3++ repo's hpsv3/trainer/common.py at
module load time). Rather than patch the upstream submodule, monkey-patch
it back in before anything imports `hpsv3`.

Old trl implementation (unchanged behavior, works the same on newer trl):
returns a single-GPU device map pinned to the current process's device.
"""

import torch
import trl


def _get_kbit_device_map():
    if torch.cuda.is_available():
        from accelerate import PartialState

        return {"": PartialState().process_index}
    return None


if not hasattr(trl, "get_kbit_device_map"):
    trl.get_kbit_device_map = _get_kbit_device_map
