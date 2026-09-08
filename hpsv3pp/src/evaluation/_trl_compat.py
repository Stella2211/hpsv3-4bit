# Copyright 2022 The HuggingFace Team. All rights reserved.
# Modifications Copyright (c) 2026 stella.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See
# licenses/TRL-Apache-2.0.txt at the repository root for the full license.
# This file is provided on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied. See the license for permissions
# and limitations.

"""Compatibility shim: trl>=1.10 removed `get_kbit_device_map` (still imported,
unconditionally, by the upstream HPSv3++ repo's hpsv3/trainer/common.py at
module load time). Rather than patch the upstream submodule, monkey-patch
it back in before anything imports `hpsv3`.

Based on TRL's `get_kbit_device_map`; reviewed upstream source: v0.12.0,
commit 14ef1aba152fddbc5a58f3a8a712b6e509e7e69d, trl/trainer/utils.py.
Source and license details are in THIRD_PARTY_NOTICES.md.

This adaptation supports CUDA only, imports PartialState lazily, uses
process_index instead of local_process_index, and installs the renamed helper
only when TRL does not provide it. Intended for single-GPU inference.
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
