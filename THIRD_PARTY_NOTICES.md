# Third-party notices

## MizzenAI/HPSv3 (MIT)

`hpsv3/src/evaluation/hpsv3_model.py` contains a model class
(`Qwen2VLRewardModelBT`) adapted from
[MizzenAI/HPSv3](https://github.com/MizzenAI/HPSv3)
(`hpsv3/model/qwen2vl_trainer.py`, upstream commit
`bd0c5fcb5f587617b0169c07222ab78d01e2f3c2`), which is distributed under the
MIT License. The upstream copyright notice and license text are reproduced
below.

```
MIT License

Copyright (c) 2024 HPSv3 Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Hugging Face TRL (Apache-2.0)

`hpsv3pp/src/evaluation/_trl_compat.py` adapts TRL's
`get_kbit_device_map` helper. The reviewed upstream reference is
[`trl/trainer/utils.py` at v0.12.0](https://github.com/huggingface/trl/blob/14ef1aba152fddbc5a58f3a8a712b6e509e7e69d/trl/trainer/utils.py),
commit `14ef1aba152fddbc5a58f3a8a712b6e509e7e69d`.

Copyright 2022 The HuggingFace Team. All rights reserved.
Modifications Copyright (c) 2026 stella.

The adaptation keeps CUDA support, removes the XPU branch, imports
`PartialState` lazily, uses `process_index` instead of `local_process_index`,
renames the helper and installs it only if TRL does not provide it.
It is intended for single-GPU inference.

This file is licensed under Apache-2.0, separately from the repository's MIT
license. The full upstream license is included in
[licenses/TRL-Apache-2.0.txt](licenses/TRL-Apache-2.0.txt).

## PlantPotatoOnMoon/HPSv3-PlusPlus (no license file)

The upstream HPSv3++ code repository publishes no LICENSE file. The legacy
project references it as a pinned git submodule. The reusable runtime also
contains adapted reward classes under `src/hpsv3_4bit/hpsv3pp/`; these carry
`LicenseRef-HPSv3PlusPlus-Permission-Unconfirmed` and must not be publicly
distributed until permission is documented. See the runtime's
`THIRD_PARTY_NOTICES.md` and `PROVENANCE.md` for its source revision and scope.
