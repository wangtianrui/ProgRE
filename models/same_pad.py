# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""SamePad."""

from mindspore import nn


class SamePad(nn.Cell):
    """SamePad.

    Args:
        kernel_size (int): kernel size of conv.
        causal (bool): whether to remove causally.
            if True, just minus 1, else will check kernel size's parity.
            default False.
    """

    def __init__(self, kernel_size, causal=False):
        super(SamePad, self).__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def construct(self, x):
        if self.remove > 0:
            x = x[:, :, :-self.remove]
        return x
