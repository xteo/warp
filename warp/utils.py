# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Utilities supporting Warp's high-level workflows."""

# isort: skip_file

# category: Array Operations

from warp._src.utils import array_cast as array_cast
from warp._src.utils import array_inner as array_inner
from warp._src.utils import array_scan as array_scan
from warp._src.utils import array_sum as array_sum


# category: Sorting

from warp._src.utils import radix_sort_pairs as radix_sort_pairs
from warp._src.utils import runlength_encode as runlength_encode
from warp._src.utils import segmented_sort_pairs as segmented_sort_pairs


# category: Graph Coloring (deferred to __getattr__ to save ~2ms import time)


# category: Misc

from warp._src.utils import create_warp_function as create_warp_function


# TODO: Remove after cleaning up the public API.

from warp._src import utils as _utils

_coloring_names = frozenset({
    "GraphColoringAlgorithm",
    "graph_coloring_assign",
    "graph_coloring_balance",
    "graph_coloring_get_groups",
})


def __getattr__(name):
    if name in _coloring_names:
        import warp._src.coloring as _coloring  # noqa: PLC0415

        for _n in _coloring_names:
            globals()[_n] = getattr(_coloring, _n)
        return globals()[name]

    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_utils, "warp", name)
