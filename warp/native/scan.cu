/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "warp.h"

#include "cuda_util.h"
#include "scan.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/device/device_scan.cuh>
#include <unordered_map>

// persistent temporary buffer for scan, keyed by CUDA stream
struct ScanTemp {
    void* mem = NULL;
    size_t size = 0;
};

static std::unordered_map<void*, ScanTemp> g_scan_temp_map;

template <typename T> void scan_device(const T* values_in, T* values_out, int n, bool inclusive)
{
    ContextGuard guard(wp_cuda_context_get_current());

    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    // compute temporary memory required
    size_t scan_temp_size;
    if (inclusive) {
        check_cuda(cub::DeviceScan::InclusiveSum(NULL, scan_temp_size, values_in, values_out, n));
    } else {
        check_cuda(cub::DeviceScan::ExclusiveSum(NULL, scan_temp_size, values_in, values_out, n));
    }

    // reuse persistent temp buffer (grow if needed)
    ScanTemp& temp = g_scan_temp_map[stream];
    if (scan_temp_size > temp.size) {
        if (temp.mem) wp_free_device(WP_CURRENT_CONTEXT, temp.mem);
        temp.mem = wp_alloc_device(WP_CURRENT_CONTEXT, scan_temp_size);
        temp.size = scan_temp_size;
    }

    // scan
    if (inclusive) {
        check_cuda(cub::DeviceScan::InclusiveSum(temp.mem, scan_temp_size, values_in, values_out, n, stream));
    } else {
        check_cuda(cub::DeviceScan::ExclusiveSum(temp.mem, scan_temp_size, values_in, values_out, n, stream));
    }
}

template void scan_device(const int*, int*, int, bool);
template void scan_device(const float*, float*, int, bool);
