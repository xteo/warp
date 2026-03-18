# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

import warp as wp

wp.clear_kernel_cache()


@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=wp.vec2i),
    spring_params: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    idx = spring_indices[tid]
    i = idx[0]
    j = idx[1]

    params = spring_params[tid]
    rest = params[0]
    ke = params[1]
    kd = params[2]

    xi = x[i]
    xj = x[j]

    xij = xi - xj

    l = wp.length(xij)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = xij * l_inv

    c = l - rest

    vi = v[i]
    vj = v[j]
    dcdt = wp.dot(dir, vi - vj)

    # damping based on relative velocity.
    fs = dir * (ke * c + kd * dcdt)

    wp.atomic_sub(f, i, fs)
    wp.atomic_add(f, j, fs)


@wp.kernel
def integrate_particles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    w: wp.array(dtype=float),
    dt: float,
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]
    f0 = f[tid]
    inv_mass = w[tid]

    g = wp.vec3()

    # treat particles with inv_mass == 0 as kinematic
    if inv_mass > 0.0:
        g = wp.vec3(0.0, 0.0 - 9.81, 0.0)

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + g) * dt
    x1 = x0 + v1 * dt

    x[tid] = x1
    v[tid] = v1

    # clear forces
    f[tid] = wp.vec3()


class WpIntegrator:
    def __init__(self, cloth, device):
        self.device = wp.get_device(device)

        with wp.ScopedDevice(self.device):
            self.positions = wp.from_numpy(cloth.positions, dtype=wp.vec3)
            self.positions_host = wp.from_numpy(cloth.positions, dtype=wp.vec3, device="cpu")
            self.invmass = wp.from_numpy(cloth.inv_masses, dtype=float)

            self.velocities = wp.zeros(cloth.num_particles, dtype=wp.vec3)
            self.forces = wp.zeros(cloth.num_particles, dtype=wp.vec3)

            # Pack spring indices as vec2i and params as vec3 to reduce
            # memory transactions from 6 array reads to 2 per spring
            indices_packed = cloth.spring_indices.reshape(-1, 2)
            self.spring_indices = wp.from_numpy(indices_packed, dtype=wp.vec2i)
            params_packed = np.stack(
                [cloth.spring_lengths, cloth.spring_stiffness, cloth.spring_damping], axis=1
            ).astype(np.float32)
            self.spring_params = wp.from_numpy(params_packed, dtype=wp.vec3)

        self.cloth = cloth

    def simulate(self, dt, substeps):
        sim_dt = dt / substeps

        for _s in range(substeps):
            wp.launch(
                kernel=eval_springs,
                dim=self.cloth.num_springs,
                inputs=[
                    self.positions,
                    self.velocities,
                    self.spring_indices,
                    self.spring_params,
                    self.forces,
                ],
                outputs=[],
                device=self.device,
            )

            # integrate
            wp.launch(
                kernel=integrate_particles,
                dim=self.cloth.num_particles,
                inputs=[self.positions, self.velocities, self.forces, self.invmass, sim_dt],
                outputs=[],
                device=self.device,
            )

        # copy data back to host
        if self.device.is_cuda:
            wp.copy(self.positions_host, self.positions)
            wp.synchronize()

            return self.positions_host.numpy()

        else:
            return self.positions.numpy()
