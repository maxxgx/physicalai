# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Metal compute kernels for :mod:`physicalai_mujoco_plugin.real2sim.metal_splat`.

Compiled at run time with ``torch.mps.compile_shader``. ``torch`` binds a Python ``float`` as a
``constant float&``, a list of floats as ``constant float*`` and a list of ints as
``constant long*``; tensors bind as ``device`` buffers.

The maths follows gsplat's classic (non-antialiased) projection and rasteriser, which is the INRIA
reference; see the module docstring of ``metal_splat`` for the few deliberate differences.
"""

TILE = 16
"""Tile edge in pixels; one threadgroup of ``TILE * TILE`` threads rasterises one tile."""

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define TILE 16
#define BLOCK (TILE * TILE)

constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2_0 = 1.0925484305920792f;
constant float SH_C2_1 = -1.0925484305920792f;
constant float SH_C2_2 = 0.31539156525252005f;
constant float SH_C2_3 = -1.0925484305920792f;
constant float SH_C2_4 = 0.5462742152960396f;
constant float SH_C3_0 = -0.5900435899266435f;
constant float SH_C3_1 = 2.890611442640554f;
constant float SH_C3_2 = -0.4570457994644658f;
constant float SH_C3_3 = 0.3731763325901154f;
constant float SH_C3_4 = -0.4570457994644658f;
constant float SH_C3_5 = 1.445305721320277f;
constant float SH_C3_6 = -0.5900435899266435f;

constant float ALPHA_MIN = 1.0f / 255.0f;
constant float ALPHA_MAX = 0.99f;
constant float T_MIN = 1e-4f;
constant float EPS2D = 0.3f;

// cam: R (row-major 3x3, world->camera), t (3), fx, fy, cx, cy, near, campos (3)
// dims: N, K (SH coefficients per channel), degree, width, height, tiles_x, tiles_y
kernel void preprocess(
    device const float* means [[buffer(0)]],
    device const float* quats [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* opacities [[buffer(3)]],
    device const float* sh [[buffer(4)]],
    constant float* cam [[buffer(5)]],
    constant long* dims [[buffer(6)]],
    device float4* g_xy [[buffer(7)]],
    device float4* g_conic [[buffer(8)]],
    device float4* g_rgb [[buffer(9)]],
    device int4* g_rect [[buffer(10)]],
    device int* g_count [[buffer(11)]],
    uint i [[thread_position_in_grid]])
{
    const uint n = (uint)dims[0];
    if (i >= n) return;
    g_count[i] = 0;
    g_rect[i] = int4(0);

    const float3 p = float3(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
    const float3 r0 = float3(cam[0], cam[1], cam[2]);
    const float3 r1 = float3(cam[3], cam[4], cam[5]);
    const float3 r2 = float3(cam[6], cam[7], cam[8]);
    const float3 pc = float3(dot(r0, p) + cam[9], dot(r1, p) + cam[10], dot(r2, p) + cam[11]);
    const float near_plane = cam[16];
    if (pc.z < near_plane || pc.z > 1e10f) return;
    const float opacity = opacities[i];
    if (opacity < ALPHA_MIN) return;

    // 3D covariance R S S^T R^T in the world frame (w-first quaternion, normalised here like gsplat).
    float4 q = float4(quats[4 * i], quats[4 * i + 1], quats[4 * i + 2], quats[4 * i + 3]);
    q *= rsqrt(dot(q, q));
    const float w = q.x, x = q.y, y = q.z, z = q.w;
    // Columns of the rotation matrix.
    const float3 c0 = float3(1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y));
    const float3 c1 = float3(2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x));
    const float3 c2 = float3(2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + y * y));
    const float3x3 M = float3x3(c0 * scales[3 * i], c1 * scales[3 * i + 1], c2 * scales[3 * i + 2]);
    const float3x3 cov = M * transpose(M);
    const float3x3 Wt = float3x3(r0, r1, r2);  // columns r0, r1, r2: the transpose of the view rotation
    const float3x3 W = transpose(Wt);
    const float3x3 covc = W * cov * Wt;

    // EWA: Jacobian of the pinhole projection, with gsplat's clamp of the off-axis direction.
    const float fx = cam[12], fy = cam[13], cx = cam[14], cy = cam[15];
    const float width = (float)dims[3], height = (float)dims[4];
    const float tan_fovx = 0.5f * width / fx, tan_fovy = 0.5f * height / fy;
    const float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    const float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    const float lim_y_neg = cy / fy + 0.3f * tan_fovy;
    const float rz = 1.f / pc.z, rz2 = rz * rz;
    const float tx = pc.z * min(lim_x_pos, max(-lim_x_neg, pc.x * rz));
    const float ty = pc.z * min(lim_y_pos, max(-lim_y_neg, pc.y * rz));
    const float3 j0 = float3(fx * rz, 0.f, -fx * tx * rz2);
    const float3 j1 = float3(0.f, fy * rz, -fy * ty * rz2);
    const float3 cj0 = covc * j0, cj1 = covc * j1;
    const float a = dot(j0, cj0) + EPS2D;
    const float b = dot(j0, cj1);
    const float c = dot(j1, cj1) + EPS2D;
    const float det = a * c - b * b;
    if (det <= 0.f) return;
    const float inv_det = 1.f / det;
    const float2 mean2d = float2(fx * pc.x * rz + cx, fy * pc.y * rz + cy);

    // gsplat's opacity-aware bounding box: alpha < 1/255 everywhere outside it.
    const float extend = min(3.33f, sqrt(2.f * log(opacity / ALPHA_MIN)));
    const float rx = ceil(extend * sqrt(a));
    const float ry = ceil(extend * sqrt(c));
    if (rx <= 0.f && ry <= 0.f) return;
    if (mean2d.x + rx <= 0.f || mean2d.x - rx >= width || mean2d.y + ry <= 0.f || mean2d.y - ry >= height) return;
    const float tiles_x = (float)dims[5], tiles_y = (float)dims[6];
    const int tx0 = (int)clamp(floor((mean2d.x - rx) / TILE), 0.f, tiles_x);
    const int tx1 = (int)clamp(ceil((mean2d.x + rx) / TILE), 0.f, tiles_x);
    const int ty0 = (int)clamp(floor((mean2d.y - ry) / TILE), 0.f, tiles_y);
    const int ty1 = (int)clamp(ceil((mean2d.y + ry) / TILE), 0.f, tiles_y);
    const int count = (tx1 - tx0) * (ty1 - ty0);
    if (count <= 0) return;

    // Colour: SH up to the PLY's degree in the direction from the camera centre, +0.5, clamped at 0.
    const int K = (int)dims[1];
    const int degree = (int)dims[2];
    device const float* s = sh + (ulong)i * (ulong)K * 3ul;
    float3 dir = p - float3(cam[17], cam[18], cam[19]);
    dir *= rsqrt(dot(dir, dir));
    #define SHC(k) float3(s[3 * (k)], s[3 * (k) + 1], s[3 * (k) + 2])
    float3 rgb = SH_C0 * SHC(0);
    if (degree >= 1) {
        const float dx = dir.x, dy = dir.y, dz = dir.z;
        rgb += SH_C1 * (-dy * SHC(1) + dz * SHC(2) - dx * SHC(3));
        if (degree >= 2) {
            const float xx = dx * dx, yy = dy * dy, zz = dz * dz;
            const float xy = dx * dy, yz = dy * dz, xz = dx * dz;
            rgb += SH_C2_0 * xy * SHC(4) + SH_C2_1 * yz * SHC(5) + SH_C2_2 * (2.f * zz - xx - yy) * SHC(6)
                 + SH_C2_3 * xz * SHC(7) + SH_C2_4 * (xx - yy) * SHC(8);
            if (degree >= 3) {
                rgb += SH_C3_0 * dy * (3.f * xx - yy) * SHC(9) + SH_C3_1 * xy * dz * SHC(10)
                     + SH_C3_2 * dy * (4.f * zz - xx - yy) * SHC(11)
                     + SH_C3_3 * dz * (2.f * zz - 3.f * xx - 3.f * yy) * SHC(12)
                     + SH_C3_4 * dx * (4.f * zz - xx - yy) * SHC(13) + SH_C3_5 * dz * (xx - yy) * SHC(14)
                     + SH_C3_6 * dx * (xx - 3.f * yy) * SHC(15);
            }
        }
    }
    #undef SHC
    rgb = max(rgb + 0.5f, 0.f);

    g_xy[i] = float4(mean2d, pc.z, opacity);
    g_conic[i] = float4(c * inv_det, -b * inv_det, a * inv_det, 0.f);
    g_rgb[i] = float4(rgb, 0.f);
    g_rect[i] = int4(tx0, ty0, tx1, ty1);
    g_count[i] = count;
}

// One (tile, depth) key per tile a Gaussian's box touches; offsets is the inclusive prefix sum of counts.
kernel void emit_keys(
    device const int4* g_rect [[buffer(0)]],
    device const int* offsets [[buffer(1)]],
    device const float4* g_xy [[buffer(2)]],
    device long* keys [[buffer(3)]],
    device int* values [[buffer(4)]],
    constant long* dims [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= (uint)dims[0]) return;
    const int4 r = g_rect[i];
    const int count = (r.z - r.x) * (r.w - r.y);
    if (count <= 0) return;
    int offset = offsets[i] - count;
    const long depth_bits = (long)as_type<uint>(g_xy[i].z);  // depth > 0: its bits sort like the float
    const int tiles_x = (int)dims[1];
    for (int ty = r.y; ty < r.w; ++ty) {
        for (int tx = r.x; tx < r.z; ++tx) {
            keys[offset] = ((long)(ty * tiles_x + tx) << 32) | depth_bits;
            values[offset] = (int)i;
            ++offset;
        }
    }
}

// [start, end) of each tile's run in the sorted keys (ranges is zero-filled beforehand).
kernel void tile_ranges(
    device const long* keys [[buffer(0)]],
    device int* ranges [[buffer(1)]],
    constant long* dims [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    // Scalar stores: writing one component of an int2 is a read-modify-write of both, which races
    // with the thread writing the other end of the same tile.
    const uint n = (uint)dims[0];
    if (i >= n) return;
    const int tile = (int)(keys[i] >> 32);
    if (i == 0 || (int)(keys[i - 1] >> 32) != tile) ranges[2 * tile] = (int)i;
    if (i == n - 1 || (int)(keys[i + 1] >> 32) != tile) ranges[2 * tile + 1] = (int)(i + 1);
}

// One threadgroup per tile, one thread per pixel; Gaussians are staged through threadgroup memory
// in batches of BLOCK, front to back.
// out: planar float32 [rgb (H*W*3, interleaved) | alpha (H*W) | expected depth (H*W) | median depth (H*W)]
// dims: width, height, tiles_x
kernel void rasterize(
    device const int2* ranges [[buffer(0)]],
    device const int* ids [[buffer(1)]],
    device const float4* g_xy [[buffer(2)]],
    device const float4* g_conic [[buffer(3)]],
    device const float4* g_rgb [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant float* bg [[buffer(6)]],
    constant long* dims [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_count [[simdgroups_per_threadgroup]])
{
    threadgroup float4 s_xy[BLOCK];
    threadgroup float4 s_conic[BLOCK];
    threadgroup float4 s_rgb[BLOCK];
    threadgroup bool s_done[BLOCK / 4];

    const int width = (int)dims[0], height = (int)dims[1], tiles_x = (int)dims[2];
    const int px = (int)group.x * TILE + (int)local.x;
    const int py = (int)group.y * TILE + (int)local.y;
    const bool inside = px < width && py < height;
    const float fx = (float)px + 0.5f, fy = (float)py + 0.5f;
    const int2 range = ranges[group.y * tiles_x + group.x];

    bool done = !inside;
    float T = 1.f;
    float3 color = float3(0.f);
    float depth_sum = 0.f;
    float weight_sum = 0.f;
    float median = 0.f;

    for (int base = range.x; base < range.y; base += BLOCK) {
        // Stop the whole tile once every pixel in it is saturated.
        const bool simd_done = simd_all(done);
        if (simd_lane == 0) s_done[simd_id] = simd_done;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool all_done = true;
        for (uint k = 0; k < simd_count; ++k) all_done = all_done && s_done[k];
        if (all_done) break;

        const int idx = base + (int)lid;
        if (idx < range.y) {
            const int g = ids[idx];
            s_xy[lid] = g_xy[g];
            s_conic[lid] = g_conic[g];
            s_rgb[lid] = g_rgb[g];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!done) {
            const int batch = min(BLOCK, range.y - base);
            for (int j = 0; j < batch; ++j) {
                const float4 xy = s_xy[j];
                const float4 co = s_conic[j];
                const float dx = xy.x - fx, dy = xy.y - fy;
                const float sigma = 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
                const float alpha = min(ALPHA_MAX, xy.w * exp(-sigma));
                if (sigma < 0.f || alpha < ALPHA_MIN) continue;
                const float next_T = T * (1.f - alpha);
                if (next_T < T_MIN) {
                    done = true;
                    break;
                }
                const float weight = alpha * T;
                color += weight * s_rgb[j].xyz;
                depth_sum += weight * xy.z;
                weight_sum += weight;
                if (T > 0.5f && next_T <= 0.5f) median = xy.z;
                T = next_T;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!inside) return;
    const int pix = py * width + px;
    const int hw = width * height;
    const float3 rgb = clamp(color + T * float3(bg[0], bg[1], bg[2]), 0.f, 1.f);
    out[3 * pix] = rgb.x;
    out[3 * pix + 1] = rgb.y;
    out[3 * pix + 2] = rgb.z;
    out[3 * hw + pix] = 1.f - T;
    out[4 * hw + pix] = weight_sum > 0.f ? depth_sum / weight_sum : 0.f;
    out[5 * hw + pix] = median;
}
"""
