#include "common.h"

template<typename T>
kernel void kernel_soft_max(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint sgptg[[threads_per_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float * psrc0 =                (device const float *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const     T * pmask = src1 != src0 ? (device const T *    ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float * psrc2 = src2 != src0 ? (device const float *) (src2)                                                 : nullptr;
    device       float * pdst  =                (device       float *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    // ALiBi
    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        lmax = MAX(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (tptg.x > sgptg) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > sgptg) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

template<typename T>
kernel void kernel_soft_max_4(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint sgptg[[threads_per_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float4 * psrc4 =                (device const float4 *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const      T * pmask = src1 != src0 ? (device const T *     ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float *  psrc2 = src2 != src0 ? (device const float * ) (src2)                                                 : nullptr;
    device       float4 * pdst4 =                (device       float4 *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        lmax4 = fmax(lmax4, psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (tptg.x > sgptg) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        const float4 exp_psrc4 = exp((psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > sgptg) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        pdst4[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>)    kernel_soft_max_t;
typedef decltype(kernel_soft_max_4<float4>) kernel_soft_max_4_t;

template [[host_name("kernel_soft_max_f16")]]   kernel kernel_soft_max_t   kernel_soft_max<half>;
template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float>;
template [[host_name("kernel_soft_max_f16_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<half4>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<float4>;

kernel void kernel_topk_moe_f32(
        constant ggml_metal_kargs_topk_moe & args,
        device const char * src_logits,
        device const char * src_bias,
        device       char * dst_weights,
        device       char * dst_ids,
        threadgroup float * sf [[threadgroup(0)]],
        threadgroup int   * si [[threadgroup(1)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiitg[[thread_index_in_threadgroup]],
        uint3 ntgg [[threads_per_threadgroup]]) {
    const uint ntg = ntgg.x;
    const int row = tgpig.x;
    if (row >= args.nrows) {
        return;
    }

    const int ne = args.n_expert;
    const int nu = args.n_expert_used;

    device const float   * logits  = (device const float   *) src_logits  + (size_t) ne*row;
    device const float   * bias    = (device const float   *) src_bias;
    device       float   * weights = (device       float   *) dst_weights + (size_t) nu*row;
    device       int32_t * ids     = (device       int32_t *) dst_ids     + (size_t) ne*row;

    threadgroup float * wt  = sf;          // value that becomes the weight
    threadgroup float * sel = sf + ne;     // value the top-k selects on (wt + bias)
    threadgroup float * rv  = sf + 2*ne;   // reduction scratch, ntg entries

    for (int i = tiitg; i < ne; i += ntg) {
        wt[i] = logits[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (args.mode == 0) {
        float lmax = -INFINITY;
        for (int i = tiitg; i < ne; i += ntg) {
            lmax = max(lmax, wt[i]);
        }
        rv[tiitg] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = ntg/2; s > 0; s >>= 1) {
            if (tiitg < s) {
                rv[tiitg] = max(rv[tiitg], rv[tiitg + s]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float vmax = rv[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float lsum = 0.0f;
        for (int i = tiitg; i < ne; i += ntg) {
            const float v = exp(wt[i] - vmax);
            wt[i] = v;
            lsum += v;
        }
        rv[tiitg] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = ntg/2; s > 0; s >>= 1) {
            if (tiitg < s) {
                rv[tiitg] += rv[tiitg + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float inv = 1.0f/rv[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tiitg; i < ne; i += ntg) {
            wt[i] *= inv;
        }
    } else if (args.mode == 1) {
        for (int i = tiitg; i < ne; i += ntg) {
            wt[i] = 1.0f/(1.0f + exp(-wt[i]));
        }
    } else if (args.mode == 2) {
        for (int i = tiitg; i < ne; i += ntg) {
            const float v = wt[i];
            wt[i] = sqrt(v > 20.0f ? v : log(1.0f + exp(v)));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // a NaN never compares greater, so the argmax below would return the same
    // expert every round; -FLT_MAX still loses to every real score
    for (int i = tiitg; i < ne; i += ntg) {
        float v = wt[i];
        if (v != v) {
            v = -FLT_MAX;
            wt[i] = v;
        }
        sel[i] = args.has_bias ? v + bias[i] : v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 0; k < nu; ++k) {
        float bv = -INFINITY;
        int   bi = ne;
        for (int i = tiitg; i < ne; i += ntg) {
            const float v = sel[i];
            if (v > bv || (v == bv && i < bi)) {
                bv = v;
                bi = i;
            }
        }
        rv[tiitg] = bv;
        si[tiitg] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = ntg/2; s > 0; s >>= 1) {
            if (tiitg < s) {
                const float ov = rv[tiitg + s];
                const int   oi = si[tiitg + s];
                if (ov > rv[tiitg] || (ov == rv[tiitg] && oi < si[tiitg])) {
                    rv[tiitg] = ov;
                    si[tiitg] = oi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tiitg == 0) {
            const int e = si[0];
            ids[k]     = e;
            weights[k] = wt[e];
            sel[e]     = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    if (args.with_norm != 0) {
        float lsum = 0.0f;
        for (int i = tiitg; i < nu; i += ntg) {
            lsum += weights[i];
        }
        rv[tiitg] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = ntg/2; s > 0; s >>= 1) {
            if (tiitg < s) {
                rv[tiitg] += rv[tiitg + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float inv = 1.0f/max(rv[0], args.clamp);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tiitg; i < nu; i += ntg) {
            weights[i] *= inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    if (args.delayed_softmax != 0) {
        float lmax = -INFINITY;
        for (int i = tiitg; i < nu; i += ntg) {
            lmax = max(lmax, weights[i]);
        }
        rv[tiitg] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = ntg/2; s > 0; s >>= 1) {
            if (tiitg < s) {
                rv[tiitg] = max(rv[tiitg], rv[tiitg + s]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float vmax = rv[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float lsum = 0.0f;
        for (int i = tiitg; i < nu; i += ntg) {
            const float v = exp(weights[i] - vmax);
            weights[i] = v;
            lsum += v;
        }
        rv[tiitg] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        for (uint s = ntg/2; s > 0; s >>= 1) {
            if (tiitg < s) {
                rv[tiitg] += rv[tiitg + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float inv = 1.0f/rv[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tiitg; i < nu; i += ntg) {
            weights[i] *= inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    if (args.scale != 1.0f) {
        for (int i = tiitg; i < nu; i += ntg) {
            weights[i] *= args.scale;
        }
    }
}
