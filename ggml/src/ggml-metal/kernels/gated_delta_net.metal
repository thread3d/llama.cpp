#include "common.h"

constant short FC_gated_delta_net_ne20 [[function_constant(FC_GATED_DELTA_NET + 0)]];
constant short FC_gated_delta_net_ne30 [[function_constant(FC_GATED_DELTA_NET + 1)]];
constant short FC_gated_delta_net_K    [[function_constant(FC_GATED_DELTA_NET + 2)]];
// read the state from the recurrent cache row of each sequence, and write the new one back there,
// instead of going through a gathered copy and a copy back
constant bool  FC_gated_delta_net_rows [[function_constant(FC_GATED_DELTA_NET + 3)]];
constant bool  FC_gated_delta_net_fold [[function_constant(FC_GATED_DELTA_NET + 4)]];
// g and b arrive before their activations: b -> sigmoid(b), g -> a[h] * softplus(g + dt[h])
constant bool  FC_gated_delta_net_raw  [[function_constant(FC_GATED_DELTA_NET + 5)]];
// q and k arrive before their l2 norm
constant bool  FC_gated_delta_net_l2q  [[function_constant(FC_GATED_DELTA_NET + 6)]];
constant bool  FC_gated_delta_net_l2k  [[function_constant(FC_GATED_DELTA_NET + 7)]];
static inline float gdn_sum32(float v) {
    v += simd_shuffle_xor(v, 16);
    v += simd_shuffle_xor(v,  8);
    v += simd_shuffle_xor(v,  4);
    v += simd_shuffle_xor(v,  2);
    v += simd_shuffle_xor(v,  1);
    return v;
}


#if 1
template<short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device const char * rows,
        device       char * sdst,
        device const char * gdt,
        device const char * ga,
        device       char * dst_fuse,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30
#define K   FC_gated_delta_net_K

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B (n_seqs)
    const uint i21 = tgpig.y; // H (head)
    const uint i20 = tgpig.x*NSG + ty; // row within S_v

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    // input state layout [S_v, S_v, H, n_seqs] (s0 only): per-seq stride is H*D.
    // state is stored transposed: M[i20][is] = S[is][i20], so row i20 is contiguous
    const uint state_in_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;
    device const float * s_ptr = FC_gated_delta_net_rows ?
        (device const float *) (s + ((device const int32_t *) rows)[i23]*args.s_nb1) + i21*S_v*S_v + i20*S_v :
        (device const float *) (s) + state_in_base;

    float ls[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        ls[j] = s_ptr[is];
    }

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K, only slots 0..n_tokens-1 are written; older slots are caller-owned.

    // output state base offset: after attention scores
    const uint attn_size = args.ne22 * args.ne21 * S_v * args.ne23;
    // output state per-slot size: S_v * S_v * H * n_seqs
    const uint state_size_per_snap = S_v * S_v * args.ne21 * args.ne23;
    // per-(seq,head) offset within a slot
    const uint state_out_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;

    const float g_dt = FC_gated_delta_net_raw ? ((device const float *) gdt)[i21] : 0.0f;
    const float g_a  = FC_gated_delta_net_raw ? ((device const float *) ga)[i21]  : 0.0f;

    // when fused with the cache cpy, write the snapshots straight into the cache buffer using
    // the slot stride; otherwise append them after the attn scores (nb_out == 0)
    const bool fused = args.nb_out > 0;
    const device float * state_out = fused ? (device float *)dst_fuse : (device float *)dst + attn_size;
    const uint slot_stride = fused ? (uint)args.nb_out : state_size_per_snap;

    for (short t = 0; t < args.ne22; t++) {
        float s_k = 0.0f;

        // l2 norms of this token's q and k rows, reduced over the lanes that hold them
        float k_n = 1.0f;
        float q_n = 1.0f;
        if (FC_gated_delta_net_l2k) {
            float kk = 0.0f;
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const float kv = k_ptr[tx*NSG + j];
                kk += kv*kv;
            }
            k_n = 1.0f/sqrt(gdn_sum32(kk) + args.eps_k);
        }
        if (FC_gated_delta_net_l2q) {
            float qq = 0.0f;
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const float qv = q_ptr[tx*NSG + j];
                qq += qv*qv;
            }
            q_n = 1.0f/sqrt(gdn_sum32(qq) + args.eps_q);
        }

        if (G == 1) {
            float g_val = g_ptr[0];
            if (FC_gated_delta_net_raw) {
                const float x = g_val + g_dt;
                g_val = g_a*select(log(1 + exp(x)), x, x > 20);
            }
            const float g_exp = exp(g_val);

            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= g_exp;

                s_k += ls[j]*k_ptr[is];
            }
        } else {
            // KDA
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= exp(g_ptr[is]);

                s_k += ls[j]*k_ptr[is];
            }
        }

        s_k = gdn_sum32(s_k)*k_n;

        const float b_val = FC_gated_delta_net_raw ? 1 / (1 + exp(-b_ptr[0])) : b_ptr[0];
        const float d = (v_ptr[i20] - s_k)*b_val;
        const float d_k = d*k_n;

        float y = 0.0f;

        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            ls[j] += k_ptr[is]*d_k;

            y += ls[j]*q_ptr[is];
        }

        y = gdn_sum32(y)*q_n;

        if (tx == 0) {
            dst_attn[t*args.ne21*S_v] = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        if (K > 1) {
            const int target_slot = (int)args.ne22 - 1 - (int)t;
            if (target_slot >= 0 && target_slot < (int)K) {
                device float * dst_state = (device float *)state_out + (uint)target_slot * slot_stride + state_out_base;
                FOR_UNROLL (short j = 0; j < NSG; j++) {
                    const short is = tx*NSG + j;
                    dst_state[is] = ls[j];
                }
            }
        }
    }

    if (K == 1) {
        device float * dst_state = FC_gated_delta_net_fold ?
            (device float *) (sdst + i23*args.sd_nb1) + i21*S_v*S_v + i20*S_v :
            (device float *)state_out + state_out_base;
        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            dst_state[is] = ls[j];
        }
    }

#undef S_v
#undef G
#undef K
}

// LPR lanes share one row of the transposed state, EPL = S_v/LPR values each, so the two dot
// products per token reduce over log2(LPR) lanes instead of 32 and every lane reads its slice of
// q and k as whole vectors. A threadgroup of 128 threads covers 128/LPR rows.
template<short LPR>
static inline float gdn_sum_lpr(float v) {
    FOR_UNROLL (short o = LPR/2; o > 0; o /= 2) {
        v += simd_shuffle_xor(v, o);
    }
    return v;
}

template<short LPR>
kernel void kernel_gated_delta_net_lpr(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device const char * rows,
        device       char * sdst,
        device const char * gdt,
        device const char * ga,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30
#define K   FC_gated_delta_net_K
    constexpr short EPL = 128/LPR; // S_v is 128 on this path
    constexpr short RPT = 128/LPR;

    const short sub = tiitg % LPR;
    const uint i23 = tgpig.z;
    const uint i21 = tgpig.y;
    const uint i20 = tgpig.x*RPT + tiitg/LPR;

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    const uint state_in_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;
    device const float4 * s_ptr = (device const float4 *) (FC_gated_delta_net_rows ?
        (device const float *) (s + ((device const int32_t *) rows)[i23]*args.s_nb1) + i21*S_v*S_v + i20*S_v :
        (device const float *) (s) + state_in_base) + sub*(EPL/4);

    float4 ls[EPL/4];
    FOR_UNROLL (short j = 0; j < EPL/4; j++) {
        ls[j] = s_ptr[j];
    }

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    const uint attn_size = args.ne22 * args.ne21 * S_v * args.ne23;
    const uint state_size_per_snap = S_v * S_v * args.ne21 * args.ne23;
    const uint state_out_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;

    const float g_dt = FC_gated_delta_net_raw ? ((device const float *) gdt)[i21] : 0.0f;
    const float g_a  = FC_gated_delta_net_raw ? ((device const float *) ga)[i21]  : 0.0f;

    for (short t = 0; t < args.ne22; t++) {
        device const float4 * k4 = (device const float4 *) k_ptr + sub*(EPL/4);
        device const float4 * q4 = (device const float4 *) q_ptr + sub*(EPL/4);

        float4 kv[EPL/4];
        float4 qv[EPL/4];
        FOR_UNROLL (short j = 0; j < EPL/4; j++) {
            kv[j] = k4[j];
            qv[j] = q4[j];
        }

        float k_n = 1.0f;
        float q_n = 1.0f;
        if (FC_gated_delta_net_l2k) {
            float kk = 0.0f;
            FOR_UNROLL (short j = 0; j < EPL/4; j++) {
                kk += dot(kv[j], kv[j]);
            }
            k_n = 1.0f/sqrt(gdn_sum_lpr<LPR>(kk) + args.eps_k);
        }
        if (FC_gated_delta_net_l2q) {
            float qq = 0.0f;
            FOR_UNROLL (short j = 0; j < EPL/4; j++) {
                qq += dot(qv[j], qv[j]);
            }
            q_n = 1.0f/sqrt(gdn_sum_lpr<LPR>(qq) + args.eps_q);
        }

        float s_k = 0.0f;
        if (G == 1) {
            float g_val = g_ptr[0];
            if (FC_gated_delta_net_raw) {
                const float x = g_val + g_dt;
                g_val = g_a*select(log(1 + exp(x)), x, x > 20);
            }
            const float g_exp = exp(g_val);
            FOR_UNROLL (short j = 0; j < EPL/4; j++) {
                ls[j] *= g_exp;
                s_k += dot(ls[j], kv[j]);
            }
        } else {
            device const float4 * g4 = (device const float4 *) g_ptr + sub*(EPL/4);
            FOR_UNROLL (short j = 0; j < EPL/4; j++) {
                ls[j] *= exp(g4[j]);
                s_k += dot(ls[j], kv[j]);
            }
        }

        s_k = gdn_sum_lpr<LPR>(s_k)*k_n;

        const float b_val = FC_gated_delta_net_raw ? 1 / (1 + exp(-b_ptr[0])) : b_ptr[0];
        const float d_k = (v_ptr[i20] - s_k)*b_val*k_n;

        float y = 0.0f;
        FOR_UNROLL (short j = 0; j < EPL/4; j++) {
            ls[j] += kv[j]*d_k;
            y += dot(ls[j], qv[j]);
        }

        y = gdn_sum_lpr<LPR>(y)*q_n;

        if (sub == 0) {
            dst_attn[t*args.ne21*S_v] = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        if (K > 1) {
            const int target_slot = (int)args.ne22 - 1 - (int)t;
            if (target_slot >= 0 && target_slot < (int)K) {
                device float4 * ds4 = (device float4 *) ((device float *) (dst) + attn_size + (uint)target_slot * state_size_per_snap + state_out_base) + sub*(EPL/4);
                FOR_UNROLL (short j = 0; j < EPL/4; j++) {
                    ds4[j] = ls[j];
                }
            }
        }
    }

    if (K == 1) {
        device float4 * ds4 = (device float4 *) (FC_gated_delta_net_fold ?
            (device float *) (sdst + i23*args.sd_nb1) + i21*S_v*S_v + i20*S_v :
            (device float *) (dst) + attn_size + state_out_base) + sub*(EPL/4);
        FOR_UNROLL (short j = 0; j < EPL/4; j++) {
            ds4[j] = ls[j];
        }
    }

#undef S_v
#undef G
#undef K
}

typedef decltype(kernel_gated_delta_net_lpr<8>) kernel_gated_delta_net_lpr_t;

template [[host_name("kernel_gated_delta_net_f32_lpr4")]]  kernel kernel_gated_delta_net_lpr_t kernel_gated_delta_net_lpr<4>;
template [[host_name("kernel_gated_delta_net_f32_lpr8")]]  kernel kernel_gated_delta_net_lpr_t kernel_gated_delta_net_lpr<8>;
template [[host_name("kernel_gated_delta_net_f32_lpr16")]] kernel kernel_gated_delta_net_lpr_t kernel_gated_delta_net_lpr<16>;
template [[host_name("kernel_gated_delta_net_f32_lpr32")]] kernel kernel_gated_delta_net_lpr_t kernel_gated_delta_net_lpr<32>;

typedef decltype(kernel_gated_delta_net_impl<4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<4>;

#else
// a simplified version of the above
// no performance improvement, so keep the above version for now

template<typename T, short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device       char * dst_fuse,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B
    const uint i21 = tgpig.y; // H
    const uint i20 = tgpig.x*NSG + ty;

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    device const float * s_ptr = (device const float *) (s) + (i23*args.ne21 + i21)*S_v*S_v + i20;

    float lsf[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        lsf[j] = s_ptr[is*S_v];
    }

    thread T * ls = (thread T *) (lsf);

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr  = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr  = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    for (short t = 0; t < args.ne22; t++) {
        device const T * qt_ptr = (device const T *) (q_ptr);
        device const T * kt_ptr = (device const T *) (k_ptr);
        device const T * gt_ptr = (device const T *) (g_ptr);

        if (G == 1) {
            *ls *= exp(g_ptr[0]);
        } else {
            // KDA
            *ls *= exp(gt_ptr[tx]);
        }

        const float s_k = simd_sum(dot(*ls, kt_ptr[tx]));

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        *ls += kt_ptr[tx]*d;

        const float y = simd_sum(dot(*ls, qt_ptr[tx]));

        if (tx == 0) {
            *dst_attn = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        dst_attn += args.ne21*S_v;
    }

    // when fused with the cache cpy, write the snapshots straight into the cache buffer using
    // the slot stride; otherwise append them after the attn scores (nb_out == 0)
    const bool fused = args.nb_out > 0;
    const device float * state_out = fused ? (device float *)dst_fuse : (device float *)dst + args.ne23*args.ne22*args.ne21*S_v;
    const uint slot_stride = fused ? (uint)args.nb_out : S_v*S_v;

    device float * dst_state  = (device float *)state_out + (i23*args.ne21 + i21)*slot_stride + i20;
    device T     * dstt_state = (device T     *) (dst_state);

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        dst_state[is*S_v] = lsf[j];
    }

#undef S_v
#undef G
}

typedef decltype(kernel_gated_delta_net_impl<float4, 4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float,  1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float2, 2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float4, 4>;
#endif
