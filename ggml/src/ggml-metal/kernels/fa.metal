#include "common.h"
#include "dequantize.h"

// dequantize a quantized KV cache tensor to contiguous F16 before running the F16 flash attention kernels
// - one thread per block; dispatched separately for K and V
// - ref: https://github.com/ggml-org/llama.cpp/pull/27390
template <
    typename block_t,
    short QK,
    void (*deq_t4x4)(device const block_t *, short, thread float4x4 &)>
kernel void kernel_flash_attn_ext_kv_f16(
        constant ggml_metal_kargs_flash_attn_ext_kv_f16 & args,
        device const char * x,
        device       half * x_dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.nblocks) {
        return;
    }

    const uint nb = args.ne0/QK;
    const uint i0 = gid%nb;
    uint ib       = gid/nb;
    const uint i1 = ib%args.ne1;
    ib /= args.ne1;
    const uint i2 = ib%args.ne2;
    const uint i3 = ib/args.ne2;

    const uint64_t offs = i0*args.nb0 + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;

    device const block_t * src = (device const block_t *) (x + offs);
    device half4 * dst = (device half4 *) x_dst + (QK/4)*gid;

    for (short i = 0; i < QK/16; ++i) {
        float4x4 reg;
        deq_t4x4(src, i, reg);
        dst[4*i + 0] = (half4) reg[0];
        dst[4*i + 1] = (half4) reg[1];
        dst[4*i + 2] = (half4) reg[2];
        dst[4*i + 3] = (half4) reg[3];
    }
}

typedef decltype(kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>) kernel_flash_attn_ext_kv_f16_t;

template [[host_name("kernel_flash_attn_ext_kv_q4_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_0, 32, dequantize_q4_0>;
template [[host_name("kernel_flash_attn_ext_kv_q4_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_1, 32, dequantize_q4_1>;
template [[host_name("kernel_flash_attn_ext_kv_q5_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_0, 32, dequantize_q5_0>;
template [[host_name("kernel_flash_attn_ext_kv_q5_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_1, 32, dequantize_q5_1>;
template [[host_name("kernel_flash_attn_ext_kv_q8_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>;

constant bool FC_flash_attn_ext_pad_has_mask [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];

constant int32_t FC_flash_attn_ext_pad_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_PAD + 25)]];

// pad the last chunk of C elements of k and v into a an extra pad buffer
kernel void kernel_flash_attn_ext_pad(
        constant ggml_metal_kargs_flash_attn_ext_pad & args,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t C = FC_flash_attn_ext_pad_ncpsg;

    device char * k_pad    = dst;
    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;

    const int32_t icp = args.ne11 % C;
    const int32_t ic0 = args.ne11 - icp;

    const int32_t i1 = tgpig[0];
    const int32_t i2 = tgpig[1];
    const int32_t i3 = tgpig[2];

    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;

        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;

        if (i1 >= icp) {
            // here it is not important the exact value that will be used as we rely on masking out the scores in the attention
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = 0;
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = 0;
            }
        } else {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = k_src[i];
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = v_src[i];
            }
        }
    }

    if (FC_flash_attn_ext_pad_has_mask) {
        if (i2 < args.ne32 && i3 < args.ne33) {
            for (int ib = i1; ib < args.ne31; ib += C) {
                device const half * mask_src = (device const half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
                device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;

                for (int i = tiitg; i < C; i += ntg.x) {
                    if (i >= icp) {
                        mask_dst[i] = -MAXHALF;
                    } else {
                        mask_dst[i] = mask_src[i];
                    }
                }
            }
        }
    }
}

constant int32_t FC_flash_attn_ext_blk_nqptg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 24)]];
constant int32_t FC_flash_attn_ext_blk_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 25)]];

// scan the blocks of the mask that are not masked
// 0 -     masked (i.e. full of -INF, skip)
// 1 - not masked (i.e. at least one element of the mask is not -INF)
// 2 - all zero
kernel void kernel_flash_attn_ext_blk(
        constant ggml_metal_kargs_flash_attn_ext_blk & args,
        device const char * mask,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    // block size C x Q
    const int32_t Q = FC_flash_attn_ext_blk_nqptg;
    const int32_t C = FC_flash_attn_ext_blk_ncpsg;

    constexpr short NW  = N_SIMDWIDTH;

    const int32_t i3 = tgpig[2]/args.ne32;
    const int32_t i2 = tgpig[2]%args.ne32;
    const int32_t i1 = tgpig[1];
    const int32_t i0 = tgpig[0];

    char res = i0*C + C > args.ne30 ? 1 : 0;

    device const half * mask_src = (device const half *) (mask + (i1*Q)*args.nb31 + i2*args.nb32 + i3*args.nb33) + i0*C + tiisg;

    // detailed check of the elements of the block
    if ((C > NW || Q > 1) && res == 0) {
        half mmin =  MAXHALF;
        half mmax = -MAXHALF;

        FOR_UNROLL (short j = 0; j < Q; ++j) {
            FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
                mmin = min(mmin, mask_src[ii*NW]);
                mmax = max(mmax, mask_src[ii*NW]);
            }

            mask_src += args.nb31/2;
        }

        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        if (mmax > -MAXHALF) {
            if (mmin == 0.0 && mmax == 0.0) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);

    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}

constant bool FC_flash_attn_ext_has_mask  [[function_constant(FC_FLASH_ATTN_EXT + 0)]];
constant bool FC_flash_attn_ext_has_sinks [[function_constant(FC_FLASH_ATTN_EXT + 1)]];
constant bool FC_flash_attn_ext_has_bias  [[function_constant(FC_FLASH_ATTN_EXT + 2)]];
constant bool FC_flash_attn_ext_has_scap  [[function_constant(FC_FLASH_ATTN_EXT + 3)]];
constant bool FC_flash_attn_ext_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT + 4)]];

constant bool FC_flash_attn_ext_bc_mask [[function_constant(FC_FLASH_ATTN_EXT + 10)]];

//constant float FC_flash_attn_ext_scale         [[function_constant(FC_FLASH_ATTN_EXT + 10)]];
//constant float FC_flash_attn_ext_max_bias      [[function_constant(FC_FLASH_ATTN_EXT + 11)]];
//constant float FC_flash_attn_ext_logit_softcap [[function_constant(FC_FLASH_ATTN_EXT + 12)]];

constant int32_t FC_flash_attn_ext_ns10 [[function_constant(FC_FLASH_ATTN_EXT + 20)]];
constant int32_t FC_flash_attn_ext_ns20 [[function_constant(FC_FLASH_ATTN_EXT + 21)]];
constant int32_t FC_flash_attn_ext_nsg  [[function_constant(FC_FLASH_ATTN_EXT + 22)]];

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<
    typename q_t,     // query types in shared memory
    typename q4_t,
    typename q8x8_t,
    typename k_t,     // key types in shared memory
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,     // value types in shared memory
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,    // Q*K types
    typename qk8x8_t,
    typename s_t,     // soft-max types
    typename s2_t,
    typename s8x8_t,
    typename o_t,     // attention accumulation types
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t, // key type in device memory
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // value type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,         // K head size
    short DV,         // V head size
    short Q,          // queries per threadgroup
    short C,          // cache items per threadgroup
    short NSG>        // number of simd groups
void kernel_flash_attn_ext_impl(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0]*Q;

#define NS10 (FC_flash_attn_ext_ns10)
#define NS20 (FC_flash_attn_ext_ns20)

    // note: I had some concerns that using this instead of the ugly macros above was affecting performance
    //       need to re-check carefully and if no regressions are observerd - remove the macros
    //       the concerns is that maybe using const variables requires extra registers? but not sure if the compiler
    //         is clever enough to avoid this. unfortunately, using constexpr is not possible with FC
    //const short NS10 = FC_flash_attn_ext_ns10;
    //const short NS20 = FC_flash_attn_ext_ns20;

    constexpr short KV   = 8;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;
    constexpr short DV4  = DV/4;
  //constexpr short DV8  = DV/8;
    constexpr short DV16 = DV/16;

    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;
  //constexpr short PV16 = PV/16;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NQ  = Q/NSG;
    constexpr short SH  = 2*C; // shared memory per simdgroup (s_t == float)

    constexpr short TS = 2*SH;
    constexpr short T  = DK + 2*PV; // shared memory size per query in (half)

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T); // holds the query data
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T); // same as above but in q4_t
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK); // the result for all queries in 8x8 matrices (the O matrix from the paper)
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T); // scratch buffer for attention, mask and diagonal matrix
    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T); // same as above but in s2_t

    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // scratch buffer to load K in shared memory
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // same as above but in k4x4_t

    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // scratch buffer to load V in shared memory
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // same as above but in v4x4_t

    // mask storage in shared mem
    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);

    // per-query mask pointers
    device const half2 * pm2[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
        const int32_t nblk0 = ((args.ne11 + C - 1)/C);

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // load heads from Q to shared memory
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);

        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = (q4_t) q4[i];
            } else {
                sq4[j*DK4 + i] = 0;
            }
        }
    }

    // zero out
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        for (short i = tiisg; i < DV4; i += NW) {
            so4[j*PV4 + i] = 0;
        }

        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };

    {
        float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

        float slope = 1.0f;

        // ALiBi
        if (FC_flash_attn_ext_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = 0; ; ++ic0) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            // the last partial chunk uses the pad buffer as source
            if (FC_flash_attn_ext_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_has_mask) {
                    threadgroup half * sm = (threadgroup half *) (sm2);

                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        for (short i = tiisg; i < C; i += NW) {
                            if (ic + i >= args.ne11) {
                                sm[2*j*SH + i] = -MAXHALF;
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        pm2[jj] = (device const half2 *) ((device const half *) mask +
                                (iq1 + j)*C +
                                (iq2%args.ne32)*(C*args.ne31) +
                                (iq3%args.ne33)*(C*args.ne31*args.ne32));
                    }
                }

                ic = 0;
            }

            char blk_cur = 1;

            // read the mask into shared mem
            if (FC_flash_attn_ext_has_mask) {
                blk_cur = blk[ic0];

                if (blk_cur == 0) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }

                    continue;
                }

                if (blk_cur == 1) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        if (FC_flash_attn_ext_bc_mask) {
                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-MAXHALF, -MAXHALF);
                        } else {
                            sm2[j*SH + tiisg] = pm2[jj][tiisg];
                        }

                        pm2[jj] += NW;
                    }
                } else if (blk_cur == 2) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }
                }

#if 0
                // note: old -INF block optimization - obsoleted by pre-computing non-masked blocks

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // used to detect blocks full of -INF
                // skip only when the entire threadgroup is masked
                half2 smax2(-MAXHALF/2, -MAXHALF/2);

                FOR_UNROLL (short j = 0; j < Q; ++j) {
                    smax2 = max(smax2, sm2[j*SH + tiisg]);
                }

                smax2 = simd_max(smax2);

                if (max(smax2[0], smax2[1]) <= -MAXHALF/2) {
                    // this barrier is important
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    continue;
                }
#endif
            }

            // Q*K^T
            // this is compile-time check, so it does not have runtime overhead
            if (is_same<kd4x4_t, k4x4_t>::value) {
                // we can read directly from global memory
                device      const k_t * pk = (device const k_t *) (k + ic*args.nb11);
                threadgroup const q_t * pq = sq;
                threadgroup       s_t * ps = ss;

                pk += sgitg*(8*NS10);
                ps += sgitg*(8*1);

                static_assert((C/8) % NSG == 0, "");

                constexpr short NC = (C/8)/NSG;

                FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    if (DK % 16 != 0) {
                        k8x8_t mk;
                        q8x8_t mq;

                        FOR_UNROLL (short i = 0; i < DK8; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mk, pk + 8*i, NS10, 0, true);
                            simdgroup_load(mq, pq + 8*i, DK);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        }
                    } else {
                        k8x8_t mk[2];
                        q8x8_t mq[2];

                        // note: too much unroll can tank the performance for large heads
                        #pragma unroll (MIN(DK8/2, 4*NSG))
                        for (short i = 0; i < DK8/2; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                            simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);

                            simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                            simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                        }
                    }

                    simdgroup_store(mqk, ps, SH, 0, false);

                    pk += 8*(NSG*NS10);
                    ps += 8*(NSG);
                }
            } else {
                // TODO: this is the quantized K cache branch - not optimized yet
                for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {
                    const short cc = ccc*NSG + sgitg;

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    for (short ii = 0; ii < DK16; ii += 4) {
                        device const kd4x4_t * pk4x4 = (device const kd4x4_t *) (k + ((ic + 8*cc + ty)*args.nb11));

                        if (DK16%4 == 0) {
                            // the head is evenly divisible by 4*16 = 64, so no need for bound checks
                            {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            FOR_UNROLL (short k = 0; k < 4; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        } else {
                            if (ii + tx < DK16) {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            for (short k = 0; k < 4 && ii + k < DK16; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        }
                    }

                    simdgroup_store(mqk, ss + 8*cc, SH, 0, false);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];

                // scale and apply the logitcap / mask
                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

                if (FC_flash_attn_ext_has_scap) {
                    s2 = args.logit_softcap*precise::tanh(s2);
                }

                // mqk = mqk + slope*mask
                if (blk_cur != 2) {
                    if (FC_flash_attn_ext_has_bias) {
                        s2 += s2_t(sm2[j*SH + tiisg])*slope;
                    } else {
                        s2 += s2_t(sm2[j*SH + tiisg]);
                    }
                }

                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

                const float  ms  = exp(m  - M[jj]);
                const float2 vs2 = exp(s2 - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

                // the P matrix from the paper (Q rows, C columns)
                ss2[j*SH/2 + tiisg] = vs2;

                if (DV4 % NW == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                        const short i = ii*NW + tiisg;

                        so4[j*PV4 + i] *= ms;
                    }
                } else {
                    for (short i = tiisg; i < DV4; i += NW) {
                        so4[j*PV4 + i] *= ms;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                // we can read directly from global memory
                if (is_same<vd4x4_t, v4x4_t>::value) {
                    static_assert(PV8 % NSG == 0, "");

                    constexpr short NO = PV8/NSG;

                    o8x8_t lo[NO];

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_load(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }

                    {
                        device const v_t * pv = (device const v_t *) (v + ic*args.nb21);

                        pv += 8*sgitg;

                        if (DV <= 64) {
                            FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
                                s8x8_t vs;
                                simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[2];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv[1], lo[2*ii + 1]);
                                }

                                pv  += 8*NS20;
                            }
                        } else {
                            constexpr short NC = (C/8)/2;

                            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                                s8x8_t vs[2];

                                simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
                                simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[4];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                                }

                                pv  += 2*8*NS20;
                            }
                        }
                    }

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_store(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }
                } else {
                    // TODO: this is the quantized V cache branch - not optimized yet

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    for (short cc = 0; cc < C/8; ++cc) {
                        s8x8_t vs;
                        simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                        for (short ii = 4*sgitg; ii < DV16; ii += 4*NSG) {
                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) (v + ((ic + 8*cc + ty)*args.nb21));

                            if (DV16%4 == 0) {
                                // no need for bound checks
                                {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                FOR_UNROLL (short k = 0; k < 4; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            } else {
                                if (ii + tx < DV16) {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < DV16; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            }
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (FC_flash_attn_ext_has_sinks) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

                M[jj] = simd_max(max(M[jj], s));

                const float ms = exp(m - M[jj]);
                const float vs = exp(s - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs);

                for (short i = tiisg; i < DV4; i += NW) {
                    so4[j*PV4 + i] *= ms;
                }
            }
        }
    }

    // store to global memory
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        if (iq1 + j >= args.ne01) {
            break;
        }

        device float4 * dst4 = (device float4 *) dst + ((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4;

        const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];

        if (DV4 % NW == 0) {
            FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                const short i = ii*NW + tiisg;

                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        } else {
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        }
    }

#undef NS10
#undef NS20
}

template<
    typename q_t,     // query types in shared memory
    typename q4_t,
    typename q8x8_t,
    typename k_t,     // key types in shared memory
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,     // value types in shared memory
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,    // Q*K types
    typename qk8x8_t,
    typename s_t,     // soft-max types
    typename s2_t,
    typename s8x8_t,
    typename o_t,     // attention accumulation types
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t, // key type in device memory
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // value type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,         // K head size
    short DV,         // V head size
    short Q  = OP_FLASH_ATTN_EXT_NQPSG, // queries per threadgroup
    short C  = OP_FLASH_ATTN_EXT_NCPSG> // cache items per threadgroup
kernel void kernel_flash_attn_ext(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
#define FWD_TMPL q_t, q4_t, q8x8_t, k_t, k4x4_t, k8x8_t, v_t, v4x4_t, v8x8_t, qk_t, qk8x8_t, s_t, s2_t, s8x8_t, o_t, o4_t, o8x8_t, kd4x4_t, nl_k, deq_k, vd4x4_t, nl_v, deq_v, DK, DV, Q, C
#define FWD_ARGS args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg
    switch (FC_flash_attn_ext_nsg) {
      // note: disabled cases to reduce library load time
      //case 1: kernel_flash_attn_ext_impl<FWD_TMPL, 1>(FWD_ARGS); break;
      //case 2: kernel_flash_attn_ext_impl<FWD_TMPL, 2>(FWD_ARGS); break;
        case 4: kernel_flash_attn_ext_impl<FWD_TMPL, 4>(FWD_ARGS); break;
        case 8: kernel_flash_attn_ext_impl<FWD_TMPL, 8>(FWD_ARGS); break;
    }
#undef FWD_TMPL
#undef FWD_ARGS
}

// TODO: this is quite ugly. in the future these types will be hardcoded in the kernel, but for now keep them as
//       template to be able to explore different combinations
//
#define FA_TYPES \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

#define FA_TYPES_BF \
    bfloat, bfloat4,   simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    float,             simdgroup_float8x8,  \
    float,  float2,    simdgroup_float8x8,  \
    half,   half4,     simdgroup_half8x8
    //float,  float4,    simdgroup_float8x8

#define FA_TYPES_F32 \
    half,   half4,     simdgroup_half8x8,  \
    float,  float4x4,  simdgroup_float8x8, \
    float,  float4x4,  simdgroup_float8x8, \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;

template [[host_name("kernel_flash_attn_ext_f32_dk32_dv32"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  32,  32>;
template [[host_name("kernel_flash_attn_ext_f32_dk40_dv40"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  40,  40>;
template [[host_name("kernel_flash_attn_ext_f32_dk48_dv48"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  48,  48>;
template [[host_name("kernel_flash_attn_ext_f32_dk64_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  64,  64>;
template [[host_name("kernel_flash_attn_ext_f32_dk72_dv72"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  72,  72>;
template [[host_name("kernel_flash_attn_ext_f32_dk80_dv80"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  80,  80>;
template [[host_name("kernel_flash_attn_ext_f32_dk96_dv96"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  96,  96>;
template [[host_name("kernel_flash_attn_ext_f32_dk112_dv112")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  112, 112>;
template [[host_name("kernel_flash_attn_ext_f32_dk128_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  128, 128>;
template [[host_name("kernel_flash_attn_ext_f32_dk192_dv192")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  192, 192>;
template [[host_name("kernel_flash_attn_ext_f32_dk192_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  192, 128>;
template [[host_name("kernel_flash_attn_ext_f32_dk256_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  256, 256>;
template [[host_name("kernel_flash_attn_ext_f32_dk320_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  320, 256>;
template [[host_name("kernel_flash_attn_ext_f32_dk512_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  512, 512>;
template [[host_name("kernel_flash_attn_ext_f32_dk576_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  576, 512>;

template [[host_name("kernel_flash_attn_ext_f16_dk32_dv32"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  32,  32>;
template [[host_name("kernel_flash_attn_ext_f16_dk40_dv40"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  40,  40>;
template [[host_name("kernel_flash_attn_ext_f16_dk48_dv48"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  48,  48>;
template [[host_name("kernel_flash_attn_ext_f16_dk64_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  64,  64>;
template [[host_name("kernel_flash_attn_ext_f16_dk72_dv72"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  72,  72>;
template [[host_name("kernel_flash_attn_ext_f16_dk80_dv80"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  80,  80>;
template [[host_name("kernel_flash_attn_ext_f16_dk96_dv96"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96,  96>;
template [[host_name("kernel_flash_attn_ext_f16_dk112_dv112")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  112, 112>;
template [[host_name("kernel_flash_attn_ext_f16_dk128_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  128, 128>;
template [[host_name("kernel_flash_attn_ext_f16_dk192_dv192")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 192>;
template [[host_name("kernel_flash_attn_ext_f16_dk192_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 128>;
template [[host_name("kernel_flash_attn_ext_f16_dk256_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  256, 256>;
template [[host_name("kernel_flash_attn_ext_f16_dk320_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  320, 256>;
template [[host_name("kernel_flash_attn_ext_f16_dk512_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  512, 512>;
template [[host_name("kernel_flash_attn_ext_f16_dk576_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  576, 512>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_bf16_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 32,  32>;
template [[host_name("kernel_flash_attn_ext_bf16_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 40,  40>;
template [[host_name("kernel_flash_attn_ext_bf16_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 48,  48>;
template [[host_name("kernel_flash_attn_ext_bf16_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 64,  64>;
template [[host_name("kernel_flash_attn_ext_bf16_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 72,  72>;
template [[host_name("kernel_flash_attn_ext_bf16_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 80,  80>;
template [[host_name("kernel_flash_attn_ext_bf16_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 96,  96>;
template [[host_name("kernel_flash_attn_ext_bf16_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 112, 112>;
template [[host_name("kernel_flash_attn_ext_bf16_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 128, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 192>;
template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 256, 256>;
template [[host_name("kernel_flash_attn_ext_bf16_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 320, 256>;
template [[host_name("kernel_flash_attn_ext_bf16_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 512, 512>;
template [[host_name("kernel_flash_attn_ext_bf16_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 576, 512>;
#endif

template [[host_name("kernel_flash_attn_ext_q4_0_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q4_1_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q5_0_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q5_1_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q8_0_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 576, 512>;

#undef FA_TYPES
#undef FA_TYPES_BF
#undef FA_TYPES_F32

constant bool FC_flash_attn_ext_vec_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 0)]];
constant bool FC_flash_attn_ext_vec_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_VEC + 1)]];
constant bool FC_flash_attn_ext_vec_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 2)]];
constant bool FC_flash_attn_ext_vec_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 3)]];
constant bool FC_flash_attn_ext_vec_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT_VEC + 4)]];

//constant float FC_flash_attn_ext_vec_scale         [[function_constant(FC_FLASH_ATTN_EXT_VEC + 10)]];
//constant float FC_flash_attn_ext_vec_max_bias      [[function_constant(FC_FLASH_ATTN_EXT_VEC + 11)]];
//constant float FC_flash_attn_ext_vec_logit_softcap [[function_constant(FC_FLASH_ATTN_EXT_VEC + 12)]];

constant int32_t FC_flash_attn_ext_vec_ns10 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 20)]];
constant int32_t FC_flash_attn_ext_vec_ns20 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 21)]];
constant int32_t FC_flash_attn_ext_vec_nsg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 22)]];
constant int32_t FC_flash_attn_ext_vec_nwg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 23)]];
constant bool    FC_flash_attn_ext_vec_has_sparse [[function_constant(FC_FLASH_ATTN_EXT_VEC + 5)]];

// compress the finite entries of each KQ mask row into a list of KV indices (ascending order),
// padded with -1 up to n_kv_max_padded (a multiple of OP_FLASH_ATTN_EXT_VEC_NCPSG)
// one threadgroup per mask row; the mask remains the single source of truth for the values
template<short NW>
kernel void kernel_flash_attn_ext_vec_idx_t(
        constant ggml_metal_kargs_flash_attn_ext_vec_idx & args,
        device const half * mask,
        device       int  * idx,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    constexpr short NLOCAL = 32; // max finite positions kept in registers per thread

    const int i1 = tgpig[0];
    const int i2 = tgpig[1];
    const int i3 = tgpig[2];

    device const half * pm  = (device const half *) ((device const char *) mask + i1*args.nb31 + i2*args.nb32 + i3*args.nb33);
    device int * pidx = idx + (((int64_t)i3*args.ne32 + i2)*args.ne31 + i1)*args.n_kv_max_padded;

    const int n  = args.ne30;
    const int q  = n/ntg.x;
    const int r  = n%ntg.x;

    // each thread handles a contiguous slice of the mask row
    const int r0 = q*tiitg + min((int) tiitg, r);
    const int r1 = r0 + q + (tiitg < r ? 1 : 0);

    // count the finite entries in the slice and keep their positions in registers (single mask read)
    int cnt = 0;  // total finite entries in the slice
    int nloc = 0; // finite entries kept in registers
    int local[NLOCAL];
    for (int i = r0; i < r1; ++i) {
        if (isfinite((float) pm[i])) {
            if (nloc < NLOCAL) {
                local[nloc] = i;
                nloc++;
            }
            cnt++;
        }
    }

    const short sgitg = tiitg/NW;
    const short tiisg = tiitg%NW;

    threadgroup int tcount[8];

    // simd_sum is a collective: all lanes must evaluate it
    const int sg_sum = simd_sum(cnt);
    if (tiisg == 0) {
        tcount[sgitg] = sg_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int total = 0;
    for (short s = 0; s < ntg.x/NW; ++s) {
        total += tcount[s];
    }

    // base offset of this thread's slice in the output list (exclusive scan within the simdgroup)
    int sg_base = 0;
    for (short s = 0; s < sgitg; ++s) {
        sg_base += tcount[s];
    }

    // exclusive prefix scan of the per-thread counts within the simdgroup
    int incl = cnt;
    for (int d = 1; d < NW; d <<= 1) {
        const int v = simd_shuffle_up(incl, d);
        if (tiisg >= d) {
            incl += v;
        }
    }
    const int base = sg_base + (incl - cnt);

    // write the finite positions in order; if the hint is violated, keep only the first n_kv_max entries
    int j = 0;
    for (; j < nloc && base + j < args.n_kv_max; ++j) {
        pidx[base + j] = local[j];
    }

    // a dense mask may have more than NLOCAL finite entries in a slice; re-read the mask to write the rest
    if (cnt > nloc && base + nloc < args.n_kv_max) {
        int j2 = 0;
        for (int i = r0; i < r1; ++i) {
            if (isfinite((float) pm[i])) {
                if (j2 >= nloc) {
                    pidx[base + j2] = i;
                }
                j2++;
                if (base + j2 >= args.n_kv_max) {
                    break;
                }
            }
        }
    }

    // pad the tail of the list with -1
    const int count = min(total, args.n_kv_max);
    for (int i = count + tiitg; i < args.n_kv_max_padded; i += ntg.x) {
        pidx[i] = -1;
    }
}

// the scan and the per-simdgroup counts are width-dependent, so the 64-lane cards need
// their own instantiation instead of the 32-lane default
typedef decltype(kernel_flash_attn_ext_vec_idx_t<N_SIMDWIDTH>) flash_attn_ext_vec_idx_t;

template [[host_name("kernel_flash_attn_ext_vec_idx")]] kernel flash_attn_ext_vec_idx_t
    kernel_flash_attn_ext_vec_idx_t<32>;
template [[host_name("kernel_flash_attn_ext_vec_idx_w64")]] kernel flash_attn_ext_vec_idx_t
    kernel_flash_attn_ext_vec_idx_t<64>;

template<
    typename q4_t,  // query types in shared memory
    typename k4_t,  // key types in shared memory
    typename v4_t,  // value types in shared memory
    typename qk_t,  // Q*K types
    typename s_t,   // soft-max types
    typename s4_t,
    typename o4_t,  // attention accumulation types
    typename kd4_t, // key type in device memory
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t, // value type in device memory
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,       // K head size
    short DV,       // V head size
    short NE = 4,   // head elements per thread
    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,  // queries per threadgroup
    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG>  // cache items per threadgroup

kernel void kernel_flash_attn_ext_vec(
        constant ggml_metal_kargs_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        device const char * idx,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

#define NWG  (FC_flash_attn_ext_vec_nwg)
#define NSG  (FC_flash_attn_ext_vec_nsg)

#define NS10 (FC_flash_attn_ext_vec_ns10)
#define NS20 (FC_flash_attn_ext_vec_ns20)

    const short iwg = tgpig[2]%NWG;

    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;

    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;

    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE; // note: this can be adjusted to support different head sizes and simdgroup work loads
    constexpr short SH  = 4*Q*C; // shared memory per simdgroup

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

  //const short T = PK + NSG*SH; // shared memory size per query in (half)

  //threadgroup q_t   * sq  = (threadgroup q_t   *) (shmem_f16 +                          0*PK); // holds the query data
    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                          0*PK); // same as above but in q4_t
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH         + Q*NSG*PK); // scratch buffer for attention
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH         + Q*NSG*PK); // same as above but in s4_t
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*Q*C + Q*NSG*PK); // scratch buffer for mask
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*Q*PV       + Q*NSG*PK + NSG*SH); // scratch buffer for the results

    // store the result for all queries in shared memory (the O matrix from the paper)
    so4 += tiisg;

    {
        q += iq1*Q*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // load Q query rows to shared memory
    {
        for (short qq = 0; qq < Q; ++qq) {
            const int iq1_q = iq1*Q + qq;
            device const float4 * q4 = (device const float4 *) ((device const char *) q + qq*args.nb01);
            if (iq1_q < args.ne01) {
                for (short i = tiisg; i < PK4; i += NW) {
                    if (i < DK4) {
                        sq4[qq*PK4 + i] = (q4_t) q4[i];
                    } else {
                        sq4[qq*PK4 + i] = (q4_t) 0.0f;
                    }
                }
            } else {
                for (short i = tiisg; i < PK4; i += NW) {
                    sq4[qq*PK4 + i] = (q4_t) 0.0f;
                }
            }
        }
    }

    // zero out so
    for (short qq = 0; qq < Q; ++qq) {
        for (short i = 0; i < DV4/NL; ++i) {
            so4[qq*DV4 + i*NL] = (o4_t) 0.0f;
        }
    }

    // zero out shared memory SH
    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S[Q];
        float M[Q];
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            S[qq] = 0.0f;
            M[qq] = -FLT_MAX/2;
        }

        // thread indices inside the simdgroup
        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        // pointer to the mask
        device const half * pm_base = (device const half *) (mask + iq1*Q*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

        // sparse indices: the list of finite mask entries per query row
        // the sparse path requires Q == 1 (enforced by the host)
        device const int * pidx = nullptr;
        if (FC_flash_attn_ext_vec_has_sparse) {
            pidx = (device const int *) idx +
                ((int64_t)(iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*args.ne31*args.n_kv_max_padded + (iq1%args.ne31)*args.n_kv_max_padded;
        }

        float slope = 1.0f;

        // ALiBi
        if (FC_flash_attn_ext_vec_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            device const half * pm[Q];
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                // padded query rows clamp to row 0 of the mask to avoid OOB; their scores
                // are forced to -inf below, so the values never affect the result.
                pm[qq] = pm_base + ((iq1*Q + qq) < args.ne01 ? qq*(args.nb31/sizeof(half)) : -iq1*Q*(args.nb31/sizeof(half)));
            }

            // the last partial chunk uses the pad buffer as source
            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_vec_has_mask) {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        if (ic + tiisg >= args.ne11) {
                            sm[qq*C + tiisg] = -MAXHALF;
                        }
                    }
                } else {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        pm[qq] = (device const half *) (mask) +
                            (iq1*Q + qq)*C +
                            (iq2%args.ne32)*(C*args.ne31) +
                            (iq3%args.ne33)*(C*args.ne31*args.ne32);
                    }
                }

                ic = 0;
            }

            if (FC_flash_attn_ext_vec_has_mask) {
                if (FC_flash_attn_ext_vec_has_sparse) {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        const int i11 = pidx[ic + tiisg];
                        if ((iq1*Q + qq) < args.ne01 && i11 >= 0) {
                            sm[qq*C + tiisg] = pm[qq][i11];
                        } else {
                            sm[qq*C + tiisg] = -MAXHALF;
                        }
                    }
                } else {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        if ((iq1*Q + qq) < args.ne01) {
                            sm[qq*C + tiisg] = pm[qq][ic + tiisg];
                        } else {
                            sm[qq*C + tiisg] = -MAXHALF;
                        }
                    }
                }
            } else {
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    if ((iq1*Q + qq) >= args.ne01) {
                        sm[qq*C + tiisg] = -MAXHALF;
                    }
                }
            }

            // skip -INF mask
            {
                bool any_finite = false;
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    if (simd_max(sm[qq*C + tiisg]) > -MAXHALF) {
                        any_finite = true;
                    }
                }
                if (!any_finite) {
                    continue;
                }
            }

            // Q*K^T
            {
                device      const k4_t * pk4 = nullptr;

                if (!FC_flash_attn_ext_vec_has_sparse) {
                    pk4 = (device const k4_t *) (k + ic*args.nb11);

                    pk4 += ty*NS10/4 + tx;
                }

                qk_t mqk[Q][C/NE];
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        mqk[qq][cc] = 0.0f;
                    }
                }

                // each simdgroup processes Q queries and NE (NW/NL) cache elements
                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    if (FC_flash_attn_ext_vec_has_sparse) {
                        // the KV rows are gathered from the index list; -1 entries are padding
                        const int i11 = pidx[ic + NE*cc + ty];
                        if (i11 >= 0) {
                            if (is_same<kd4_t, k4_t>::value) {
                                device const k4_t * pk4s = (device const k4_t *) (k + i11*args.nb11) + tx;
                                FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                                    const k4_t k_elem = pk4s[ii*NL];
                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                        mqk[qq][cc] += dot((float4) k_elem, (float4) sq4[qq*PK4 + ii*NL + tx]);
                                    }
                                }
                            } else {
                                device const kd4_t * pk = (device const kd4_t *) (k + i11*args.nb11);

                                k4_t mk;

                                FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                                    const short i = ii*NL + tx;

                                    deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                        mqk[qq][cc] += dot((float4) mk, (float4) sq4[qq*PK4 + i]);
                                    }
                                }
                            }
                        }
                    } else if (is_same<kd4_t, k4_t>::value) {
                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const k4_t k_elem = pk4[cc*NE*NS10/4 + ii*NL];
                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                mqk[qq][cc] += dot((float4) k_elem, (float4) sq4[qq*PK4 + ii*NL + tx]);
                            }
                        }
                    } else {
                        device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE*cc + ty)*args.nb11));

                        k4_t mk;

                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                mqk[qq][cc] += dot((float4) mk, (float4) sq4[qq*PK4 + i]);
                            }
                        }
                    }

                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        if (NE == 1) {
                            mqk[qq][cc] = simd_sum(mqk[qq][cc]);
                        } else {
                            // simdgroup reduce (NE = 4)
                            // [ 0 ..  7] -> [ 0]
                            // [ 8 .. 15] -> [ 8]
                            // [16 .. 23] -> [16]
                            // [24 .. 31] -> [24]
                            if (NE <= 1) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc], 16);
                            }
                            if (NE <= 2) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  8);
                            }
                            if (NE <= 4) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  4);
                            }
                            if (NE <= 8) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  2);
                            }
                            if (NE <= 16) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  1);
                            }

                            // broadcast
                            mqk[qq][cc] = simd_shuffle(mqk[qq][cc], NL*ty);
                        }
                    }
                }

                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    if (FC_flash_attn_ext_vec_has_mask &&
                       !FC_flash_attn_ext_vec_has_scap &&
                       !FC_flash_attn_ext_vec_has_bias) {
                        ss[qq*C + NE*tx + ty] = fma(mqk[qq][tx], args.scale, (qk_t) sm[qq*C + NE*tx + ty]);
                    } else {
                        mqk[qq][tx] *= args.scale;

                        if (FC_flash_attn_ext_vec_has_scap) {
                            mqk[qq][tx] = args.logit_softcap*precise::tanh(mqk[qq][tx]);
                        }

                        if (FC_flash_attn_ext_vec_has_bias) {
                            mqk[qq][tx] += (qk_t) sm[qq*C + NE*tx + ty]*slope;
                        } else {
                            mqk[qq][tx] += (qk_t) sm[qq*C + NE*tx + ty];
                        }

                        ss[qq*C + NE*tx + ty] = mqk[qq][tx];
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            {
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    const float m = M[qq];
                    const float s = ss[qq*C + tiisg];

                    M[qq] = simd_max(max(M[qq], s));

                    const float ms = exp(m - M[qq]);
                    const float vs = exp(s - M[qq]);

                    S[qq] = S[qq]*ms + simd_sum(vs);

                    // the P matrix from the paper (Q rows, C columns)
                    ss[qq*C + tiisg] = vs;

                    // O = diag(ms)*O
                    if ((DV4/NL % NW == 0) || ty == 0) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            so4[qq*DV4 + ii*NL] *= ms;
                        }
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                o4_t lo[Q][DV4/NL];
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        lo[qq][ii] = 0.0f;
                    }
                }

                if (FC_flash_attn_ext_vec_has_sparse) {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        // the KV rows are gathered from the index list; -1 entries are padding
                        const int i11 = pidx[ic + NE*cc + ty];
                        if (i11 >= 0) {
                            if (is_same<vd4_t, v4_t>::value) {
                                device const v4_t * pv4 = (device const v4_t *) (v + i11*args.nb21);

                                pv4 += tx;

                                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                                    const v4_t v_elem = pv4[ii*NL];
                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                        lo[qq][ii] += o4_t(float4(v_elem)*float4(ss[qq*C + cc*NE + ty]));
                                    }
                                }
                            } else {
                                device const vd4_t * pv4 = (device const vd4_t *) (v + i11*args.nb21);

                                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                                    const short i = ii*NL + tx;

                                    v4_t mv;

                                    deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                        lo[qq][ii] += o4_t(float4(mv)*float4(ss[qq*C + cc*NE + ty]));
                                    }
                                }
                            }
                        }
                    }
                } else if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);

                    pv4 += ty*NS20/4 + tx;

                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const v4_t v_elem = pv4[cc*NE*NS20/4 + ii*NL];
                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                lo[qq][ii] += o4_t(float4(v_elem)*float4(ss[qq*C + cc*NE + ty]));
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE*cc + ty)*args.nb21));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                lo[qq][ii] += o4_t(float4(mv)*float4(ss[qq*C + NE*cc + ty]));
                            }
                        }
                    }
                }

                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        if (NE > 1) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0], 16);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1], 16);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2], 16);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3], 16);
                        }

                        if (NE > 2) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  8);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  8);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  8);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  8);
                        }

                        if (NE > 4) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  4);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  4);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  4);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  4);
                        }

                        if (NE > 8) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  2);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  2);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  2);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  2);
                        }

                        if (NE > 16) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  1);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  1);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  1);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  1);
                        }
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            so4[qq*DV4 + ii*NL] += lo[qq][ii];
                        }
                    }
                }
            }
        }

        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                const float m = M[qq];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

                M[qq] = simd_max(max(M[qq], s));

                const float ms = exp(m - M[qq]);
                const float vs = exp(s - M[qq]);

                S[qq] = S[qq]*ms + simd_sum(vs);

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[qq*DV4 + ii*NL] *= ms;
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        if (tiisg == 0) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                ss[2*qq + 0] = (s_t) S[qq];
                ss[2*qq + 1] = (s_t) M[qq];
            }
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduce
    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                const float S0 = ss[                2*qq + 0];
                const float S1 = ss[r*(SH/2) +      2*qq + 0];

                const float M0 = ss[                2*qq + 1];
                const float M1 = ss[r*(SH/2) +      2*qq + 1];

                const float Mx  = max(M0, M1);

                const float ms0 = exp(M0 - Mx);
                const float ms1 = exp(M1 - Mx);

                const float Sx  = S0*ms0 + S1*ms1;

                if (tiisg == 0) {
                    ss[2*qq + 0] = Sx;
                    ss[2*qq + 1] = Mx;
                }

                // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
                for (short i = tiisg; i < DV4; i += NW) {
                    so4[qq*DV4 + i] = so4[qq*DV4 + i]*ms0 + so4[qq*DV4 + i + r*Q*PV4]*ms1;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG; // the S and M are stored after the results

        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            const int iq1_q = iq1*Q + qq;
            if (iq1_q >= args.ne01) {
                continue;
            }

            const int64_t rid = iq3*args.ne2*args.ne1 + iq2 + iq1_q*args.ne1;

            const float Sval = NWG == 1 ? (ss[2*qq + 0] == 0.0f ? 0.0f : 1.0f/ss[2*qq + 0]) : 1.0f;

            // interleave the workgroup data
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[qq*DV4 + i]*Sval;
            }

            // store S and M
            if (NWG > 1) {
                if (tiisg == 0) {
                    dst1[rid*(2*NWG) + 2*iwg + 0] = ss[2*qq + 0];
                    dst1[rid*(2*NWG) + 2*iwg + 1] = ss[2*qq + 1];
                }
            }
        }
    }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
}

// note: I think the s_t can be half instead of float, because the Q*K scaling is done before storing to shared mem
//       in the other (non-vec) kernel, we need s_t to also be float because we scale during the soft_max
//
#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk32_dv32")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 32, 32, 4>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk64_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 64, 64, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk96_dv96")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 96, 96, 4>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk128_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 128, 128, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk192_dv192")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 192, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk192_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 128, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk256_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 256, 256, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk320_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 320, 256, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 512, 512, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk576_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 576, 512, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 4>;


#undef FA_TYPES
#undef FA_TYPES_F32

constant int32_t FC_flash_attn_ext_vec_reduce_DV  [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];
constant int32_t FC_flash_attn_ext_vec_reduce_NWG [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 1)]];

kernel void kernel_flash_attn_ext_vec_reduce(
        constant ggml_metal_kargs_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort sgptg[[threads_per_simdgroup]],
        ushort tptg[[threads_per_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t rid = tgpig;

    const short iwg = tiisg;
    const bool active = iwg < NWG;
    const short nsg = tptg/sgptg;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;

    // NWG is 32. On wave64 the upper lanes contribute neutral values instead
    // of indexing beyond the 32 partial results.
    float S = active ? ss[rid*(2*NWG) + 2*iwg + 0] : 0.0f;
    float M = active ? ss[rid*(2*NWG) + 2*iwg + 1] : -1.0e30f;

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += nsg) {
        const float4 partial = active ? htmp4[i*NWG + iwg] : float4(0.0f);
        const float4 v = simd_sum(partial*ms);

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }

#undef NWG
#undef DV
}

template<
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread half4x4 &)>
kernel void kernel_lightning_indexer(
        constant ggml_metal_kargs_lightning_indexer & args,
        device const char * q,
        device const char * k,
        device const char * w,
        device const char * m,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK    = OP_LIGHTNING_INDEXER_DK;
    constexpr short NH    = OP_LIGHTNING_INDEXER_NH;
    constexpr short NHPTG = OP_LIGHTNING_INDEXER_NHPTG;
    constexpr short NKPSG = OP_LIGHTNING_INDEXER_NKPSG;
    constexpr short NSG   = OP_LIGHTNING_INDEXER_NSG;
    constexpr short NBPTG = OP_LIGHTNING_INDEXER_NBPTG;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;

    constexpr short NK  = NKPSG*NSG; // keys    per threadgroup
    constexpr short NTG = 32*NSG;    // threads per threadgroup

    const int i_stream = tgpig.z;
    const int i_kv_0   = tgpig.x*NK;            // first key of this threadgroup
    const int i_kv     = i_kv_0 + sgitg*NKPSG;  // first key of this simdgroup

    threadgroup half sk[NK * DK16 * 16];
    threadgroup half4x4 * sk4x4 = (threadgroup half4x4 *) sk;

    for (short i = tiitg; i < NK*DK16; i += NTG) {
        const short ik  = i/DK16;
        const short i16 = i%DK16;

        half4x4 tmp;

        if (i_kv_0 + ik < args.n_kv) {
            device const kd4x4_t * kr = (device const kd4x4_t *) (k + (i_kv_0 + ik)*args.nbk2 + i_stream*args.nbk3);

            deq_k(kr + i16/nl_k, i16%nl_k, tmp);
        } else {
            FOR_UNROLL (short j = 0; j < 4; ++j) {
                tmp[j] = half4(0.0h);
            }
        }

        sk4x4[i] = tmp;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // K tile of this simdgroup, transposed to [DK, NKPSG]
    simdgroup_half8x8 mk[DK8];

    FOR_UNROLL (short i = 0; i < DK8; ++i) {
        simdgroup_load(mk[i], sk + sgitg*NKPSG*DK + 8*i, DK, 0, true);
    }

    threadgroup half4   sq4[NHPTG*DK4];
    threadgroup half  * sq = (threadgroup half *) sq4;

    threadgroup float sw [NHPTG];
    threadgroup float sqk[NSG*NHPTG*NKPSG];

    const int i_batch_0 = tgpig.y*NBPTG;
    const int n_batch   = min((int) NBPTG, args.n_batch - i_batch_0);

    for (short ib = 0; ib < n_batch; ++ib) {
        const int i_batch = i_batch_0 + ib;

        device const char * pq = q + i_batch*args.nbq2 + i_stream*args.nbq3;
        device const char * pw = w + i_batch*args.nbw1 + i_stream*args.nbw3;

        float score = 0.0f;

        FOR_UNROLL (short i_head = 0; i_head < NH; i_head += NHPTG) {
            // stage the Q tile [DK, NHPTG] and the (prescaled) head weights
            for (short i = tiitg; i < NHPTG*DK4; i += NTG) {
                const short ih = i/DK4;
                const short i4 = i%DK4;

                device const float4 * q4 = (device const float4 *) (pq + (i_head + ih)*args.nbq1);

                sq4[ih*DK4 + i4] = half4(q4[i4]);
            }

            if (tiitg < NHPTG) {
                sw[tiitg] = ((device const float *) pw)[i_head + tiitg];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            FOR_UNROLL (short i = 0; i < DK8; ++i) {
                simdgroup_half8x8 mq;

                simdgroup_load(mq, sq + 8*i, DK, 0, false);
                simdgroup_multiply_accumulate(mqk, mq, mk[i], mqk);
            }

            threadgroup float * pqk = sqk + sgitg*NHPTG*NKPSG;

            simdgroup_store(mqk, pqk, NKPSG, 0, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // one lane per key: ReLU, apply the head weight and accumulate over the head tile
            if (tiisg < NKPSG) {
                FOR_UNROLL (short ih = 0; ih < NHPTG; ++ih) {
                    score += max(pqk[ih*NKPSG + tiisg], 0.0f)*sw[ih];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tiisg < NKPSG) {
            const int ik = i_kv + tiisg;
            if (ik < args.n_kv) {
                device const half  * pm = (device const half  *) (m   + i_batch*args.nbm1 + (i_stream % args.mask_ne3)*args.nbm3);
                device       float * pd = (device       float *) (dst + i_batch*args.nb1  + i_stream*args.nb3);

                pd[ik] = score + (float) pm[ik];
            }
        }
    }
}

typedef decltype(kernel_lightning_indexer<half4x4, 1, dequantize_f16>) kernel_lightning_indexer_t;

template [[host_name("kernel_lightning_indexer_f32")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<float4x4, 1, dequantize_f32>;
template [[host_name("kernel_lightning_indexer_f16")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<half4x4,  1, dequantize_f16>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_lightning_indexer_bf16")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<bfloat4x4, 1, dequantize_bf16>;
#endif

template [[host_name("kernel_lightning_indexer_q4_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_lightning_indexer_q4_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_lightning_indexer_q5_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_lightning_indexer_q5_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_lightning_indexer_q8_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q8_0, 2, dequantize_q8_0>;

// vec kernel above miscompiles. Generalized over the K/V element type via the
// dequantize_*_t4 helpers; standard kernels cover their native dimensions and
// TurboQuant kernels cover padded head dimensions 128/256/384/512/640.

template<
    short DK, short NSG,
    typename k_t, short NKK, void (*deq_k)(device const k_t *, short, thread float4 &),
    typename v_t, short NKV, void (*deq_v)(device const v_t *, short, thread float4 &),
    short NW = 32, short NWG = 1, short SUBWP = 0, short DVP = 0>
kernel void kernel_flash_attn_ext_vec_amd(
        constant ggml_metal_kargs_flash_attn_ext_amd & args [[buffer(0)]],
        device const char * q     [[buffer(1)]],
        device const char * k     [[buffer(2)]],
        device const char * v     [[buffer(3)]],
        device const char * mask  [[buffer(4)]],
        device const char * sinks [[buffer(5)]],
        device const char * pad   [[buffer(6)]],
        device       char * dst   [[buffer(7)]],
        device const int  * idx   [[buffer(8)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short DV  = DVP ? DVP : DK;   // MLA has DK != DV (K carries the rope tail, V is the bare latent)
    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;   // float4 per head row (32 for DV=128, 64 for DV=256)
    // chunks per lane scale with the simd width; on wave64 with DK=128 the upper
    // lanes own no chunk (guarded below) but still join the simd_sum reductions.
    // heads narrower than the simd width would leave the upper lanes idle, so the
    // simdgroup splits into RPS sub-groups that each take their own cache row
    constexpr short SUBW = SUBWP ? SUBWP : (DV4 < NW ? DV4 : NW);
    constexpr short RPS  = NW/SUBW;
    constexpr short NCV  = (DV4 + SUBW - 1)/SUBW;
    constexpr short NCK  = (DK4 + SUBW - 1)/SUBW;
    constexpr short NG   = NSG*RPS;
    constexpr short GPL  = (NG + NW - 1)/NW;
    // NSG (simdgroups per threadgroup) is a template parameter, kept in sync with
    // the host dispatch. More simdgroups split the KV stream into shorter serial
    // loops, which is a large decode win at depth; it is chosen per head dim so
    // the sh_acc threadgroup buffer (NSG*DV4 float4) stays at 16 KiB: 32 for
    // DK=128, 16 for DK=256.

    // padded so the merge below, which reads down a column with the lane as the
    // simdgroup index, does not land every lane on the same bank
    constexpr short SAS = DV4 + 1;
    threadgroup float4 sh_acc[NG*SAS];
    threadgroup float  sh_m[NG];
    threadgroup float  sh_s[NG];

    const int iq1 = tgpig[0];              // query token
    const int iq2 = tgpig[1];              // query head
    const int iq3 = tgpig[2]/NWG;          // batch
    const short iwg = NWG == 1 ? 0 : tgpig[2]%NWG; // this workgroup's slice of the cache

    device const float4 * q4 = (device const float4 *) (q + iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03);

    // grouped-query attention: map query head -> kv head
    const int ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const int ikv3 = iq3 / (args.ne03 / args.ne_12_3);

    device const char * kp = k + ikv2*args.nb12 + ikv3*args.nb13;
    device const char * vp = v + ikv2*args.nb22 + ikv3*args.nb23;

    // Without a mask ne32/ne33 are 0, so the index math would divide by zero.
    device const half * pm = (device const half *) mask;
    if (args.has_mask) {
        pm = (device const half *) (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    // sparse: the index list is laid out per (row, head, batch), same as the mask
    // sparse when the host filled the index list; uniform across the threadgroup
    const bool has_sparse = args.n_kv_max_padded > 0;

    device const int * pidx = nullptr;
    if (has_sparse) {
        pidx = idx +
            ((int64_t)(iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*args.ne31*args.n_kv_max_padded +
            (iq1%args.ne31)*args.n_kv_max_padded;
    }

    const short sub  = tiisg/SUBW;
    const short lch  = tiisg%SUBW;
    const short gidx = sgitg*RPS + sub;

    float4 myq[NCK];  // this lane's slice(s) of the query
    for (short c = 0; c < NCK; ++c) {
        const short ch = lch + c*SUBW;
        myq[c] = ch < DK4 ? q4[ch] : float4(0.0f);
    }

    float4 acc[NCV];  // this lane's slice(s) of the output
    for (short c = 0; c < NCV; ++c) {
        acc[c] = float4(0.0f);
    }
    float m = -FLT_MAX/2;
    float s = 0.0f;

    // sparse walks the compacted index list instead of the whole cache; a padded slot is -1
    const int n_kv = has_sparse ? args.n_kv_max_padded : args.ne11;

    for (int j = (iwg*NSG + sgitg)*RPS + sub; j < n_kv; j += NWG*NG) {
        const int i11 = has_sparse ? pidx[j] : j;

        // no early-out: the sub-groups of a simdgroup sit on different rows, so the
        // skip is not uniform and the shuffle reduction below would diverge
        const float mv = (has_sparse && i11 < 0) ? -MAXHALF
                       : (args.has_mask ? (float) pm[i11] : 0.0f);
        const bool  msk = mv <= -1e30f;
        if (RPS == 1 && msk) {
            continue;
        }

        // a padded slot keeps running through the body to hold the reduction uniform, and its
        // weight ends up zero, but the address is still formed: read row 0 instead of -1
        const int i11a = i11 < 0 ? 0 : i11;

        device const k_t * kb = (device const k_t *) (kp + (uint64_t) i11a*args.nb11);
        float pl = 0.0f;
        for (short c = 0; c < NCK; ++c) {
            const short ch = lch + c*SUBW;
            if (ch < DK4) {
                float4 kreg;
                deq_k(kb + ch/NKK, ch%NKK, kreg);
                pl += dot(myq[c], kreg);
            }
        }
        if (SUBW < NW) {
            for (ushort o = 1; o < SUBW; o <<= 1) {
                pl += simd_shuffle_xor(pl, o);
            }
        } else {
            pl = simd_sum(pl);
        }
        const float p = pl; // full DK-dim Q.K

        const float pp   = p*args.scale + mv;
        const float nm   = msk ? m : max(m, pp);
        const float corr = exp(m  - nm);
        const float w    = msk ? 0.0f : exp(pp - nm);

        device const v_t * vb = (device const v_t *) (vp + (uint64_t) i11a*args.nb21);
        for (short c = 0; c < NCV; ++c) {
            const short ch = lch + c*SUBW;
            if (ch < DV4) {
                float4 vreg;
                deq_v(vb + ch/NKV, ch%NKV, vreg);
                acc[c] = acc[c]*corr + w*vreg;
            }
        }
        s = s*corr + w;
        m = nm;
    }

    // publish this simdgroup's partials (acc, m, s) and merge in simdgroup 0
    for (short c = 0; c < NCV; ++c) {
        const short ch = lch + c*SUBW;
        if (ch < DV4) {
            sh_acc[gidx*SAS + ch] = acc[c];
        }
    }
    if (lch == 0) {
        sh_m[gidx] = m;
        sh_s[gidx] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // lane g carries group g's (m, s), so the merge is two simd reductions and every
    // simdgroup takes a slice, instead of simdgroup 0 walking NSG entries alone
    float lm = -FLT_MAX/2;
    for (short r = 0; r < GPL; ++r) {
        const short g = tiisg*GPL + r;
        if (g < NG) {
            lm = max(lm, sh_m[g]);
        }
    }
    const float gm = simd_max(lm);

    float fr[GPL];
    float den = 0.0f;
    for (short r = 0; r < GPL; ++r) {
        const short g = tiisg*GPL + r;
        fr[r] = g < NG ? exp(sh_m[g] - gm) : 0.0f;
        den  += g < NG ? fr[r]*sh_s[g] : 0.0f;
    }
    den = simd_sum(den);

    float scorr = 1.0f;
    if (args.has_sinks) {
        const float sk = ((device const float *) sinks)[iq2];
        const float nm = max(gm, sk);
        scorr = exp(gm - nm);
        den   = den*scorr + exp(sk - nm);
    }

    const float inv = (den > 0.0f ? 1.0f/den : 0.0f)*scorr;

    const int64_t rid = (int64_t) iq3*args.ne2*args.ne1 + iq2 + (int64_t) iq1*args.ne1;

    if (NWG == 1) {
        device float4 * dst4 = (device float4 *) dst;
        for (short i = sgitg; i < DV4; i += NSG) {
            float4 v = 0.0f;
            for (short r = 0; r < GPL; ++r) {
                const short g = tiisg*GPL + r;
                if (g < NG) {
                    v += fr[r]*sh_acc[g*SAS + i];
                }
            }
            v = simd_sum(v);
            if (tiisg == 0) {
                dst4[rid*DV4 + i] = v*inv;
            }
        }
    } else {
        // unnormalized partial in this slice's own max frame, plus (S, M), which is
        // the layout kernel_flash_attn_ext_vec_reduce expects
        const int64_t nrows = (int64_t) args.ne1*args.ne2*args.ne3;

        device float4 * htmp4 = (device float4 *) dst + rid*DV4*NWG;
        device float  * ss    = (device float  *) dst + nrows*DV4*4*NWG;

        for (short i = sgitg; i < DV4; i += NSG) {
            float4 v = 0.0f;
            for (short r = 0; r < GPL; ++r) {
                const short g = tiisg*GPL + r;
                if (g < NG) {
                    v += fr[r]*sh_acc[g*SAS + i];
                }
            }
            v = simd_sum(v);
            if (tiisg == 0) {
                htmp4[i*NWG + iwg] = v;
            }
        }
        if (sgitg == 0 && tiisg == 0) {
            ss[rid*(2*NWG) + 2*iwg + 0] = den;
            ss[rid*(2*NWG) + 2*iwg + 1] = gm;
        }
    }
}

// ===== TurboQuant store: centroids + quantize + set_rows kernels =====

typedef decltype(kernel_flash_attn_ext_vec_amd<128, 32, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4>) flash_attn_ext_vec_amd_t;

// narrowing subw doubles the chunks per lane; the register pressure caps the pipeline,
// so pairs with q4_0 dequant take fewer simdgroups (see the PAIR macros below).
#define FA_AMD_INST_S(dk, nsg, subw, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_" nm)]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 32, 1, subw>;
#define FA_AMD_INST_WG(dk, nsg, subw, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_" nm "_wg32")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 32, 32, subw>;
// dk64 narrows to subw=8 like dk128; clean pairs keep n64=32, q4_0 in K or V drops to 24.
#define FA_AMD_PAIR_S(n64, s128, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(256, 16, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S( 64, n64, 8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S(128, 16, s128, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S(256, 16, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S(512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    /* the narrow heads need the split too, but only at depths the host gates apart */ \
    FA_AMD_INST_WG( 64, n64, 8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(128, 16, s128, nm, kt, nkk, dkf, vt, nkv, dvf)
// q4_0 keys: dk64 and dk128 both take 24 simdgroups for the heavier key dequant
#define FA_AMD_PAIR_Q4(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(256, 16, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S( 64, 24, 8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S(128, 24, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S(256, 16, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S(512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG( 64, 24, 8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(128, 24, 16, nm, kt, nkk, dkf, vt, nkv, dvf)

// symmetric pairs
FA_AMD_PAIR_S(32, 8,  "kf16_vf16",         half4,          1,  dequantize_f16_t4,      half4,          1,  dequantize_f16_t4)
FA_AMD_PAIR_S(32, 8,  "kq8_0_vq8_0",       block_q8_0,     8,  dequantize_q8_0_t4,     block_q8_0,     8,  dequantize_q8_0_t4)
FA_AMD_PAIR_Q4("kq4_0_vq4_0",       block_q4_0,     8,  dequantize_q4_0_t4,     block_q4_0,     8,  dequantize_q4_0_t4)
// asymmetric standard pairs (any mix of f16 / q8_0 / q4_0 for K and V), so every
// standard K/V combination runs on the GPU instead of falling back to the CPU.
FA_AMD_PAIR_S(32, 8,  "kq8_0_vf16",        block_q8_0,     8,  dequantize_q8_0_t4,     half4,          1,  dequantize_f16_t4)
FA_AMD_PAIR_Q4("kq4_0_vf16",        block_q4_0,     8,  dequantize_q4_0_t4,     half4,          1,  dequantize_f16_t4)
FA_AMD_PAIR_S(32, 8,  "kf16_vq8_0",        half4,          1,  dequantize_f16_t4,      block_q8_0,     8,  dequantize_q8_0_t4)
FA_AMD_PAIR_S(24, 8,  "kf16_vq4_0",        half4,          1,  dequantize_f16_t4,      block_q4_0,     8,  dequantize_q4_0_t4)
FA_AMD_PAIR_S(24, 8,  "kq8_0_vq4_0",       block_q8_0,     8,  dequantize_q8_0_t4,     block_q4_0,     8,  dequantize_q4_0_t4)
FA_AMD_PAIR_Q4("kq4_0_vq8_0",       block_q4_0,     8,  dequantize_q4_0_t4,     block_q8_0,     8,  dequantize_q8_0_t4)
// MLA (deepseek2): K is the latent plus the 64-wide rope tail (DK=576); V is the
// bare latent (DV=512). Same cache tensor, so K and V share one type.
#define FA_AMD_INST_ASYM(dk, dv, nsg, subw, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_dv" #dv "_" nm)]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 32, 1, subw, dv>; \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_dv" #dv "_" nm "_wg32")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 32, 32, subw, dv>;
FA_AMD_INST_ASYM(576, 512, 16, 32, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_INST_ASYM(576, 512, 16, 32, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_INST_ASYM(320, 256, 16, 16, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_INST_ASYM(320, 256, 16, 16, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
#undef FA_AMD_INST_ASYM
#define FA_AMD_INST_ASYM_W64(dk, dv, nsg, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_dv" #dv "_" nm "_w64")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 64, 1, 64, dv>; \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_dv" #dv "_" nm "_w64_wg32")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 64, 32, 64, dv>;
FA_AMD_INST_ASYM_W64(576, 512, 8, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_INST_ASYM_W64(576, 512, 8, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_INST_ASYM_W64(320, 256, 8, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_INST_ASYM_W64(320, 256, 8, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
#undef FA_AMD_INST_ASYM_W64
// wave64 (AMD GCN/Vega): 64-lane simdgroups, so nsg is the host's base count for the
// head halved (host dispatches simd_width*nsg); a mismatch corrupts the merge. Host
// appends "_w64". Untested on real GCN/Vega, needs a hardware pass before release.
#define FA_AMD_INST_W64(dk, nsg, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_" nm "_w64")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 64>;
#define FA_AMD_INST_W64_S(dk, nsg, subw, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_" nm "_w64")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 64, 1, subw>;
#define FA_AMD_INST_W64_WG(dk, nsg, subw, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_" nm "_w64_wg32")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 64, 32, subw>;
#define FA_AMD_PAIR_W64(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_WG(256, 8, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_WG(512, 8, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S( 64, 16,  8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(128,  8,  8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(256,  8, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(512,  8, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_WG( 64, 16,  8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_WG(128,  8,  8, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_PAIR_W64("kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_PAIR_W64("kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4)

// DK=72 (vision towers: qwen3vl_merger and SigLIP are 1152/16). f16 only: 72 is
// not a multiple of the 32-element quant block, so quantized K/V cannot occur.
// DV4=18 is not a power of two, so the auto subw would break the shuffle reduction;
// pin subw=16 and let the ch < DV4 guards cover the top two chunks.
FA_AMD_INST_S   (72, 32, 16, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_INST_W64_S(72, 16, 16, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
// UNet diffusion heads (SD 1.5 runs 40 at full resolution, then 80 and 160). Same
// reasoning as dk72: DV4 is not a power of two, so subw is pinned and the guards
// cover the tail chunk. f16 only, since none of them is a whole quant block.
FA_AMD_INST_S   ( 40, 32,  8, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_INST_W64_S( 40, 16,  8, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_INST_S   ( 80, 32, 16, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_INST_W64_S( 80, 16, 16, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_INST_S   (160, 16, 32, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_INST_W64_S(160,  8, 32, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_PAIR_W64("kq4_0_vq4_0", block_q4_0, 8, dequantize_q4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PAIR_W64("kq8_0_vf16",  block_q8_0, 8, dequantize_q8_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_PAIR_W64("kq4_0_vf16",  block_q4_0, 8, dequantize_q4_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_PAIR_W64("kf16_vq8_0",  half4,      1, dequantize_f16_t4,  block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_PAIR_W64("kf16_vq4_0",  half4,      1, dequantize_f16_t4,  block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PAIR_W64("kq8_0_vq4_0", block_q8_0, 8, dequantize_q8_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PAIR_W64("kq4_0_vq8_0", block_q4_0, 8, dequantize_q4_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
#undef FA_AMD_PAIR_W64
// TurboQuant KV: DK is a multiple of 128 (the turbo block is 128 elements, each
// WHT-rotated independently). Dequant stays in the WHT-rotated domain; the graph
// un-rotates. Heads below 128 are zero-padded to 128 in the KV cache.
#define FA_AMD_TURBO(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S (128, 16, 8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(128, 16, 8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S (256, 16, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(256, 16, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S (384, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S (512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S (640, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_TURBO("kq8_0_vturbo2", block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO("kq8_0_vturbo3", block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO("kq8_0_vturbo4", block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO("kf16_vturbo2", half4, 1, dequantize_f16_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO("kf16_vturbo3", half4, 1, dequantize_f16_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO("kf16_vturbo4", half4, 1, dequantize_f16_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO("kturbo2_vq8_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TURBO("kturbo2_vf16", block_turbo2_0, 32, dequantize_turbo2_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TURBO("kturbo2_vturbo2", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO("kturbo2_vturbo3", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO("kturbo2_vturbo4", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO("kturbo3_vq8_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TURBO("kturbo3_vf16", block_turbo3_0, 32, dequantize_turbo3_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TURBO("kturbo3_vturbo2", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO("kturbo3_vturbo3", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO("kturbo3_vturbo4", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO("kturbo4_vq8_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TURBO("kturbo4_vf16", block_turbo4_0, 32, dequantize_turbo4_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TURBO("kturbo4_vturbo2", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO("kturbo4_vturbo3", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO("kturbo4_vturbo4", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO("kq4_0_vturbo2", block_q4_0, 8, dequantize_q4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO("kq4_0_vturbo3", block_q4_0, 8, dequantize_q4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO("kq4_0_vturbo4", block_q4_0, 8, dequantize_q4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO("kturbo2_vq4_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TURBO("kturbo3_vq4_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TURBO("kturbo4_vq4_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
#undef FA_AMD_TURBO
// wave64 (GCN/Vega) parity; nsg=8. Untested on real hardware.
#define FA_AMD_TURBO_W64(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(128, 8,  8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_WG(256, 8, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(256, 8, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(384, 8, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_WG(512, 8, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(512, 8, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(640, 8, 32, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_TURBO_W64("kq8_0_vturbo2", block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO_W64("kq8_0_vturbo3", block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO_W64("kq8_0_vturbo4", block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO_W64("kf16_vturbo2", half4, 1, dequantize_f16_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO_W64("kf16_vturbo3", half4, 1, dequantize_f16_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO_W64("kf16_vturbo4", half4, 1, dequantize_f16_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO_W64("kturbo2_vq8_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TURBO_W64("kturbo2_vf16", block_turbo2_0, 32, dequantize_turbo2_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TURBO_W64("kturbo2_vturbo2", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO_W64("kturbo2_vturbo3", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO_W64("kturbo2_vturbo4", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO_W64("kturbo3_vq8_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TURBO_W64("kturbo3_vf16", block_turbo3_0, 32, dequantize_turbo3_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TURBO_W64("kturbo3_vturbo2", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO_W64("kturbo3_vturbo3", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO_W64("kturbo3_vturbo4", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO_W64("kturbo4_vq8_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TURBO_W64("kturbo4_vf16", block_turbo4_0, 32, dequantize_turbo4_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TURBO_W64("kturbo4_vturbo2", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO_W64("kturbo4_vturbo3", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO_W64("kturbo4_vturbo4", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO_W64("kq4_0_vturbo2", block_q4_0, 8, dequantize_q4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO_W64("kq4_0_vturbo3", block_q4_0, 8, dequantize_q4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO_W64("kq4_0_vturbo4", block_q4_0, 8, dequantize_q4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO_W64("kturbo2_vq4_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TURBO_W64("kturbo3_vq4_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TURBO_W64("kturbo4_vq4_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
#undef FA_AMD_TURBO_W64

#undef FA_AMD_INST_W64
#undef FA_AMD_INST_W64_S
#undef FA_AMD_INST_W64_WG
#undef FA_AMD_PAIR_S
#undef FA_AMD_PAIR_Q4
#undef FA_AMD_INST_S

// Prefill companion of the AMD vec kernel above: NQ simdgroups (one query each)
// share K/V tiles staged in threadgroup memory, so the cache is streamed from
// device memory once per NQ queries instead of once per query.
template<
    short DK, short NQ,
    typename k_t, short NKK, void (*deq_k)(device const k_t *, short, thread float4 &),
    typename v_t, short NKV, void (*deq_v)(device const v_t *, short, thread float4 &),
    short NW = 32, short DVP = 0>
kernel void kernel_flash_attn_ext_tile_amd(
        constant ggml_metal_kargs_flash_attn_ext_amd & args [[buffer(0)]],
        device const char * q     [[buffer(1)]],
        device const char * k     [[buffer(2)]],
        device const char * v     [[buffer(3)]],
        device const char * mask  [[buffer(4)]],
        device const char * sinks [[buffer(5)]],
        device const char * pad   [[buffer(6)]],
        device       char * dst   [[buffer(7)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short DV  = DVP ? DVP : DK;
    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short NCK = (DK4 + NW - 1)/NW;
    constexpr short NCV = (DV4 + NW - 1)/NW;
    constexpr short TK  = 8192/(DK + DV); // KV rows per tile (sk + sv <= 16 KiB)

    threadgroup half4 sk[TK*DK4];
    threadgroup half4 sv[TK*DV4];
    threadgroup short sh_any;

    const int iq1 = tgpig.x*NQ + sgitg; // this simdgroup's query token
    const int iq2 = tgpig.y;            // query head
    const int iq3 = tgpig.z;            // batch

    // tail simdgroups past ne01 recompute the last query and skip the store;
    // they must stay for the cooperative loads and barriers
    const int iq1c = iq1 < args.ne01 ? iq1 : args.ne01 - 1;

    device const float4 * q4 = (device const float4 *) (q + iq1c*args.nb01 + iq2*args.nb02 + iq3*args.nb03);

    // grouped-query attention: map query head -> kv head
    const int ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const int ikv3 = iq3 / (args.ne03 / args.ne_12_3);

    device const char * kp = k + ikv2*args.nb12 + ikv3*args.nb13;
    device const char * vp = v + ikv2*args.nb22 + ikv3*args.nb23;

    // Without a mask ne32/ne33 are 0, so the index math would divide by zero.
    device const half * pm = (device const half *) mask;
    if (args.has_mask) {
        pm = (device const half *) (mask + iq1c*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    float4 myq[NCK];
    for (short c = 0; c < NCK; ++c) {
        const short ch = tiisg + c*NW;
        myq[c] = ch < DK4 ? q4[ch] : float4(0.0f);
    }

    float4 acc[NCV];
    for (short c = 0; c < NCV; ++c) {
        acc[c] = float4(0.0f);
    }
    float m = -FLT_MAX/2;
    float s = 0.0f;

    const short tl = NW*sgitg + tiisg; // linear thread id for cooperative loads

    for (int j0 = 0; j0 < args.ne11; j0 += TK) {
        const short tn = (args.ne11 - j0 < TK) ? (short)(args.ne11 - j0) : TK;

        // skip tiles no query of this group attends to (causal top-right triangle)
        if (tl == 0) {
            sh_any = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            float mx = -FLT_MAX;
            for (short jj = tiisg; jj < tn; jj += NW) {
                mx = max(mx, args.has_mask ? (float) pm[j0 + jj] : 0.0f);
            }
            mx = simd_max(mx);
            if (tiisg == 0 && mx > -1e30f) {
                sh_any = 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sh_any == 0) {
            continue;
        }

        // cooperative dequant of the K/V tile rows into threadgroup memory;
        // the loops are separate because MLA has DK != DV
        for (short c = tl; c < tn*DK4; c += NW*NQ) {
            const short jj = c / DK4;
            const short ch = c % DK4;
            float4 t;
            device const k_t * kb = (device const k_t *) (kp + (uint64_t)(j0 + jj)*args.nb11);
            deq_k(kb + ch/NKK, ch%NKK, t);
            sk[c] = (half4) t;
        }
        for (short c = tl; c < tn*DV4; c += NW*NQ) {
            const short jj = c / DV4;
            const short ch = c % DV4;
            float4 t;
            device const v_t * vb = (device const v_t *) (vp + (uint64_t)(j0 + jj)*args.nb21);
            deq_v(vb + ch/NKV, ch%NKV, t);
            sv[c] = (half4) t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short jj = 0; jj < tn; ++jj) {
            const float mv = args.has_mask ? (float) pm[j0 + jj] : 0.0f;
            if (mv <= -1e30f) {
                continue;
            }

            float pl = 0.0f;
            for (short c = 0; c < NCK; ++c) {
                const short ch = tiisg + c*NW;
                if (ch < DK4) {
                    pl += dot(myq[c], (float4) sk[jj*DK4 + ch]);
                }
            }
            const float p = simd_sum(pl); // full DK-dim Q.K

            const float pp   = p*args.scale + mv;
            const float nm   = max(m, pp);
            const float corr = exp(m  - nm);
            const float w    = exp(pp - nm);

            for (short c = 0; c < NCV; ++c) {
                const short ch = tiisg + c*NW;
                if (ch < DV4) {
                    acc[c] = acc[c]*corr + w*(float4) sv[jj*DV4 + ch];
                }
            }
            s = s*corr + w;
            m = nm;
        }
        // the two barriers at the top of the next tile fence sk/sv reuse
    }

    // attention sink: one extra virtual token per head, weight only
    if (args.has_sinks) {
        const float sk_ = ((device const float *) sinks)[iq2];
        const float nm = max(m, sk_);
        const float corr = exp(m - nm);
        for (short c = 0; c < NCV; ++c) {
            acc[c] *= corr;
        }
        s = s*corr + exp(sk_ - nm);
    }

    if (iq1 >= args.ne01) {
        return;
    }

    const float inv = s > 0.0f ? 1.0f/s : 0.0f;
    const int64_t rid = (int64_t) iq3*args.ne2*args.ne1 + iq2 + (int64_t) iq1*args.ne1;
    device float4 * dst4 = (device float4 *) dst;
    for (short c = 0; c < NCV; ++c) {
        const short ch = tiisg + c*NW;
        if (ch < DV4) {
            dst4[rid*DV4 + ch] = acc[c]*inv;
        }
    }
}

typedef decltype(kernel_flash_attn_ext_tile_amd<128, 16, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4>) flash_attn_ext_tile_amd_t;

// standard K/V pairs only; turbo caches keep using the vec kernel for prefill
#define FA_AMD_TILE_INST(dk, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_tile_amd_dk" #dk "_" nm)]] kernel flash_attn_ext_tile_amd_t \
    kernel_flash_attn_ext_tile_amd<dk, 16, kt, nkk, dkf, vt, nkv, dvf>;
#define FA_AMD_TILE_PAIR(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST( 64, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(128, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(256, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(512, nm, kt, nkk, dkf, vt, nkv, dvf)

FA_AMD_TILE_PAIR("kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_TILE_PAIR("kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TILE_PAIR("kq4_0_vq4_0", block_q4_0, 8, dequantize_q4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_PAIR("kq8_0_vf16",  block_q8_0, 8, dequantize_q8_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_TILE_PAIR("kq4_0_vf16",  block_q4_0, 8, dequantize_q4_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_TILE_PAIR("kf16_vq8_0",  half4,      1, dequantize_f16_t4,  block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TILE_PAIR("kf16_vq4_0",  half4,      1, dequantize_f16_t4,  block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_PAIR("kq8_0_vq4_0", block_q8_0, 8, dequantize_q8_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_PAIR("kq4_0_vq8_0", block_q4_0, 8, dequantize_q4_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
// MLA (deepseek2) keeps the 64-wide rope tail in K only.
#define FA_AMD_TILE_INST_ASYM(dk, dv, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_tile_amd_dk" #dk "_dv" #dv "_" nm)]] kernel flash_attn_ext_tile_amd_t \
    kernel_flash_attn_ext_tile_amd<dk, 16, kt, nkk, dkf, vt, nkv, dvf, 32, dv>;
FA_AMD_TILE_INST_ASYM(576, 512, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_TILE_INST_ASYM(576, 512, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TILE_INST_ASYM(320, 256, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_TILE_INST_ASYM(320, 256, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
#undef FA_AMD_TILE_INST_ASYM
// TurboQuant KV (prefill companion of the FA_AMD_TURBO vec kernels).
#define FA_AMD_TILE_TURBO(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(128, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(256, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(384, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(512, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(640, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_TILE_TURBO("kq8_0_vturbo2", block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO("kq8_0_vturbo3", block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO("kq8_0_vturbo4", block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO("kf16_vturbo2", half4, 1, dequantize_f16_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO("kf16_vturbo3", half4, 1, dequantize_f16_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO("kf16_vturbo4", half4, 1, dequantize_f16_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO("kturbo2_vq8_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TILE_TURBO("kturbo2_vf16", block_turbo2_0, 32, dequantize_turbo2_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_TURBO("kturbo2_vturbo2", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO("kturbo2_vturbo3", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO("kturbo2_vturbo4", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO("kturbo3_vq8_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TILE_TURBO("kturbo3_vf16", block_turbo3_0, 32, dequantize_turbo3_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_TURBO("kturbo3_vturbo2", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO("kturbo3_vturbo3", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO("kturbo3_vturbo4", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO("kturbo4_vq8_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TILE_TURBO("kturbo4_vf16", block_turbo4_0, 32, dequantize_turbo4_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_TURBO("kturbo4_vturbo2", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO("kturbo4_vturbo3", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO("kturbo4_vturbo4", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO("kq4_0_vturbo2", block_q4_0, 8, dequantize_q4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO("kq4_0_vturbo3", block_q4_0, 8, dequantize_q4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO("kq4_0_vturbo4", block_q4_0, 8, dequantize_q4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO("kturbo2_vq4_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_TURBO("kturbo3_vq4_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_TURBO("kturbo4_vq4_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
#undef FA_AMD_TILE_TURBO
// wave64 variants: 64-lane simdgroups, 8 queries per threadgroup (512 threads).
#define FA_AMD_TILE_INST_W64(dk, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_tile_amd_dk" #dk "_" nm "_w64")]] kernel flash_attn_ext_tile_amd_t \
    kernel_flash_attn_ext_tile_amd<dk, 8, kt, nkk, dkf, vt, nkv, dvf, 64>;
#define FA_AMD_TILE_PAIR_W64(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64( 64, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(128, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(256, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(512, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_TILE_PAIR_W64("kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_TILE_PAIR_W64("kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TILE_PAIR_W64("kq4_0_vq4_0", block_q4_0, 8, dequantize_q4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_PAIR_W64("kq8_0_vf16",  block_q8_0, 8, dequantize_q8_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_TILE_PAIR_W64("kq4_0_vf16",  block_q4_0, 8, dequantize_q4_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_TILE_PAIR_W64("kf16_vq8_0",  half4,      1, dequantize_f16_t4,  block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TILE_PAIR_W64("kf16_vq4_0",  half4,      1, dequantize_f16_t4,  block_q4_0, 8, dequantize_q4_0_t4)

// DK=72 vision towers; f16 only, see the vec kernel. TK = 4096/72 = 56 keeps
// sk+sv at 15.75 KiB.
FA_AMD_TILE_INST    (72, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_INST_W64(72, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_PAIR_W64("kq8_0_vq4_0", block_q8_0, 8, dequantize_q8_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_PAIR_W64("kq4_0_vq8_0", block_q4_0, 8, dequantize_q4_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
// MLA wave64 parity (Vega/GCN): eight 64-lane query groups.
#define FA_AMD_TILE_INST_ASYM_W64(dk, dv, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_tile_amd_dk" #dk "_dv" #dv "_" nm "_w64")]] kernel flash_attn_ext_tile_amd_t \
    kernel_flash_attn_ext_tile_amd<dk, 8, kt, nkk, dkf, vt, nkv, dvf, 64, dv>;
FA_AMD_TILE_INST_ASYM_W64(576, 512, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_TILE_INST_ASYM_W64(576, 512, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TILE_INST_ASYM_W64(320, 256, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_TILE_INST_ASYM_W64(320, 256, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
#undef FA_AMD_TILE_INST_ASYM_W64
// TurboQuant KV wave64 parity. Untested on real hardware.
#define FA_AMD_TILE_TURBO_W64(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(128, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(256, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(384, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(512, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(640, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_TILE_TURBO_W64("kq8_0_vturbo2", block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO_W64("kq8_0_vturbo3", block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO_W64("kq8_0_vturbo4", block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO_W64("kf16_vturbo2", half4, 1, dequantize_f16_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO_W64("kf16_vturbo3", half4, 1, dequantize_f16_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO_W64("kf16_vturbo4", half4, 1, dequantize_f16_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo2_vq8_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo2_vf16", block_turbo2_0, 32, dequantize_turbo2_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_TURBO_W64("kturbo2_vturbo2", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo2_vturbo3", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo2_vturbo4", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo3_vq8_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo3_vf16", block_turbo3_0, 32, dequantize_turbo3_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_TURBO_W64("kturbo3_vturbo2", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo3_vturbo3", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo3_vturbo4", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo4_vq8_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo4_vf16", block_turbo4_0, 32, dequantize_turbo4_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_TURBO_W64("kturbo4_vturbo2", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo4_vturbo3", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo4_vturbo4", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO_W64("kq4_0_vturbo2", block_q4_0, 8, dequantize_q4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO_W64("kq4_0_vturbo3", block_q4_0, 8, dequantize_q4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO_W64("kq4_0_vturbo4", block_q4_0, 8, dequantize_q4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo2_vq4_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo3_vq4_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo4_vq4_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
#undef FA_AMD_TILE_TURBO_W64
#undef FA_AMD_TILE_PAIR_W64
#undef FA_AMD_TILE_INST_W64
#undef FA_AMD_TILE_PAIR
#undef FA_AMD_TILE_INST

// Prefill attention for AMD: BQ queries stay resident while the cache streams in
// blocks of BK, so the score matrix never reaches device memory.
//
// Lanes split into groups of 16 so every softmax reduction stays inside one
// simdgroup on wave32 and wave64 alike. Row strides are padded off the 32-bank
// period to keep the 16 concurrent rows on 16 different banks.
template<
    short DK, short DV, short BQ, short BK, short NT,
    typename k_t, short NKK, void (*deq_k)(device const k_t *, short, thread float4 &),
    typename v_t, short NKV, void (*deq_v)(device const v_t *, short, thread float4 &)>
kernel void kernel_flash_attn_ext_pf_amd(
        constant ggml_metal_kargs_flash_attn_ext_amd & args [[buffer(0)]],
        device const char * q     [[buffer(1)]],
        device const char * k     [[buffer(2)]],
        device const char * v     [[buffer(3)]],
        device const char * mask  [[buffer(4)]],
        device const char * sinks [[buffer(5)]],
        device const char * pad   [[buffer(6)]],
        device       char * dst   [[buffer(7)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short NQG = NT/16;
    constexpr short NQR = BQ/NQG;
    constexpr short NKC = BK/16;
    constexpr short NDC = (DV + 15)/16;  // DK 72 (vision) leaves the top lanes idle
    // half2 run before flushing to float; it has to divide DK, and 576 (MLA) is not a multiple of 128
    constexpr short DKF = DK % 128 == 0 ? 128 : (DK % 64 == 0 ? 64 : DK);
    constexpr short SQS = DK + 4;
    constexpr short SVS = BK + 4;
    constexpr short SKV = BK*SQS > DV*SVS ? BK*SQS : DV*SVS;

    threadgroup half sq [BQ*SQS];
    threadgroup half skv[SKV];     // K rows first, then V transposed over the same memory
    threadgroup half sp [BQ*SVS];  // softmax weights

    const short kg = tiitg % 16;
    const short qg = tiitg / 16;

    const int iq1 = tgpig.x*BQ;
    const int iq2 = tgpig.y;
    const int iq3 = tgpig.z;

    const int ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const int ikv3 = iq3 / (args.ne03 / args.ne_12_3);

    device const char * kp = k + ikv2*args.nb12 + ikv3*args.nb13;
    device const char * vp = v + ikv2*args.nb22 + ikv3*args.nb23;

    // scale folded into Q so the score pass is a plain dot product
    for (short c = tiitg; c < BQ*DK4; c += NT) {
        const short jq = c / DK4;
        const short ch = c % DK4;
        float4 qv = 0.0f;
        if (iq1 + jq < args.ne01) {
            qv = ((device const float4 *) (q + (iq1 + jq)*args.nb01 + iq2*args.nb02 + iq3*args.nb03))[ch]*args.scale;
        }
        *(threadgroup half4 *) (sq + jq*SQS + 4*ch) = (half4) qv;
    }

    // the mask values a thread needs are exactly the ones it reduces, so they stay
    // in registers; staging them through LDS measured as costly as the Q*K pass
    device const half * pmr[NQR];
    if (args.has_mask) {
        for (short rr = 0; rr < NQR; ++rr) {
            const int jq = iq1 + qg + NQG*rr;
            const int gq = jq < args.ne01 ? jq : args.ne01 - 1;
            pmr[rr] = (device const half *)
                (mask + gq*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
        }
    }

    float m   [NQR];
    float srun[NQR];
    float acc [NQR][NDC];
    for (short rr = 0; rr < NQR; ++rr) {
        m[rr]    = -FLT_MAX/2;
        srun[rr] = 0.0f;
        for (short dd = 0; dd < NDC; ++dd) {
            acc[rr][dd] = 0.0f;
        }
    }

    for (int j0 = 0; j0 < args.ne11; j0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // mask and K read together so their loads overlap
        half  mv[NQR][NKC];
        float any = 1.0f;
        if (args.has_mask) {
            any = 0.0f;
            for (short rr = 0; rr < NQR; ++rr) {
                for (short cc = 0; cc < NKC; ++cc) {
                    const short col = kg + 16*cc;
                    const half x = j0 + col < args.ne11 ? pmr[rr][j0 + col] : (half) -MAXHALF;
                    mv[rr][cc] = x;
                    any = max(any, (float) x > -1e30f ? 1.0f : 0.0f);
                }
            }
        }

        for (short c = tiitg; c < BK*DK4; c += NT) {
            const short jj = c / DK4;
            const short ch = c % DK4;
            float4 tv = 0.0f;
            if (j0 + jj < args.ne11) {
                device const k_t * kb = (device const k_t *) (kp + (uint64_t)(j0 + jj)*args.nb11);
                deq_k(kb + ch/NKK, ch%NKK, tv);
            }
            *(threadgroup half4 *) (skv + jj*SQS + 4*ch) = (half4) tv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // skip is per query group; the loads and barriers below stay uniform
        for (ushort o = 1; o < 16; o <<= 1) {
            any = max(any, simd_shuffle_xor(any, o));
        }
        if (any > 0.0f) {

        float sc[NQR][NKC];
        for (short rr = 0; rr < NQR; ++rr) {
            for (short cc = 0; cc < NKC; ++cc) {
                sc[rr][cc] = 0.0f;
            }
        }
        for (short d0 = 0; d0 < DK; d0 += DKF) {
            half2 pacc[NQR][NKC];
            for (short rr = 0; rr < NQR; ++rr) {
                for (short cc = 0; cc < NKC; ++cc) {
                    pacc[rr][cc] = 0.0h;
                }
            }
            FOR_UNROLL (short d = 0; d < DKF; d += 2) {
                half2 av[NQR];
                half2 bv[NKC];
                for (short rr = 0; rr < NQR; ++rr) {
                    av[rr] = *(threadgroup const half2 *) (sq + (qg + NQG*rr)*SQS + d0 + d);
                }
                for (short cc = 0; cc < NKC; ++cc) {
                    bv[cc] = *(threadgroup const half2 *) (skv + (kg + 16*cc)*SQS + d0 + d);
                }
                for (short rr = 0; rr < NQR; ++rr) {
                    for (short cc = 0; cc < NKC; ++cc) {
                        pacc[rr][cc] = av[rr]*bv[cc] + pacc[rr][cc];
                    }
                }
            }
            for (short rr = 0; rr < NQR; ++rr) {
                for (short cc = 0; cc < NKC; ++cc) {
                    sc[rr][cc] += (float) pacc[rr][cc][0] + (float) pacc[rr][cc][1];
                }
            }
        }

        for (short rr = 0; rr < NQR; ++rr) {
            const short row = qg + NQG*rr;
            float mx = -FLT_MAX;
            for (short cc = 0; cc < NKC; ++cc) {
                if (args.has_mask) {
                    sc[rr][cc] += (float) mv[rr][cc];
                } else if (j0 + kg + 16*cc >= args.ne11) {
                    sc[rr][cc] = -INFINITY;
                }
                mx = max(mx, sc[rr][cc]);
            }
            for (ushort o = 1; o < 16; o <<= 1) {
                mx = max(mx, simd_shuffle_xor(mx, o));
            }

            const float mn   = max(m[rr], mx);
            const float corr = exp(m[rr] - mn);

            float ls = 0.0f;
            for (short cc = 0; cc < NKC; ++cc) {
                const float w = exp(sc[rr][cc] - mn);
                sp[row*SVS + kg + 16*cc] = (half) w;
                ls += w;
            }
            for (ushort o = 1; o < 16; o <<= 1) {
                ls += simd_shuffle_xor(ls, o);
            }

            m[rr]    = mn;
            srun[rr] = srun[rr]*corr + ls;
            for (short dd = 0; dd < NDC; ++dd) {
                acc[rr][dd] *= corr;
            }
        }

        }

        // V lands transposed so the P*V pass reads two cache rows in one half2
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short c = tiitg; c < BK*DV4; c += NT) {
            const short jj = c / DV4;
            const short ch = c % DV4;
            float4 tv = 0.0f;
            if (j0 + jj < args.ne11) {
                device const v_t * vb = (device const v_t *) (vp + (uint64_t)(j0 + jj)*args.nb21);
                deq_v(vb + ch/NKV, ch%NKV, tv);
            }
            for (short i = 0; i < 4; ++i) {
                skv[(4*ch + i)*SVS + jj] = (half) tv[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (any == 0.0f) {
            continue;
        }

        for (short j = 0; j < BK; j += BK) {
            half2 pacc[NQR][NDC];
            for (short rr = 0; rr < NQR; ++rr) {
                for (short dd = 0; dd < NDC; ++dd) {
                    pacc[rr][dd] = 0.0h;
                }
            }
            FOR_UNROLL (short jj = 0; jj < BK; jj += 2) {
                half2 pv[NQR];
                half2 vv[NDC];
                for (short rr = 0; rr < NQR; ++rr) {
                    pv[rr] = *(threadgroup const half2 *) (sp + (qg + NQG*rr)*SVS + j + jj);
                }
                for (short dd = 0; dd < NDC; ++dd) {
                    vv[dd] = DV % 16 == 0 || kg + 16*dd < DV
                        ? *(threadgroup const half2 *) (skv + (kg + 16*dd)*SVS + j + jj) : 0.0h;
                }
                for (short rr = 0; rr < NQR; ++rr) {
                    for (short dd = 0; dd < NDC; ++dd) {
                        pacc[rr][dd] = pv[rr]*vv[dd] + pacc[rr][dd];
                    }
                }
            }
            for (short rr = 0; rr < NQR; ++rr) {
                for (short dd = 0; dd < NDC; ++dd) {
                    acc[rr][dd] += (float) pacc[rr][dd][0] + (float) pacc[rr][dd][1];
                }
            }
        }
    }

    if (args.has_sinks) {
        const float sv = ((device const float *) sinks)[iq2];
        for (short rr = 0; rr < NQR; ++rr) {
            const float mn   = max(m[rr], sv);
            const float corr = exp(m[rr] - mn);
            for (short dd = 0; dd < NDC; ++dd) {
                acc[rr][dd] *= corr;
            }
            srun[rr] = srun[rr]*corr + exp(sv - mn);
            m[rr]    = mn;
        }
    }

    device float * dst1 = (device float *) dst;
    for (short rr = 0; rr < NQR; ++rr) {
        const int iq = iq1 + qg + NQG*rr;
        if (iq >= args.ne01) {
            continue;
        }
        const float inv = srun[rr] > 0.0f ? 1.0f/srun[rr] : 0.0f;
        const int64_t rid = (int64_t) iq3*args.ne2*args.ne1 + iq2 + (int64_t) iq*args.ne1;
        for (short dd = 0; dd < NDC; ++dd) {
            if (DV % 16 == 0 || kg + 16*dd < DV) {
                dst1[rid*DV + kg + 16*dd] = acc[rr][dd]*inv;
            }
        }
    }
}

typedef decltype(kernel_flash_attn_ext_pf_amd<128, 128, 64, 64, 256, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4>) flash_attn_ext_pf_amd_t;

// BQ falls as the head grows so the O accumulators stay near 32 registers per
// thread. Head 512 would leave one query row per thread, and stays on the tile kernel.
#define FA_AMD_PF_INST(dk, bq, bk, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_pf_amd_dk" #dk "_" nm)]] kernel flash_attn_ext_pf_amd_t \
    kernel_flash_attn_ext_pf_amd<dk, dk, bq, bk, 256, kt, nkk, dkf, vt, nkv, dvf>;
// asymmetric head: MLA keeps K wider than V
#define FA_AMD_PF_INST_DV(dk, dv, bq, bk, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_pf_amd_dk" #dk "_dv" #dv "_" nm)]] kernel flash_attn_ext_pf_amd_t \
    kernel_flash_attn_ext_pf_amd<dk, dv, bq, bk, 256, kt, nkk, dkf, vt, nkv, dvf>;
#define FA_AMD_PF_PAIR(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_PF_INST( 64, 64, 64, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_PF_INST(128, 64, 64, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_PF_INST(256, 32, 64, nm, kt, nkk, dkf, vt, nkv, dvf)

FA_AMD_PF_PAIR("kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_PF_PAIR("kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_PF_PAIR("kq4_0_vq4_0", block_q4_0, 8, dequantize_q4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PF_PAIR("kq8_0_vf16",  block_q8_0, 8, dequantize_q8_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_PF_PAIR("kq4_0_vf16",  block_q4_0, 8, dequantize_q4_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_PF_PAIR("kf16_vq8_0",  half4,      1, dequantize_f16_t4,  block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_PF_PAIR("kf16_vq4_0",  half4,      1, dequantize_f16_t4,  block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PF_PAIR("kq8_0_vq4_0", block_q8_0, 8, dequantize_q8_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PF_PAIR("kq4_0_vq8_0", block_q4_0, 8, dequantize_q4_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
// TurboQuant KV (large-batch prefill companion of the tile/vec turbo kernels).
// dk384/dk512/dk640 have no PF variant; the host routes them to the tile kernel.
#define FA_AMD_PF_TURBO(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_PF_INST(128, 64, 64, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_PF_INST(256, 32, 64, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_PF_TURBO("kq8_0_vturbo2", block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_PF_TURBO("kq8_0_vturbo3", block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_PF_TURBO("kq8_0_vturbo4", block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_PF_TURBO("kf16_vturbo2", half4, 1, dequantize_f16_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_PF_TURBO("kf16_vturbo3", half4, 1, dequantize_f16_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_PF_TURBO("kf16_vturbo4", half4, 1, dequantize_f16_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_PF_TURBO("kturbo2_vq8_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_PF_TURBO("kturbo2_vf16", block_turbo2_0, 32, dequantize_turbo2_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_PF_TURBO("kturbo2_vturbo2", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_PF_TURBO("kturbo2_vturbo3", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_PF_TURBO("kturbo2_vturbo4", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_PF_TURBO("kturbo3_vq8_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_PF_TURBO("kturbo3_vf16", block_turbo3_0, 32, dequantize_turbo3_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_PF_TURBO("kturbo3_vturbo2", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_PF_TURBO("kturbo3_vturbo3", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_PF_TURBO("kturbo3_vturbo4", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_PF_TURBO("kturbo4_vq8_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
FA_AMD_PF_TURBO("kturbo4_vf16", block_turbo4_0, 32, dequantize_turbo4_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_PF_TURBO("kturbo4_vturbo2", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_PF_TURBO("kturbo4_vturbo3", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_PF_TURBO("kturbo4_vturbo4", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_PF_TURBO("kq4_0_vturbo2", block_q4_0, 8, dequantize_q4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_PF_TURBO("kq4_0_vturbo3", block_q4_0, 8, dequantize_q4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_PF_TURBO("kq4_0_vturbo4", block_q4_0, 8, dequantize_q4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_PF_TURBO("kturbo2_vq4_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PF_TURBO("kturbo3_vq4_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PF_TURBO("kturbo4_vq4_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
#undef FA_AMD_PF_TURBO

// vision towers, f16 only like the vec and tile kernels
FA_AMD_PF_INST(72, 64, 64, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
// UNet diffusion heads, f16 only like head 72
FA_AMD_PF_INST( 40, 64, 64, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_PF_INST( 80, 64, 64, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_PF_INST(160, 32, 64, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
// head 512 has no PF variant and falls to the tile kernel, ~21% worse at depth; BK=32 keeps
// the shared memory at 54 KB, the same league as dk256
#define FA_AMD_PF_512(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_PF_INST(512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_PF_512("kf16_vf16",  half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_PF_512("kq8_0_vf16", block_q8_0, 8, dequantize_q8_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_PF_512("kq4_0_vf16", block_q4_0, 8, dequantize_q4_0_t4, half4,      1, dequantize_f16_t4)
// MLA: K is 576 (512 latent + 64 rope) and V is 512; the tile kernel is ~21% slower at depth
FA_AMD_PF_INST_DV(576, 512, 16, 32, "kf16_vf16",  half4,      1, dequantize_f16_t4,  half4, 1, dequantize_f16_t4)
FA_AMD_PF_INST_DV(576, 512, 16, 32, "kq8_0_vf16", block_q8_0, 8, dequantize_q8_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_PF_INST_DV(320, 256, 16, 32, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_PF_INST_DV(320, 256, 16, 32, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4)
#undef FA_AMD_PF_512
#undef FA_AMD_PF_PAIR
#undef FA_AMD_PF_INST

// note: I think the s_t can be half instead of float, because the Q*K scaling is done before storing to shared mem
//       in the other (non-vec) kernel, we need s_t to also be float because we scale during the soft_max
//
