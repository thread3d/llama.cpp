#include "ggml-metal-ops.h"

#ifdef TOSH_ENABLE_DYNAMIC_MOE
#include "tosh-moe.h"
#else
static inline const void * tosh_moe_state_of(const void *) { return nullptr; }
static inline const void * tosh_moe_ram_of(const void *)   { return nullptr; }
static inline const void * tosh_moe_slots_of(const void *) { return nullptr; }
static inline const void * tosh_moe_stage(void)            { return nullptr; }
static inline int tosh_moe_experts_of(const void *)         { return 0; }
static inline int tosh_moe_fixed_of(const void *)           { return 0; }
static inline bool tosh_moe_should_route(const void *, int) { return false; }
static inline int tosh_moe_ids_room(int)                   { return 0; }
static inline int tosh_moe_off_slot_for_id(int, int, int)  { return 0; }
static inline int tosh_moe_off_cold_for_id(int, int, int)  { return 0; }
static inline int tosh_moe_off_id_of_slot(int, int, int)   { return 0; }
static inline int tosh_moe_off_usage(int, int, int)        { return 0; }
static inline int tosh_moe_off_step(int, int, int)         { return 0; }
static inline int tosh_moe_off_n_indices(int, int, int)    { return 0; }
static inline int tosh_moe_off_evict(int, int, int)        { return 0; }
static inline int tosh_moe_off_src(int, int, int)          { return 0; }
static inline int tosh_moe_off_ids(int, int, int)          { return 0; }
static inline int tosh_moe_off_active(int, int, int)       { return 0; }
#endif

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-impl.h"
#include "ggml-metal-common.h"
#include "ggml-metal-device.h"
#include "ggml-metal-fusion.h"
#include "ggml-metal-tuning.h"

#include <mutex>
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <cmath>
#include <unordered_map>

// Encoding runs on one thread per card, and the staged MoE path writes these
// while other cards read them.
static std::mutex g_tosh_moe_mutex;
static std::unordered_map<const ggml_tensor *, ggml_metal_buffer_id> g_tosh_moe_staged_ids;

static std::unordered_map<const ggml_tensor *, ggml_metal_buffer_id> g_tosh_moe_full_cold;
static ggml_metal_buffer_id ggml_metal_get_buffer_id(const ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) buffer->context;

    return ggml_metal_buffer_get_id(ctx, t);
}

static ggml_metal_buffer_id ggml_metal_get_host_buffer_id(
        ggml_metal_device_t dev, const ggml_tensor * t) {
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    // The canonical CPU buffer type intentionally has no device pointer in ggml. Detect the
    // property we actually need instead: whether the storage is host-addressable.
    if (!ggml_backend_buffer_is_host(buffer)) {
        return ggml_metal_get_buffer_id(t);
    }

    // Wrap the whole CPU allocation once. The expert tensors are slices of this allocation;
    // mapping them independently duplicates address ranges and eventually exhausts Metal's
    // resource table on large models.
    void * const base = ggml_backend_buffer_get_base(buffer);
    const size_t size = ggml_backend_buffer_get_size(buffer);
    const size_t off  = (const char *) t->data - (const char *) base;

    size_t offs = 0;
    void * metal = ggml_metal_device_wrap_host(dev, base, size, &offs);
    if (metal == nullptr) {
        // A driver can reject a single allocation larger than maxBufferLength. Keep a bounded
        // compatibility fallback for smaller tensors; never mistake this for a VRAM budget.
        metal = ggml_metal_device_wrap_host(dev, t->data, ggml_nbytes(t), &offs);
        GGML_ASSERT(metal != nullptr && "failed to wrap the Dynamic MoE CPU bank for Metal");
        return { metal, offs };
    }

    return { metal, offs + off };
}

int ggml_metal_op_cached_moe_layer(const ggml_tensor * op) {
#ifdef TOSH_ENABLE_DYNAMIC_MOE
    if (op && op->op == GGML_OP_MUL_MAT_ID && op->src[0] && tosh_moe_state_of(op->src[0])) {
        const int registered = tosh_moe_layer_of(op->src[0]);
        if (registered >= 0) {
            return registered;
        }

        // Graph construction can hand Metal an alias of the registered slots tensor. The
        // state association above proves this is a cached bank; recover the layer from the
        // stable GGUF tensor name so window partitioning does not silently stay disabled.
        int il = -1;
        if (sscanf(op->src[0]->name, "blk.%d.", &il) == 1) {
            return il;
        }
    }
#else
    (void) op;
#endif
    return -1;
}

const ggml_tensor * ggml_metal_op_cached_moe_host(const ggml_tensor * op) {
#ifdef TOSH_ENABLE_DYNAMIC_MOE
    if (ggml_metal_op_cached_moe_layer(op) >= 0) {
        return (const ggml_tensor *) tosh_moe_ram_of(op->src[0]);
    }
#else
    (void) op;
#endif
    return nullptr;
}

void ggml_metal_op_cached_moe_ids_set(const ggml_tensor * slots, ggml_metal_buffer_id ids) {
    std::lock_guard<std::mutex> lock(g_tosh_moe_mutex);
    g_tosh_moe_staged_ids[slots] = ids;
}

void ggml_metal_op_cached_moe_ids_clear(void) {
    std::lock_guard<std::mutex> lock(g_tosh_moe_mutex);
    g_tosh_moe_staged_ids.clear();
}

void ggml_metal_op_cached_moe_full_set(const ggml_tensor * slots, ggml_metal_buffer_id cold) {
    std::lock_guard<std::mutex> lock(g_tosh_moe_mutex);
    g_tosh_moe_full_cold[slots] = cold;
}

void ggml_metal_op_cached_moe_full_clear(void) {
    std::lock_guard<std::mutex> lock(g_tosh_moe_mutex);
    g_tosh_moe_full_cold.clear();
}

struct ggml_metal_op {
    ggml_metal_op(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        ggml_metal_fusion_info * finfo,
        int  idx_start,
        int  idx_end,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph) {
        this->dev             = dev;
        this->lib             = ggml_metal_device_get_library(dev);
        this->enc             = ggml_metal_encoder_init(cmd_buf, use_concurrency);
        this->mem_ranges      = ggml_mem_ranges_init(debug_graph);
        this->finfo           = finfo;
        this->idx_start       = idx_start;
        this->idx_end         = idx_end;
        this->use_concurrency = use_concurrency;
        this->use_capture     = use_capture;
        this->debug_graph     = debug_graph;
        this->gf              = gf;

        idxs.reserve(gf->n_nodes);

        // filter empty nodes
        // TODO: this can be removed when the allocator starts filtering them earlier
        //       https://github.com/ggml-org/llama.cpp/pull/16130#issuecomment-3327905830
        for (int i = idx_start; i < idx_end; i++) {
            if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
                idxs.push_back(i);
            }
        }
    }

    ~ggml_metal_op() {
        ggml_metal_encoder_end_encoding(this->enc);
        ggml_metal_encoder_free(this->enc);
        ggml_mem_ranges_free(this->mem_ranges);
    }

    int n_nodes() const {
        return idxs.size();
    }

    ggml_tensor * node(int i) const {
        assert(i >= 0 && i < (int) idxs.size());
        return ggml_graph_node(gf, idxs[i]);
    }

    const ggml_cgraph * graph() const {
        return gf;
    }

    bool has_n_uses(int i, int n) const {
        assert(i >= 0 && i < (int) idxs.size());
        return ggml_node_has_n_uses(gf, idxs[i], n);
    }

    bool can_fuse(int i0, const ggml_op * ops, int n_ops) const {
        assert(use_fusion());
        assert(i0 >= 0 && i0 < n_nodes());

        if (i0 + n_ops > n_nodes()) {
            return false;
        }

        return ggml_can_fuse_ext(gf, idxs.data() + i0, ops, n_ops);
    }

    // consult the fusion table for the longest pattern starting at i0
    // returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes
    const ggml_metal_fusion * can_fuse(int i0, enum ggml_metal_fusion_mode mode, int * n_out) const {
        assert(use_fusion());
        assert(i0 >= 0 && i0 < n_nodes());

        return ggml_metal_fusion_next(gf, idxs.data(), (int) idxs.size(), i0, mode, n_out);
    }

    // whether to attempt fusion; the toggle lives in the shared fusion debugging context owned
    // by the device (initialized from GGML_METAL_FUSION_DISABLE, overridable by the test)
    bool use_fusion() const {
        return ggml_metal_fusion_info_enabled(finfo);
    }

    // record that a fusion fired, indexed by the matching table entry
    void count_fusions(const ggml_metal_fusion * fusion) const {
        ggml_metal_fusion_info_count_fusion(finfo, fusion);
    }

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;
    ggml_metal_encoder_t enc;
    ggml_mem_ranges_t    mem_ranges;

    // shared fusion debugging context
    ggml_metal_fusion_info * finfo;

    bool use_concurrency;
    bool use_capture;

    int debug_graph;

private:
    ggml_cgraph * gf;

    int idx_start;
    int idx_end;

    // non-empty node indices
    std::vector<int> idxs;
};

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        ggml_metal_fusion_info * finfo,
        int idx_start,
        int idx_end,
        bool use_concurrency,
        bool use_capture,
        int debug_graph) {
    ggml_metal_op_t res = new ggml_metal_op(
        dev,
        cmd_buf,
        gf,
        finfo,
        idx_start,
        idx_end,
        use_concurrency,
        use_capture,
        debug_graph);

    return res;
}

void ggml_metal_op_free(ggml_metal_op_t ctx) {
    delete ctx;
}

int ggml_metal_op_n_nodes(ggml_metal_op_t ctx) {
    return ctx->n_nodes();
}

static bool ggml_metal_op_concurrency_reset(ggml_metal_op_t ctx) {
    if (!ctx->mem_ranges) {
        return true;
    }

    ggml_metal_encoder_memory_barrier(ctx->enc);

    ggml_mem_ranges_reset(ctx->mem_ranges);

    return true;
}

static bool ggml_metal_op_concurrency_check(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return false;
    }

    return ggml_mem_ranges_check(ctx->mem_ranges, node);
}

static bool ggml_metal_op_concurrency_add(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return true;
    }

    return ggml_mem_ranges_add(ctx->mem_ranges, node);
}

static int32_t ggml_metal_use_count(const ggml_cgraph * gf, const ggml_tensor * t);
static const ggml_tensor * ggml_metal_unwrap_reshape(const ggml_cgraph * gf, const ggml_tensor * t);
static int ggml_metal_op_rms_norm_fwht(ggml_metal_op_t ctx, int idx);
static bool ggml_metal_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b);
static bool ggml_metal_gdn_state_fusion_on(ggml_metal_op_t ctx);
static bool ggml_metal_gdn_state_skip(ggml_metal_op_t ctx, int idx);

static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) {
    struct ggml_tensor * node = ctx->node(idx);

    //GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, ggml_op_name(node->op));

    if (ggml_is_empty(node)) {
        return 1;
    }

    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            {
                // noop -> next node
                if (ctx->debug_graph > 0) {
                    GGML_LOG_DEBUG("%s: node[%5d] - %-12s %s\n", __func__, idx, ggml_op_name(node->op), "(noop)");
                }
            } return 1;
        default:
            {
            } break;
    }

    // A tensor split hands a device an empty slice when the weight has fewer whole
    // quantization blocks than there are devices. The op then contributes nothing, but
    // its result still enters the sum across devices, so write zeros rather than abort.
    if (!ggml_is_empty(node)) {
        bool empty_src = false;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node->src[i] && ggml_is_empty(node->src[i])) {
                empty_src = true;
                break;
            }
        }
        if (empty_src && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)) {
            if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                return 1;
            }
            ggml_metal_pipeline_with_params pz = ggml_metal_library_get_pipeline(ctx->lib, "kernel_zero_bytes");
            if (!pz.pipeline) {
                ggml_metal_cv_t cv = ggml_metal_cv_init();
                pz = ggml_metal_library_compile_pipeline(ctx->lib, "kernel_zero_bytes", "kernel_zero_bytes", cv);
                ggml_metal_cv_free(cv);
            }
            if (pz.pipeline) {
                const uint64_t n = ggml_nbytes(node);
                const int nth = std::min<int>(256, ggml_metal_pipeline_max_theads_per_threadgroup(pz));
                ggml_metal_encoder_set_pipeline(ctx->enc, pz);
                ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(node), 0);
                ggml_metal_encoder_set_bytes   (ctx->enc, (void *) &n, sizeof(n), 1);
                ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (int) ((n + nth - 1)/nth), 1, 1, nth, 1, 1);
                return 1;
            }
        }
    }

    if (!ggml_metal_device_supports_op(ctx->dev, node)) {
        GGML_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(node));
        GGML_ABORT("unsupported op");
    }

    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return 1;
    }

    int n_fuse = 1;

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        bool is_concurrent = ggml_metal_op_concurrency_check(ctx, node);

        if (is_concurrent && ctx->use_fusion()) {
            int n_fuse = 1;
            const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
            if (fusion) {
                // fused kernels write to the last node of the group, not necessarily to the first node's dst
                is_concurrent = ggml_mem_ranges_check(ctx->mem_ranges, ctx->node(idx + n_fuse - 1));
            }
        }

        if (!is_concurrent) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        if (ctx->debug_graph > 0) {
            GGML_LOG_DEBUG("%s: node[%5d] - %-12s %-12s %s\n", __func__, idx, ggml_op_name(node->op), ggml_get_name(node), is_concurrent ? "(concurrent)" : "");
        }
        if (ctx->debug_graph > 1) {
            GGML_TENSOR_LOCALS( int64_t, ne0, node->src[0], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb0, node->src[0], nb);
            GGML_TENSOR_LOCALS( int64_t, ne1, node->src[1], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb1, node->src[1], nb);
            GGML_TENSOR_LOCALS( int64_t, ne2, node->src[2], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb2, node->src[2], nb);
            GGML_TENSOR_LOCALS( int64_t, ne3, node->src[3], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb3, node->src[3], nb);
            GGML_TENSOR_LOCALS( int64_t, ne,  node,         ne);
            GGML_TENSOR_LOCALS(uint64_t, nb,  node,         nb);

            if (node->src[0]) {
                GGML_LOG_DEBUG("%s: src0 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[0]->type), ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                        ggml_is_contiguous(node->src[0]), node->src[0]->name);
            }
            if (node->src[1]) {
                GGML_LOG_DEBUG("%s: src1 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[1]->type), ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                        ggml_is_contiguous(node->src[1]), node->src[1]->name);
            }
            if (node->src[2]) {
                GGML_LOG_DEBUG("%s: src2 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[2]->type), ne20, ne21, ne22, ne23, nb20, nb21, nb22, nb23,
                        ggml_is_contiguous(node->src[2]), node->src[2]->name);
            }
            if (node->src[3]) {
                GGML_LOG_DEBUG("%s: src3 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[3]->type), ne30, ne31, ne32, ne33, nb30, nb31, nb32, nb33,
                        ggml_is_contiguous(node->src[3]), node->src[3]->name);
            }
            if (node) {
                GGML_LOG_DEBUG("%s: node  - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n", __func__, ggml_type_name(node->type), ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                        node->name);
            }
        }
    }

    if ((node->op == GGML_OP_GET_ROWS || node->op == GGML_OP_CPY) &&
        ggml_metal_gdn_state_fusion_on(ctx) && ggml_metal_gdn_state_skip(ctx, idx)) {
        return 1;
    }

    if ((node->op == GGML_OP_RMS_NORM || node->op == GGML_OP_ADD) && ctx->use_fusion() && !ctx->use_concurrency) {
        const int n = ggml_metal_op_rms_norm_fwht(ctx, idx);
        if (n > 0) {
            return n;
        }
    }

    const int n_fwht = ctx->use_fusion() && (node->op == GGML_OP_MUL || node->op == GGML_OP_CONT) ?
        ggml_metal_op_fwht_fused(ctx, idx) : 0;

    if (n_fwht > 0) {
        n_fuse = n_fwht;
    } else
    switch (node->op) {
        case GGML_OP_CONCAT:
            {
                n_fuse = ggml_metal_op_concat(ctx, idx);
            } break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            {
                n_fuse = ggml_metal_op_bin(ctx, idx);
            } break;
        case GGML_OP_ADD_ID:
            {
                n_fuse = ggml_metal_op_add_id(ctx, idx);
            } break;
        case GGML_OP_REPEAT:
            {
                n_fuse = ggml_metal_op_repeat(ctx, idx);
            } break;
        case GGML_OP_ACC:
            {
                n_fuse = ggml_metal_op_acc(ctx, idx);
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
            {
                n_fuse = ggml_metal_op_unary(ctx, idx);
            } break;
        case GGML_OP_SILU_BACK:
            {
                n_fuse = ggml_metal_op_silu_back(ctx, idx);
            } break;
        case GGML_OP_GLU:
            {
                n_fuse = ggml_metal_op_glu(ctx, idx);
            } break;
        case GGML_OP_SUM:
            {
                n_fuse = ggml_metal_op_sum(ctx, idx);
            } break;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            {
                n_fuse = ggml_metal_op_sum_rows(ctx, idx);
            } break;
        case GGML_OP_CUMSUM:
            {
                n_fuse = ggml_metal_op_cumsum(ctx, idx);
            } break;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                n_fuse = ggml_metal_op_lightning_indexer(ctx, idx);
            } break;
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
            {
                n_fuse = ggml_metal_op_dsv4_hc(ctx, idx);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_fuse = ggml_metal_op_topk_moe(ctx, idx);
                if (n_fuse == 0) {
                    n_fuse = ggml_metal_op_soft_max(ctx, idx);
                }
            } break;
        case GGML_OP_SSM_CONV:
            {
                n_fuse = ggml_metal_op_ssm_conv(ctx, idx);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                n_fuse = ggml_metal_op_ssm_scan(ctx, idx);
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            {
                n_fuse = ggml_metal_op_rwkv(ctx, idx);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                n_fuse = ggml_metal_op_gated_delta_net(ctx, idx);
            } break;
        case GGML_OP_TURBO_WHT:
            {
                n_fuse = ggml_metal_op_turbo_wht(ctx, idx);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                n_fuse = ggml_metal_op_solve_tri(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT:
            {
                n_fuse = ggml_metal_op_mul_mat(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                n_fuse = ggml_metal_op_mul_mat_id(ctx, idx);
            } break;
        case GGML_OP_GET_ROWS:
            {
                n_fuse = ggml_metal_op_get_rows(ctx, idx);
            } break;
        case GGML_OP_SET_ROWS:
            {
                n_fuse = ggml_metal_op_set_rows(ctx, idx);
            } break;
        case GGML_OP_DIAG:
            {
                n_fuse = ggml_metal_op_diag(ctx, idx);
            } break;
        case GGML_OP_L2_NORM:
            {
                n_fuse = ggml_metal_op_l2_norm(ctx, idx);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                n_fuse = ggml_metal_op_group_norm(ctx, idx);
            } break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            {
                n_fuse = ggml_metal_op_norm(ctx, idx);
            } break;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            {
                n_fuse = ggml_metal_op_rope(ctx, idx);
            } break;
        case GGML_OP_IM2COL:
            {
                n_fuse = ggml_metal_op_im2col(ctx, idx);
            } break;
        case GGML_OP_CONV_2D:
            {
                n_fuse = ggml_metal_op_conv_2d(ctx, idx);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                n_fuse = ggml_metal_op_conv_2d_dw(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                n_fuse = ggml_metal_op_conv_transpose_1d(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_fuse = ggml_metal_op_conv_transpose_2d(ctx, idx);
            } break;
        case GGML_OP_COL2IM_1D:
            {
                n_fuse = ggml_metal_op_col2im_1d(ctx, idx);
            } break;
        case GGML_OP_CONV_3D:
            {
                n_fuse = ggml_metal_op_conv_3d(ctx, idx);
            } break;
        case GGML_OP_UPSCALE:
            {
                n_fuse = ggml_metal_op_upscale(ctx, idx);
            } break;
        case GGML_OP_PAD:
            {
                n_fuse = ggml_metal_op_pad(ctx, idx);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                n_fuse = ggml_metal_op_pad_reflect_1d(ctx, idx);
            } break;
        case GGML_OP_ROLL:
            {
                n_fuse = ggml_metal_op_roll(ctx, idx);
            } break;
        case GGML_OP_ARANGE:
            {
                n_fuse = ggml_metal_op_arange(ctx, idx);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                n_fuse = ggml_metal_op_timestep_embedding(ctx, idx);
            } break;
        case GGML_OP_ARGSORT:
            {
                n_fuse = ggml_metal_op_argsort(ctx, idx);
            } break;
        case GGML_OP_TOP_K:
            {
                n_fuse = ggml_metal_op_top_k(ctx, idx);
            } break;
        case GGML_OP_TRI:
            {
                n_fuse = ggml_metal_op_tri(ctx, idx);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                n_fuse = ggml_metal_op_flash_attn_ext(ctx, idx);
            } break;
        case GGML_OP_SET:
            {
                n_fuse = ggml_metal_op_set(ctx, idx);
            } break;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            {
                n_fuse = ggml_metal_op_cpy(ctx, idx);
            } break;
        case GGML_OP_POOL_1D:
            {
                n_fuse = ggml_metal_op_pool_1d(ctx, idx);
            } break;
        case GGML_OP_POOL_2D:
            {
                n_fuse = ggml_metal_op_pool_2d(ctx, idx);
            } break;
        case GGML_OP_ARGMAX:
            {
                n_fuse = ggml_metal_op_argmax(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                n_fuse = ggml_metal_op_opt_step_adamw(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_SGD:
            {
                n_fuse = ggml_metal_op_opt_step_sgd(ctx, idx);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                n_fuse = ggml_metal_op_count_equal(ctx, idx);
            } break;
        default:
            {
                GGML_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, idx, ggml_op_name(node->op));
                GGML_ABORT("fatal error");
            }
    }

    if (ctx->debug_graph > 0) {
        if (n_fuse > 1) {
            GGML_LOG_DEBUG("%s:               fuse %d ops\n", __func__, n_fuse);
        }
    }

    // update the mem ranges in the encoding context
    for (int i = 0; i < n_fuse; ++i) {
        if (!ggml_metal_op_concurrency_add(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    }

    return n_fuse;
}


// TOSH_OPPROF=1: CPU time spent encoding each op kind, to compare dispatch cost across engines
#include <time.h>
static uint64_t g_opprof_ns[GGML_OP_COUNT];
static uint64_t g_opprof_n [GGML_OP_COUNT];
static bool tosh_opprof_on(void) {
    static int v = -1;
    if (v < 0) { v = getenv("TOSH_OPPROF") != nullptr; }
    return v;
}
static void tosh_opprof_dump(void) {
    if (!tosh_opprof_on()) return;
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        if (g_opprof_n[i] == 0) continue;
        fprintf(stderr, "TOSH_OPPROF %-22s n=%-7llu total_us=%-10.1f us/op=%.3f\n",
                ggml_op_name((enum ggml_op) i),
                (unsigned long long) g_opprof_n[i],
                g_opprof_ns[i]/1e3, (double) g_opprof_ns[i]/g_opprof_n[i]/1e3);
    }
}

int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx) {
    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_push(ctx->enc, ggml_op_desc(ctx->node(idx)));
    }

    static bool reg = false;
    if (!reg && tosh_opprof_on()) { reg = true; atexit(tosh_opprof_dump); }
    const enum ggml_op prof_op = ctx->node(idx)->op;
    const uint64_t prof_t0 = tosh_opprof_on() ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;

    int res = ggml_metal_op_encode_impl(ctx, idx);

    if (prof_t0) {
        g_opprof_ns[prof_op] += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - prof_t0;
        g_opprof_n [prof_op] += 1;
    }
    if (idx + res > ctx->n_nodes()) {
        GGML_ABORT("fusion error: nodes spanning multiple encoders have been fused. this indicates a bug in the fusion logic %s",
                "https://github.com/ggml-org/llama.cpp/pull/14849");
    }

    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_pop(ctx->enc);
    }

    return res;
}

int ggml_metal_op_concat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t dim = ((const int32_t *) op->op_params)[0];

    const bool is_q = ggml_is_quantized(op->type);

    // for quantized types, concat is done at the block level (nb0 == type_size == block size)
    int32_t ne00_arg = ne00;
    int32_t ne10_arg = ne10;
    int32_t ne0_arg  = ne0;
    if (is_q) {
        const int32_t blck = ggml_blck_size(op->type);
        GGML_ASSERT(ne00 % blck == 0);
        GGML_ASSERT(ne10 % blck == 0);
        GGML_ASSERT(ne0  % blck == 0);
        ne00_arg = ne00/blck;
        ne10_arg = ne10/blck;
        ne0_arg  = ne0/blck;
    }

    ggml_metal_kargs_concat args = {
        /*.ne00 =*/ ne00_arg,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10_arg,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0_arg,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.dim  =*/ dim,
    };

    auto pipeline = ggml_metal_library_get_pipeline_concat(lib, op->type);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    int nth = std::min(256, ne0_arg);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;
    if (nth < 256) {
        nrptg = std::min((256 + nth - 1) / nth, ne1);
        if (nrptg * nth > 256) {
            nrptg = 256 / nth;
        }
    }

    const int nw0 = (ne1 + nrptg - 1) / nrptg;

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0, ne2, ne3, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_repeat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_repeat(lib, op->type);

    ggml_metal_kargs_repeat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_acc(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type, false);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ pnb1,
        /*.nb02 =*/ pnb2,
        /*.nb03 =*/ pnb3,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
        /*.offs =*/ offs,
        /*.o1   =*/ { 0 },
    };

    auto pipeline = ggml_metal_library_get_pipeline_bin_one(lib, GGML_OP_ADD);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    int nth = 1;

    while (2*nth < args.ne0 && nth < nth_max) {
        nth *= 2;
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, ne11, ne12, ne13, nth, 1, 1);

    return 1;
}

int ggml_metal_op_unary(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_unary args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.slope =*/ 0.0,
        /*.scale =*/ 0.0,
        /*.bias  =*/ 0.0,
        /*.val   =*/ 0.0,
        /*.min   =*/ 0.0,
        /*.max   =*/ 0.0,
    };

    if (op->op == GGML_OP_LEAKY_RELU) {
        args.slope = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_SCALE) {
        args.scale = ggml_get_op_params_f32(op, 0);
        args.bias  = ggml_get_op_params_f32(op, 1);
    }

    if (op->op == GGML_OP_FILL) {
        args.val = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_CLAMP) {
        args.min = ggml_get_op_params_f32(op, 0);
        args.max = ggml_get_op_params_f32(op, 1);
    }

    if (op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
        args.slope = ggml_get_op_params_f32(op, 1); // alpha_n
        args.scale = ggml_get_op_params_f32(op, 2); // alpha_p
        args.bias  = ggml_get_op_params_f32(op, 3); // beta
        args.val   = ggml_get_op_params_f32(op, 4); // eps
    }

    auto pipeline = ggml_metal_library_get_pipeline_unary(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    if (pipeline.cnt) {
        const int n = pipeline.c4 ? ggml_nelements(op)/4 : ggml_nelements(op);

        ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        const int nth = MIN(args.ne00, nth_max);
        const int nk0 = (args.ne00 + nth - 1)/nth;

        ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne01, ne02, ne03, nth, 1, 1);
    }

    return 1;
}

int ggml_metal_op_glu(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    if (op->src[1]) {
        GGML_ASSERT(ggml_are_same_shape(op->src[0], op->src[1]));
    }

    auto pipeline = ggml_metal_library_get_pipeline_glu(lib, op);

    const int32_t swp = ggml_get_op_params_i32(op, 1);
    const float alpha = ggml_get_op_params_f32(op, 2);
    const float limit = ggml_get_op_params_f32(op, 3);

    const int32_t i00 = swp ? ne0 : 0;
    const int32_t i10 = swp ? 0 : ne0;

    ggml_metal_kargs_glu args = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.ne10 =*/ op->src[1] ? ne10 : ne00,
        /*.nb11 =*/ op->src[1] ? nb11 : nb01,
        /*.ne0  =*/ ne0,
        /*.nb1  =*/ nb1,
        /*.i00  =*/ op->src[1] ? 0 : i00,
        /*.i10  =*/ op->src[1] ? 0 : i10,
        /*.alpha=*/ alpha,
        /*.limit=*/ limit
    };

    const int64_t nrows = ggml_nrows(op->src[0]);

    const int32_t nth = std::max(1, std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00/2));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op  = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const uint64_t n = (uint64_t) ggml_nelements(op->src[0]);

    ggml_metal_kargs_sum args = {
        /*.np =*/ n,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum(lib, op);

    const int nw = ggml_metal_device_get_props(ctx->dev)->simd_width;

    int nth = nw;

    while (nth < (int) n && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) n);

    const int nsg = (nth + nw - 1) / nw;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, nsg * sizeof(float), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_sum_rows args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum_rows(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < args.ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) args.ne00);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_cumsum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline_blk = ggml_metal_library_get_pipeline_cumsum_blk(lib, op);

    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_blk)) {
        nth *= 2;
    }

    GGML_ASSERT(ne00 <= nth*nth);

    const int64_t net0 = (ne00 + nth - 1) / nth;
    const int64_t net1 = ne01;
    const int64_t net2 = ne02;
    const int64_t net3 = ne03;

    const uint64_t nbt0 = sizeof(float);
    const uint64_t nbt1 = net0*nbt0;
    const uint64_t nbt2 = net1*nbt1;
    const uint64_t nbt3 = net2*nbt2;

    // one float per lane of the real simd width (64 on wave64)
    const size_t smem = GGML_PAD(ggml_metal_device_get_props(ctx->dev)->simd_width*sizeof(float), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    {
        ggml_metal_kargs_cumsum_blk args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.net0 =*/ net0,
            /*.net1 =*/ net1,
            /*.net2 =*/ net2,
            /*.net3 =*/ net3,
            /*.nbt0 =*/ nbt0,
            /*.nbt1 =*/ nbt1,
            /*.nbt2 =*/ nbt2,
            /*.nbt3 =*/ nbt3,
            /*.outb =*/ ne00 > nth,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
    }

    if (ne00 > nth) {
        ggml_metal_op_concurrency_reset(ctx);

        {
            ggml_metal_kargs_cumsum_blk args = {
                /*.ne00 =*/ net0,
                /*.ne01 =*/ net1,
                /*.ne02 =*/ net2,
                /*.ne03 =*/ net3,
                /*.nb00 =*/ nbt0,
                /*.nb01 =*/ nbt1,
                /*.nb02 =*/ nbt2,
                /*.nb03 =*/ nbt3,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
                /*.outb =*/ false,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, net1, net2, net3, nth, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        {
            auto pipeline_add = ggml_metal_library_get_pipeline_cumsum_add(lib, op);

            ggml_metal_kargs_cumsum_add args = {
                /*.ne00 =*/ ne00,
                /*.ne01 =*/ ne01,
                /*.ne02 =*/ ne02,
                /*.ne03 =*/ ne03,
                /*.nb00 =*/ nb00,
                /*.nb01 =*/ nb01,
                /*.nb02 =*/ nb02,
                /*.nb03 =*/ nb03,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_add);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
        }
    }

    return 1;
}

int ggml_metal_op_get_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_get_rows(lib, op->src[0]->type);

    ggml_metal_kargs_get_rows args = {
        /*.ne00t =*/ ggml_is_quantized(op->src[0]->type) ? ne00/16 : ne00,
        /*.ne00  =*/ ne00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne10  =*/ ne10,
        /*.nb10  =*/ nb10,
        /*.nb11  =*/ nb11,
        /*.nb12  =*/ nb12,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    const int nth = std::min(args.ne00t, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const int nw0 = (args.ne00t + nth - 1)/nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne10, ne11, ne12, nth, 1, 1);

    return 1;
}

int ggml_metal_op_set_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int simd_width = ggml_metal_device_get_props(ctx->dev)->simd_width;
    const bool turbo_coop = op->type == GGML_TYPE_TURBO3_0 || op->type == GGML_TYPE_TURBO4_0;
    auto pipeline = ggml_metal_library_get_pipeline_set_rows(lib, op, turbo_coop && simd_width == 64);

    const int32_t nk0 = ne0/ggml_blck_size(op->type);

    int nth = 32; // SIMD width

    while (nth < nk0 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    int nrptg = 1;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1)/nk0;
        nth   = nk0;

        if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nrptg--;
        }
    }

    nth = std::min(nth, nk0);

    ggml_metal_kargs_set_rows args = {
        /*.nk0  =*/ nk0,
        /*.ne01 =*/ ne01,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    // The turbo3/4 store is cooperative: one 32-lane simdgroup per 128-element group,
    // dispatched over a flattened (group, row) grid so decode fills the GPU.
    if (turbo_coop) {
        GGML_ASSERT((simd_width == 32 || simd_width == 64) && "TurboQuant cooperative store requires wave32 or wave64");
        ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne01, ne02, ne03, simd_width, 1, 1);
    } else {
        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);
    }

    return 1;
}

int ggml_metal_op_diag(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(int32_t,  ne, op, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb, op, nb);

    ggml_metal_kargs_diag args = {
        /*.ne00 =*/ne00,
        /*.ne01 =*/ne01,
        /*.ne02 =*/ne02,
        /*.ne03 =*/ne03,
        /*.nb00 =*/nb00,
        /*.nb01 =*/nb01,
        /*.nb02 =*/nb02,
        /*.nb03 =*/nb03,
        /*.ne0  =*/ne0,
        /*.ne1  =*/ne1,
        /*.ne2  =*/ne2,
        /*.ne3  =*/ne3,
        /*.nb0  =*/nb0,
        /*.nb1  =*/nb1,
        /*.nb2  =*/nb2,
        /*.nb3  =*/nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_diag(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, 32, 1, 1);

    return 1;
}

int ggml_metal_op_lightning_indexer(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_LIGHTNING_INDEXER);

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * w = op->src[2];
    const ggml_tensor * m = op->src[3];

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32  ||
                k->type == GGML_TYPE_F16  ||
                k->type == GGML_TYPE_BF16 ||
                k->type == GGML_TYPE_Q4_0 ||
                k->type == GGML_TYPE_Q4_1 ||
                k->type == GGML_TYPE_Q5_0 ||
                k->type == GGML_TYPE_Q5_1 ||
                k->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(m->type == GGML_TYPE_F16);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    GGML_ASSERT(q->ne[0] == OP_LIGHTNING_INDEXER_DK);
    GGML_ASSERT(q->ne[1] == OP_LIGHTNING_INDEXER_NH);

    ggml_metal_kargs_lightning_indexer args = {
        /*.n_kv      =*/ (int32_t) k->ne[2],
        /*.n_batch   =*/ (int32_t) q->ne[2],
        /*.mask_ne3  =*/ (int32_t) m->ne[3],
        /*.nb1       =*/ op->nb[1],
        /*.nb3       =*/ op->nb[3],
        /*.nbq1      =*/ q->nb[1],
        /*.nbq2      =*/ q->nb[2],
        /*.nbq3      =*/ q->nb[3],
        /*.nbk2      =*/ k->nb[2],
        /*.nbk3      =*/ k->nb[3],
        /*.nbw1      =*/ w->nb[1],
        /*.nbw3      =*/ w->nb[3],
        /*.nbm1      =*/ m->nb[1],
        /*.nbm3      =*/ m->nb[3],
    };

    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(q),  1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(k),  2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(w),  3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(m),  4);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 5);

    const int nsg   = OP_LIGHTNING_INDEXER_NSG;
    const int nkptg = OP_LIGHTNING_INDEXER_NKPSG*nsg;
    const int nbptg = OP_LIGHTNING_INDEXER_NBPTG;

    auto pipeline = ggml_metal_library_get_pipeline_lightning_indexer(ctx->lib, op);
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_dispatch_threadgroups(enc,
            (k->ne[2] + nkptg - 1)/nkptg,
            (q->ne[2] + nbptg - 1)/nbptg,
            q->ne[3], 32, nsg, 1);

    return 1;
}

int ggml_metal_op_dsv4_hc(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_encoder_t enc = ctx->enc;
    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc(ctx->lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);

    switch (op->op) {
        case GGML_OP_DSV4_HC_COMB:
            {
                const ggml_tensor * mixes = op->src[0];
                const ggml_tensor * scale = op->src[1];
                const ggml_tensor * base  = op->src[2];

                GGML_ASSERT(mixes->type == GGML_TYPE_F32);
                GGML_ASSERT(scale->type == GGML_TYPE_F32);
                GGML_ASSERT(base->type  == GGML_TYPE_F32);
                GGML_ASSERT(op->type    == GGML_TYPE_F32);
                GGML_ASSERT(mixes->ne[0] == 24);
                GGML_ASSERT(op->ne[0] == 4 && op->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_comb args = {
                    /*.n_tokens =*/ (int32_t) mixes->ne[1],
                    /*.n_iter   =*/ ggml_get_op_params_i32(op, 1),
                    /*.nb_m0    =*/ mixes->nb[0],
                    /*.nb_m1    =*/ mixes->nb[1],
                    /*.nb_s0    =*/ scale->nb[0],
                    /*.nb_b0    =*/ base->nb[0],
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.nb_d2    =*/ op->nb[2],
                    /*.eps      =*/ ggml_get_op_params_f32(op, 0),
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(mixes), 1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(scale), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(base),  3);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),    4);

                // One SIMDgroup owns one 4x4 Sinkhorn matrix. Packing up to four
                // independent tokens per threadgroup keeps both decode and prompt
                // dispatches compact without any threadgroup-memory synchronization.
                const int nsg = std::min(4, args.n_tokens);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (args.n_tokens + nsg - 1)/nsg, 1, 1, 32, nsg, 1);
            } break;
        case GGML_OP_DSV4_HC_PRE:
            {
                const ggml_tensor * x       = op->src[0];
                const ggml_tensor * weights = op->src[1];

                GGML_ASSERT(x->type       == GGML_TYPE_F32);
                GGML_ASSERT(weights->type == GGML_TYPE_F32);
                GGML_ASSERT(op->type      == GGML_TYPE_F32);

                ggml_metal_kargs_dsv4_hc_pre args = {
                    /*.n_embd   =*/ (int32_t) x->ne[0],
                    /*.n_tokens =*/ (int32_t) x->ne[2],
                    /*.nb_x0    =*/ x->nb[0],
                    /*.nb_x1    =*/ x->nb[1],
                    /*.nb_x2    =*/ x->nb[2],
                    /*.nb_w0    =*/ weights->nb[0],
                    /*.nb_w1    =*/ weights->nb[1],
                    /*.nb_w2    =*/ weights->nb[2],
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.scale    =*/ ggml_get_op_params_f32(op, 0),
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),       1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weights), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),      3);

                const int n_tiles = (args.n_embd + 31)/32;
                const int nsg = std::min(4, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, 32, nsg, 1);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                const ggml_tensor * x        = op->src[0];
                const ggml_tensor * residual = op->src[1];
                const ggml_tensor * post     = op->src[2];
                const ggml_tensor * comb     = op->src[3];

                GGML_ASSERT(x->type        == GGML_TYPE_F32);
                GGML_ASSERT(residual->type == GGML_TYPE_F32);
                GGML_ASSERT(post->type     == GGML_TYPE_F32);
                GGML_ASSERT(op->type       == GGML_TYPE_F32);
                GGML_ASSERT(residual->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_post args = {
                    /*.n_embd   =*/ (int32_t) x->ne[0],
                    /*.n_tokens =*/ (int32_t) x->ne[1],
                    /*.nb_x0    =*/ x->nb[0],
                    /*.nb_x1    =*/ x->nb[1],
                    /*.nb_r0    =*/ residual->nb[0],
                    /*.nb_r1    =*/ residual->nb[1],
                    /*.nb_r2    =*/ residual->nb[2],
                    /*.nb_p0    =*/ post->nb[0],
                    /*.nb_p1    =*/ post->nb[1],
                    /*.nb_c0    =*/ comb ? comb->nb[0] : 0,
                    /*.nb_c1    =*/ comb ? comb->nb[1] : 0,
                    /*.nb_c2    =*/ comb ? comb->nb[2] : 0,
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.nb_d2    =*/ op->nb[2],
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),        1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(residual), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(post),     3);
                if (comb) {
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(comb), 4);
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   5);
                } else {
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   4);
                }

                const int n_tiles = (args.n_embd + 31)/32;
                const int nsg = std::min(4, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, 32, nsg, 1);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    return 1;
}

int ggml_metal_op_soft_max(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_TOPK_MOE) {
            return ggml_metal_op_topk_moe(ctx, idx);
        }
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float scale;
    float max_bias;

    memcpy(&scale,    ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(max_bias));

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // softmax

    ggml_metal_kargs_soft_max args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne11        =*/ ne11,
        /*.ne12        =*/ ne12,
        /*.ne13        =*/ ne13,
        /*.nb11        =*/ nb11,
        /*.nb12        =*/ nb12,
        /*.nb13        =*/ nb13,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.scale       =*/ scale,
        /*.max_bias    =*/ max_bias,
        /*.m0          =*/ m0,
        /*.m1          =*/ m1,
        /*.n_head_log2 =*/ n_head_log2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_soft_max(lib, op);

    int nth = 32; // SIMD width

    if (ne00%4 == 0) {
        while (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_ssm_conv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    int n_fuse = 1;
    bool use_silu = false;

    if (ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SSM_CONV_SILU) {
            n_fuse = n;
            use_silu = true;

            ctx->count_fusions(fusion);
        }
    }

    ggml_metal_kargs_ssm_conv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne11 =*/ ne11,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
    };

    const ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(n_fuse > 1 ? ctx->node(idx + n_fuse - 1) : op);

    // Use batched kernel for prefill (ne1 > 1) to reduce threadgroup dispatch overhead
    const bool use_batched = (ne1 > 1);

    if (use_batched) {
        // Determine the smallest power of 2 that's >= ne1, but <= 256
        int BATCH_SIZE;
        if      (ne1 > 128) BATCH_SIZE = 256;
        else if (ne1 > 64 ) BATCH_SIZE = 128;
        else if (ne1 > 32 ) BATCH_SIZE = 64;
        else if (ne1 > 16 ) BATCH_SIZE = 32;
        else if (ne1 > 8  ) BATCH_SIZE = 16;
        else if (ne1 > 4  ) BATCH_SIZE = 8;
        else                BATCH_SIZE = 2;

        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv_batched(lib, op, BATCH_SIZE, (int32_t) ne10, use_silu);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

        // Dispatch: ne01 rows, ceil(ne1/BATCH_SIZE) token batches, ne02 sequences
        // Each threadgroup has BATCH_SIZE threads, each handling one token
        const int n_token_batches = (ne1 + BATCH_SIZE - 1) / BATCH_SIZE;
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, n_token_batches, ne02, BATCH_SIZE, 1, 1);
    } else {
        // Decode (ne1 == 1): pack rows across a threadgroup; token/seq stay grid
        // dims. 64 threads keeps enough threadgroups to fill the CUs. Fuse a
        // following SiLU on this output into the kernel (drops its dispatch and
        // the intermediate tensor); the output layout is identical (elementwise).
        bool fuse_silu = false;
        ggml_tensor * out = op;
        if (ctx->use_fusion()) {
            const ggml_op fops[2] = { GGML_OP_SSM_CONV, GGML_OP_UNARY };
            if (ctx->can_fuse(idx, fops, 2)) {
                ggml_tensor * next = ctx->node(idx + 1);
                if (ggml_get_unary_op(next) == GGML_UNARY_OP_SILU && next->src[0] == op) {
                    fuse_silu = true;
                    out = next;
                }
            }
        }

        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv_decode(lib, op, fuse_silu);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(out),        3);

        const int nth  = 64;
        const int ntgx = (ne01 + nth - 1) / nth;
        ggml_metal_encoder_dispatch_threadgroups(enc, ntgx, ne1, ne02, nth, 1, 1);

        return fuse_silu ? 2 : 1;
    }

    if (n_fuse > 1 && ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: SSM_CONV + UNARY\n", __func__);
    }

    return n_fuse;
}

int ggml_metal_op_ssm_scan(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;
    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne4, op->src[4], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb4, op->src[4], nb);
    GGML_TENSOR_LOCALS( int32_t, ne5, op->src[5], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb5, op->src[5], nb);
    GGML_TENSOR_LOCALS( int32_t, ne6, op->src[6], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb6, op->src[6], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const ggml_tensor * src3 = op->src[3];
    const ggml_tensor * src4 = op->src[4];
    const ggml_tensor * src5 = op->src[5];
    const ggml_tensor * src6 = op->src[6];

    GGML_ASSERT(src3);
    GGML_ASSERT(src4);
    GGML_ASSERT(src5);
    GGML_ASSERT(src6);

    const int64_t d_state      = ne00;
    const int64_t d_inner      = ne01;
    const int64_t n_head       = ne02;
    const int64_t n_group      = ne41;
    const int64_t n_seq_tokens = ne12;
    const int64_t n_seqs       = ne13;
    const int64_t K            = ggml_get_op_params_i32(op, 0);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(op->src[1]) + K*d_state*d_inner*n_head*n_seqs == ggml_nelements(op));

    ggml_metal_kargs_ssm_scan args = {
        /*.d_state      =*/ d_state,
        /*.d_inner      =*/ d_inner,
        /*.n_head       =*/ n_head,
        /*.n_group      =*/ n_group,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seq_tokens_total =*/ n_seq_tokens,
        /*.token_offset =*/ 0,
        /*.n_seqs       =*/ n_seqs,
        /*.K            =*/ K,
        /*.s_off        =*/ ggml_nelements(op->src[1]) * sizeof(float),
        /*.nb00         =*/ nb00,
        /*.nb01         =*/ nb01,
        /*.nb02         =*/ nb02,
        /*.nb03         =*/ nb03,
        /*.nb10         =*/ nb10,
        /*.nb11         =*/ nb11,
        /*.nb12         =*/ nb12,
        /*.ns12         =*/ nb12/nb10,
        /*.nb13         =*/ nb13,
        /*.nb20         =*/ nb20,
        /*.nb21         =*/ nb21,
        /*.ns21         =*/ nb21/nb20,
        /*.nb22         =*/ nb22,
        /*.ne30         =*/ ne30,
        /*.nb31         =*/ nb31,
        /*.nb41         =*/ nb41,
        /*.nb42         =*/ nb42,
        /*.ns42         =*/ nb42/nb40,
        /*.nb43         =*/ nb43,
        /*.nb51         =*/ nb51,
        /*.nb52         =*/ nb52,
        /*.ns52         =*/ nb52/nb50,
        /*.nb53         =*/ nb53,
        /*.nb0          =*/ nb0,
    };

    constexpr int64_t CHUNK = OP_SSM_SCAN_SSD_CS;

    const int64_t snap_reserve = K > 1 ? K : 0; // tokens reserved for sequential kernel rollback snapshots
    const int64_t mma_tokens = ((n_seq_tokens - snap_reserve) / CHUNK) * CHUNK; // largest multiple of CHUNK that leaves snap_reserve for the tail
    const bool use_mma =
        mma_tokens > 0 &&
        ne30 == 1 && // checks that A tensor is set to scalar decay per head (A shape {1, n_head})
        props_dev->has_simdgroup_mm && // hardware check for M1 or newer
        d_state % 8 == 0 && // d_state must be multiple of 8 to align with simdgroup_float 8x8 tiles
        d_inner == OP_SSM_SCAN_SSD_HD; // mma kernel is specialized for the Mamba-2 head dim; this checks it

    const auto dispatch = [&](ggml_metal_pipeline_with_params pipeline, int64_t nth, int64_t n_tg_x) {
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        GGML_ASSERT(pipeline.smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 5);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), 6);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), 7);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         8);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, n_tg_x, n_head, n_seqs, nth, 1, 1);
    };

    if (!use_mma) {
        dispatch(ggml_metal_library_get_pipeline_ssm_scan(lib, op, false), d_state, d_inner);
        return 1;
    }

    args.n_seq_tokens = mma_tokens;
    dispatch(
        ggml_metal_library_get_pipeline_ssm_scan_ssd_mma(lib, op),
        OP_SSM_SCAN_SSD_NSG*32,
        1);

    if (mma_tokens < n_seq_tokens) {
        ggml_metal_op_concurrency_reset(ctx);

        args.n_seq_tokens = n_seq_tokens - mma_tokens;
        args.token_offset = mma_tokens;
        dispatch(ggml_metal_library_get_pipeline_ssm_scan(lib, op, true), d_state, d_inner);
    }

    return 1;
}

int ggml_metal_op_rwkv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int64_t B = op->op == GGML_OP_RWKV_WKV6 ? op->src[5]->ne[1] : op->src[6]->ne[1];
    const int64_t T = op->src[0]->ne[2];
    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    auto pipeline = ggml_metal_library_get_pipeline_rwkv(lib, op);

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++);
    if (op->op == GGML_OP_RWKV_WKV7) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), ida++);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &B, sizeof(B), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &T, sizeof(T), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &C, sizeof(C), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &H, sizeof(H), ida++);

    ggml_metal_encoder_dispatch_threadgroups(enc, B * H, 1, 1, C/H, 1, 1);

    return 1;
}

static bool ggml_metal_gdn_single_use(const ggml_cgraph * gf, const ggml_tensor * t) {
    return t && ggml_metal_use_count(gf, t) == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT) &&
           t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

// the l2 norm of q or k, scale(rms_norm(x, eps/n), 1/sqrt(n)), done by the kernel
struct ggml_metal_gdn_l2 {
    const ggml_tensor * scale = nullptr, * norm = nullptr, * x = nullptr;
    float eps = 0.0f;
};

static bool ggml_metal_gdn_l2_src(const ggml_cgraph * gf, const ggml_tensor * t, ggml_metal_gdn_l2 & r) {
    // a prompt keeps its separate norm: inside the kernel it would sit on the serial token loop
    if (t->ne[2] != 1) {
        return false;
    }
    r.scale = ggml_metal_unwrap_reshape(gf, t);
    if (!r.scale || r.scale->op != GGML_OP_SCALE || !ggml_metal_gdn_single_use(gf, r.scale) ||
        ggml_get_op_params_f32(r.scale, 1) != 0.0f) {
        return false;
    }
    r.norm = r.scale->src[0];
    if (r.norm->op != GGML_OP_RMS_NORM || !ggml_metal_gdn_single_use(gf, r.norm)) {
        return false;
    }
    r.x = r.norm->src[0];
    const float n = (float) r.x->ne[0];
    if (r.x->type != GGML_TYPE_F32 || r.x->nb[0] != sizeof(float) || !ggml_are_same_shape(r.x, t) ||
        fabsf(ggml_get_op_params_f32(r.scale, 0) - 1.0f/sqrtf(n)) > 1e-7f) {
        return false;
    }
    r.eps = ggml_get_op_params_f32(r.norm, 0)*n;
    return true;
}

// A single-sequence gated_delta_net can read its state from the recurrent cache row and write
// the new state back into the cache, so the gather before it and the copy after it are skipped.
// The three nodes decide with the same predicates, so a skipped node is always covered.
static const ggml_tensor * ggml_metal_gdn_rows_src(const ggml_cgraph * gf, const ggml_tensor * gdn) {
    const ggml_tensor * st = gdn->src[5];
    if (st->ne[3] != 1 || ggml_get_op_params_i32(gdn, 0) != 1) {
        return nullptr;
    }
    const ggml_tensor * gr = st;
    while (gr && gr->op == GGML_OP_RESHAPE) {
        if (ggml_metal_use_count(gf, gr) != 1) {
            return nullptr;
        }
        gr = gr->src[0];
    }
    if (!gr || gr->op != GGML_OP_GET_ROWS || ggml_metal_use_count(gf, gr) != 1 ||
        (gr->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        gr->type != GGML_TYPE_F32 || gr->src[0]->type != GGML_TYPE_F32 || gr->src[1]->type != GGML_TYPE_I32 ||
        gr->src[0]->nb[0] != sizeof(float) || gr->src[0]->ne[0] != ggml_nelements(st) ||
        gr->ne[1] != 1 || gr->src[1]->ne[0] != 1) {
        return nullptr;
    }
    return gr;
}

static const ggml_tensor * ggml_metal_gdn_fold_dst(ggml_metal_op_t ctx, int idx) {
    const ggml_tensor * gdn = ctx->node(idx);
    if (gdn->src[5]->ne[3] != 1 || ggml_get_op_params_i32(gdn, 0) != 1) {
        return nullptr;
    }
    const size_t attn_size = (size_t) ggml_nelements(gdn->src[2])*sizeof(float);
    const int64_t n_state  = ggml_nelements(gdn->src[5]);
    for (int i = idx + 1; i < std::min(ctx->n_nodes(), idx + 16); ++i) {
        const ggml_tensor * c = ctx->node(i);
        if (c->op != GGML_OP_CPY) {
            continue;
        }
        const ggml_tensor * a = c->src[0];
        if (a->view_src == gdn && a->view_offs == attn_size && a->type == GGML_TYPE_F32 &&
            ggml_nelements(a) == n_state && ggml_is_contiguous(a) &&
            c->type == GGML_TYPE_F32 && ggml_is_contiguous(c) &&
            ggml_metal_use_count(ctx->graph(), a) == 1) {
            return c;
        }
    }
    return nullptr;
}

static bool ggml_metal_gdn_state_fusion_on(ggml_metal_op_t ctx) {
    static const bool disabled = getenv("TOSH_GDN_STATE_FUSION_DISABLE") != nullptr;
    return ctx->use_fusion() && !ctx->use_concurrency && !disabled;
}

// true when node idx is a gather or a copy that a nearby gated_delta_net takes over
static bool ggml_metal_gdn_state_skip(ggml_metal_op_t ctx, int idx) {
    const ggml_tensor * node = ctx->node(idx);
    if (node->op == GGML_OP_GET_ROWS) {
        for (int i = idx + 1; i < std::min(ctx->n_nodes(), idx + 32); ++i) {
            const ggml_tensor * g = ctx->node(i);
            if (g->op == GGML_OP_GATED_DELTA_NET && ggml_metal_gdn_rows_src(ctx->graph(), g) == node) {
                return true;
            }
        }
    } else if (node->op == GGML_OP_CPY) {
        for (int i = idx - 1; i >= std::max(0, idx - 16); --i) {
            if (ctx->node(i)->op == GGML_OP_GATED_DELTA_NET) {
                return ggml_metal_gdn_fold_dst(ctx, i) == node;
            }
        }
    } else if (node->op == GGML_OP_RMS_NORM || node->op == GGML_OP_SCALE) {
        for (int i = idx + 1; i < std::min(ctx->n_nodes(), idx + 40); ++i) {
            const ggml_tensor * g = ctx->node(i);
            if (g->op != GGML_OP_GATED_DELTA_NET) {
                continue;
            }
            ggml_metal_gdn_l2 lq, lk;
            if (((ggml_metal_gdn_l2_src(ctx->graph(), g->src[0], lq) && (node == lq.scale || node == lq.norm)) ||
                (ggml_metal_gdn_l2_src(ctx->graph(), g->src[1], lk) && (node == lk.scale || node == lk.norm)))) {
                return true;
            }
            return false;
        }
    }
    return false;
}

int ggml_metal_op_gated_delta_net(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion();
    const int  debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    const bool fusion = ggml_metal_gdn_state_fusion_on(ctx);
    const ggml_tensor * rows_src = fusion ? ggml_metal_gdn_rows_src(ctx->graph(), op) : nullptr;
    const ggml_tensor * fold_dst = fusion ? ggml_metal_gdn_fold_dst(ctx, idx) : nullptr;

    ggml_metal_gdn_l2 l2q, l2k;
    const bool use_raw = ggml_get_op_params_i32(op, 1) != 0;
    const bool use_l2q = fusion && ggml_metal_gdn_l2_src(ctx->graph(), op->src[0], l2q);
    const bool use_l2k = fusion && ggml_metal_gdn_l2_src(ctx->graph(), op->src[1], l2k);

    auto pipeline = ggml_metal_library_get_pipeline_gated_delta_net(lib, op, rows_src != nullptr, fold_dst != nullptr,
            use_raw, use_l2q, use_l2k);

    const ggml_tensor * t_q = use_l2q ? l2q.x : op->src[0];
    const ggml_tensor * t_k = use_l2k ? l2k.x : op->src[1];

    GGML_TENSOR_LOCALS( int32_t, ne0, t_q,        ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, t_q,        nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, t_k,        ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, t_k,        nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    int ida = 0;

    ggml_metal_kargs_gated_delta_net args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne20 =*/ ne20,
        /*.ne21 =*/ ne21,
        /*.ne22 =*/ ne22,
        /*.ne23 =*/ ne23,
        /*.nb20 =*/ nb20,
        /*.nb21 =*/ nb21,
        /*.nb22 =*/ nb22,
        /*.nb23 =*/ nb23,
        /*.ns02 =*/ (int32_t) (nb02/sizeof(float)),
        /*.ns12 =*/ (int32_t) (nb12/sizeof(float)),
        /*.ns22 =*/ (int32_t) (nb22/sizeof(float)),
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.s_nb1  =*/ rows_src ? rows_src->src[0]->nb[1] : 0,
        /*.sd_nb1 =*/ fold_dst ? fold_dst->nb[1] : 0,
        /*.eps_q  =*/ use_l2q ? l2q.eps : 0.0f,
        /*.eps_k  =*/ use_l2k ? l2k.eps : 0.0f,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                  ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(t_q),        ida++); // q
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(t_k),        ida++); // k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++); // v
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++); // gate
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++); // beta
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(rows_src ? rows_src->src[0] : op->src[5]), ida++); // state
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++); // dst
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(rows_src ? rows_src->src[1] : op), ida++); // state rows
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(fold_dst ? fold_dst : op),          ida++); // new state
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(use_raw ? op->src[7] : op), ida++); // dt bias
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(use_raw ? op->src[8] : op), ida++); // a

    const int nsg = pipeline.nsg;

    if (nsg < 0) {
        // one flat group of 128 threads covers 128/lpr rows
        const int lpr = -nsg;
        ggml_metal_encoder_dispatch_threadgroups(enc, op->src[2]->ne[0]/(128/lpr), op->src[2]->ne[1], op->src[2]->ne[3], 128, 1, 1);
    } else {
        ggml_metal_encoder_dispatch_threadgroups(enc, op->src[2]->ne[0]/nsg, op->src[2]->ne[1], op->src[2]->ne[3], 32, nsg, 1);
    }

    return 1;
}

int ggml_metal_op_turbo_wht(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    int direction;
    memcpy(&direction, op->op_params, sizeof(int));

    const int64_t n_elements = ggml_nelements(op->src[0]);
    const int64_t n_groups = n_elements / 128;

    auto pipeline = ggml_metal_library_get_pipeline_turbo_wht(lib);

    ggml_metal_kargs_turbo_wht args = {
        /*.n_elements =*/ n_elements,
        /*.direction  =*/ direction,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    const int threads_per_tg = 256;
    const int n_threadgroups = (n_groups + threads_per_tg - 1) / threads_per_tg;
    ggml_metal_encoder_dispatch_threadgroups(enc, n_threadgroups, 1, 1, threads_per_tg, 1, 1);

    return 1;
}

int ggml_metal_op_solve_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_solve_tri args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_solve_tri(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nsg = pipeline.nsg;
    const int nw  = ggml_metal_device_get_props(ctx->dev)->simd_width;

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne10 + nsg - 1)/nsg, ne02, ne03, nw, nsg, 1);

    return 1;
}

int ggml_metal_op_set(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type, false);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[1]->type, op->type, false);

    GGML_ASSERT(ne10 % ggml_blck_size(op->src[1]->type) == 0);

    int64_t nk0 = ne10;
    if (ggml_is_quantized(op->src[1]->type)) {
        nk0 = ne10/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne10/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0*ne11, 256);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[1]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb10,
        /*.nb01 =*/ nb11,
        /*.nb02 =*/ nb12,
        /*.nb03 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ ggml_element_size(op),
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    bid_dst.offs += offs;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne11 + nrptg - 1)/nrptg, ne12, ne13, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_cpy(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const bool contig = ggml_blck_size(op->src[0]->type) == 1 && ggml_blck_size(op->type) == 1 &&
                        ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op) &&
                        getenv("TOSH_CPY_CONTIG_DISABLE") == nullptr;

    if (contig) {
        const int64_t n = ggml_nelements(op);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ n,
            /*.ne00 =*/ ne00, /*.ne01 =*/ ne01, /*.ne02 =*/ ne02, /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00, /*.nb01 =*/ nb01, /*.nb02 =*/ nb02, /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,  /*.ne1  =*/ ne1,  /*.ne2  =*/ ne2,  /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,  /*.nb1  =*/ nb1,  /*.nb2  =*/ nb2,  /*.nb3  =*/ nb3,
        };

        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type, true);

        const int nth = std::min<int>(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, (n + nth - 1)/nth, 1, 1, nth, 1, 1);

        return 1;
    }

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type, false);

    GGML_ASSERT(ne00 % ggml_blck_size(op->src[0]->type) == 0);

    const auto is_turbo = [](ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
    };

    int64_t nk0 = ne00;
    if (is_turbo(op->src[0]->type)) {
        // turbo dequant does the inverse WHT over the whole 128-element group per thread
        nk0 = ne00/ggml_blck_size(op->src[0]->type);
    } else if (ggml_is_quantized(op->src[0]->type)) {
        nk0 = ne00/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne00/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0*ne01, 256);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[0]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_pool_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t s0 = opts[2];
    const int32_t p0 = opts[3];

    const int64_t IW = op->src[0]->ne[0];
    const int64_t OW = op->ne[0];

    const int64_t np = ggml_nelements(op);

    ggml_metal_kargs_pool_1d args_pool_1d = {
        /* .k0 = */  k0,
        /* .s0 = */  s0,
        /* .p0 = */  p0,
        /* .IW = */  IW,
        /* .OW = */  OW,
        /* .np = */  np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_1d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_1d, sizeof(args_pool_1d),  0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

// supported FWHT sizes, must stay in sync with the
// kernel_fwht_f32_<N> templates in ggml-metal.metal
static bool ggml_metal_fwht_supported_size(int64_t n) {
    return n == 64 || n == 128 || n == 256 || n == 512 || n == 1024;
}

int ggml_metal_op_fwht(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ggml_tensor * src1 = op->src[1];

    const int64_t n = src1->ne[0];
    const int64_t nrows = ggml_nrows(src1);

    ggml_metal_kargs_fwht args = {
        /*.nrows = */ (int32_t) nrows,
    };

    const int simd_size = ggml_metal_device_get_props(ctx->dev)->simd_width;

    auto pipeline = ggml_metal_library_get_pipeline_fwht(lib, n, simd_size == 64);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src1), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 2);

    const int th_max = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);

    int sg_per_tg = 2;
    sg_per_tg = std::min(sg_per_tg, th_max/simd_size);
    sg_per_tg = std::max(sg_per_tg, 1);

    const int64_t n_tg = (nrows + sg_per_tg - 1) / sg_per_tg;
    ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, simd_size*sg_per_tg, 1, 1);

    return 1;
}

static int32_t ggml_metal_use_count(const ggml_cgraph * gf, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&gf->visited_hash_set, t);
    if (!ggml_bitset_get(gf->visited_hash_set.used, pos)) {
        return 0;
    }
    return gf->use_counts[pos];
}

// the tensor under a chain of single-use reshapes, or nullptr if any of them is read elsewhere
static const ggml_tensor * ggml_metal_unwrap_reshape(const ggml_cgraph * gf, const ggml_tensor * t) {
    while (t && t->op == GGML_OP_RESHAPE) {
        if (ggml_metal_use_count(gf, t) != 1) {
            return nullptr;
        }
        t = t->src[0];
    }
    return t;
}

// rms_norm -> mul by the norm weight -> the Hadamard transform of a folded weight's input, all
// adjacent: one kernel writes the transform, and the norm too when other nodes read it
static int ggml_metal_op_rms_norm_fwht(ggml_metal_op_t ctx, int idx) {
    static const bool disabled = getenv("TOSH_NORM_FWHT_FUSION_DISABLE") != nullptr;
    if (disabled || idx + 3 >= ctx->n_nodes()) {
        return 0;
    }
    const ggml_cgraph * gf = ctx->graph();

    // an add before the norm (the residual) is taken in as well
    const ggml_tensor * add = ctx->node(idx)->op == GGML_OP_ADD ? ctx->node(idx) : nullptr;
    const int i0 = add ? idx + 1 : idx;
    if (i0 + 3 >= ctx->n_nodes()) {
        return 0;
    }

    const ggml_tensor * nrm = ctx->node(i0);
    const ggml_tensor * mw  = ctx->node(i0 + 1);
    const ggml_tensor * ms  = ctx->node(i0 + 2);
    const ggml_tensor * mm  = ctx->node(i0 + 3);

    if (nrm->op != GGML_OP_RMS_NORM || mw->op != GGML_OP_MUL || ms->op != GGML_OP_MUL || mm->op != GGML_OP_MUL_MAT ||
        ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD || mm->src[1]->ne[0] != 1024) {
        return 0;
    }
    const ggml_tensor * x = nrm->src[0];
    const int64_t ne00 = x->ne[0];
    const ggml_tensor * w     = mw->src[0] == nrm ? mw->src[1] : mw->src[0];
    const ggml_tensor * signs = ms->src[1];

    const bool ok =
        x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) && ne00 % 1024 == 0 && ne00/1024 <= 8 &&
        x->ne[2] == 1 && x->ne[3] == 1 && ggml_are_same_shape(nrm, x) &&
        (mw->src[0] == nrm || mw->src[1] == nrm) && ggml_are_same_shape(mw, nrm) && mw->type == GGML_TYPE_F32 &&
        w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && ggml_nelements(w) == ne00 &&
        ms->src[0] == mw && ggml_are_same_shape(ms, mw) && ms->type == GGML_TYPE_F32 &&
        signs->type == GGML_TYPE_F32 && ggml_is_contiguous(signs) && ggml_nelements(signs) == ne00 &&
        ggml_is_contiguous(mw) && ggml_is_contiguous(mm) && mm->type == GGML_TYPE_F32 &&
        ggml_metal_use_count(gf, nrm) == 1 && ggml_metal_use_count(gf, ms) == 1 &&
        !(nrm->flags & GGML_TENSOR_FLAG_OUTPUT) && !(ms->flags & GGML_TENSOR_FLAG_OUTPUT) &&
        ggml_metal_unwrap_reshape(gf, mm->src[1]) == ms &&
        !ggml_metal_ranges_overlap(mm, mw) && !ggml_metal_ranges_overlap(mm, w) && !ggml_metal_ranges_overlap(mm, signs);
    if (!ok) {
        return 0;
    }
    const ggml_tensor * a1 = nullptr;
    if (add) {
        const ggml_tensor * a0 = add->src[0];
        a1 = add->src[1];
        if (x != add || a0->type != GGML_TYPE_F32 || a1->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(a0, add) || !ggml_are_same_shape(a1, add) ||
            a0->nb[0] != sizeof(float) || a1->nb[0] != sizeof(float) || add->nb[0] != sizeof(float) ||
            ggml_metal_ranges_overlap(mm, a0) || ggml_metal_ranges_overlap(mm, a1) || ggml_metal_ranges_overlap(mm, add) ||
            ggml_metal_ranges_overlap(mw, a0) || ggml_metal_ranges_overlap(mw, a1)) {
            return 0;
        }
        x = a0;
    }

    const bool write_norm = ggml_metal_use_count(gf, mw) > 1 || (mw->flags & GGML_TENSOR_FLAG_OUTPUT);
    const int simd_size = ggml_metal_device_get_props(ctx->dev)->simd_width;

    const char * base = simd_size == 64 ? "kernel_rms_norm_fwht_f32_w64" : "kernel_rms_norm_fwht_f32";
    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(ctx->lib, base);
    if (!pipeline.pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(ctx->lib, base, base, nullptr);
    }

    ggml_metal_kargs_rms_norm_fwht args = {
        /*.ne00       =*/ (int32_t) ne00,
        /*.eps        =*/ ggml_get_op_params_f32(nrm, 0),
        /*.nb01       =*/ x->nb[1],
        /*.nb1n       =*/ mw->nb[1],
        /*.nb1h       =*/ (uint64_t) ne00*sizeof(float),
        /*.write_norm =*/ write_norm,
        /*.add        =*/ add != nullptr,
        /*.nb11       =*/ a1 ? a1->nb[1] : 0,
        /*.nb1a       =*/ add ? add->nb[1] : 0,
    };

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),     1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(w),     2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(signs), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mw),    4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mm),    5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(a1 ? a1 : x),   6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(add ? add : mw), 7);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, (64 + ne00)*sizeof(float), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, x->ne[1], 1, 1, 128*(ne00/1024), 1, 1);

    return add ? 5 : 4;
}

// [CONT of a permuted view] -> MUL by a sign vector -> MUL_MAT with the Hadamard hint, as the
// graph builds the activation transform of a Hadamard-folded weight: one kernel instead of three
int ggml_metal_op_fwht_fused(ggml_metal_op_t ctx, int idx) {
    const ggml_cgraph * gf = ctx->graph();

    const ggml_tensor * cont = nullptr;
    int i = idx;
    if (ctx->node(i)->op == GGML_OP_CONT) {
        cont = ctx->node(i);
        i++;
    }
    if (i + 1 >= ctx->n_nodes()) {
        return 0;
    }

    const ggml_tensor * mul = ctx->node(i);
    const ggml_tensor * mm  = ctx->node(i + 1);

    if (mul->op != GGML_OP_MUL || mm->op != GGML_OP_MUL_MAT ||
        ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        return 0;
    }

    const ggml_tensor * x     = mul->src[0];
    const ggml_tensor * signs = mul->src[1];
    const int64_t n = mm->src[1]->ne[0];

    const bool shapes_ok =
        mul->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && signs->type == GGML_TYPE_F32 &&
        mm->type == GGML_TYPE_F32 && ggml_are_same_shape(mul, x) &&
        ggml_is_contiguous(signs) && ggml_nelements(signs) == x->ne[0] && signs->ne[0] == x->ne[0] &&
        x->ne[0] % n == 0 && (n == 512 || n == 1024) &&
        ggml_is_contiguous(mm) && ggml_are_same_shape(mm, mm->src[1]);
    if (!shapes_ok) {
        return 0;
    }

    // every intermediate must be read only by the next step, since none of them gets written
    if (ggml_metal_use_count(gf, mul) != 1 || (mul->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        ggml_metal_unwrap_reshape(gf, mm->src[1]) != mul) {
        return 0;
    }

    // with the copy, only the regrouping [hd, nk, rep] -> [hd, rep, nk] of a contiguous tensor
    const ggml_tensor * src = x;
    int hd = 0, rep = 0, nk = 0;
    if (cont) {
        const ggml_tensor * p = cont->src[0];
        if (ggml_metal_use_count(gf, cont) != 1 || (cont->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_metal_unwrap_reshape(gf, x) != cont || cont->type != GGML_TYPE_F32 ||
            p->op != GGML_OP_PERMUTE || p->type != GGML_TYPE_F32 || !ggml_is_contiguous(p->src[0])) {
            return 0;
        }
        hd  = (int) p->ne[0];
        rep = (int) p->ne[1];
        nk  = (int) p->ne[2];
        const size_t es = sizeof(float);
        const int64_t W = (int64_t) hd*rep*nk;
        if (p->nb[0] != es || p->nb[1] != es*hd*nk || p->nb[2] != es*hd || p->nb[3] != es*W ||
            x->ne[0] != W || W % n != 0 || hd > INT16_MAX || rep > INT16_MAX || nk > INT16_MAX) {
            return 0;
        }
        src = p;
    } else if (!ggml_is_contiguous(x)) {
        return 0;
    }

    const int64_t nrows = ggml_nelements(mm) / n;

    ggml_metal_kargs_fwht_fused args = {
        /*.nrows =*/ (int32_t) nrows,
        /*.nsign =*/ (int32_t) signs->ne[0],
    };

    const int simd_size = ggml_metal_device_get_props(ctx->dev)->simd_width;

    auto pipeline = ggml_metal_library_get_pipeline_fwht_fused(ctx->lib, (int) n, hd, rep, nk, simd_size == 64);

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src),   1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(signs), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mm),    3);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, n*sizeof(float), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 128, 1, 1);

    return i + 2 - idx;
}

int ggml_metal_op_pool_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t k1 = opts[2];
    const int32_t s0 = opts[3];
    const int32_t s1 = opts[4];
    const int32_t p0 = opts[5];
    const int32_t p1 = opts[6];

    const int64_t IH = op->src[0]->ne[1];
    const int64_t IW = op->src[0]->ne[0];

    const int64_t N  = op->ne[3];
    const int64_t OC = op->ne[2];
    const int64_t OH = op->ne[1];
    const int64_t OW = op->ne[0];

    const int64_t np = N * OC * OH * OW;

    ggml_metal_kargs_pool_2d args_pool_2d = {
        /* .k0 = */ k0,
        /* .k1 = */ k1,
        /* .s0 = */ s0,
        /* .s1 = */ s1,
        /* .p0 = */ p0,
        /* .p1 = */ p1,
        /* .IH = */ IH,
        /* .IW = */ IW,
        /* .OH = */ OH,
        /* .OW = */ OW,
        /* .np = */ np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_2d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_2d, sizeof(args_pool_2d), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

static int ggml_metal_op_mul_mat_qvk(ggml_metal_op_t ctx, int idx) {
    static const bool disabled = getenv("TOSH_QKV_FUSION_DISABLE") != nullptr;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);
    const bool width_ok = props_dev->simd_width == 32 ||
                          (props_dev->simd_width == 64 && props_dev->wave64_decode);
    if (disabled || !ctx->use_fusion() || !width_ok || !props_dev->use_mm_manual || idx + 2 >= ctx->n_nodes()) {
        return 0;
    }

    // Identify the three consecutive MUL_MAT nodes by weight-name suffix, not
    // position: llama.cpp's build_qkv() emits Q,K,V today, but no permutation
    // is guaranteed, so the match must not depend on the emission order.
    // Ends-with (not substring) so names like "*.attn_q.weight.extra" fail closed.
    auto has_suffix = [](const char * name, const char * suffix) {
        const size_t n = strlen(name), s = strlen(suffix);
        return n >= s && strcmp(name + n - s, suffix) == 0;
    };
    ggml_tensor * q = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * k = nullptr;
    for (int i = 0; i < 3; i++) {
        ggml_tensor * n = ctx->node(idx + i);
        if (n->op != GGML_OP_MUL_MAT) {
            return 0;
        }
        const char * wname = ggml_get_name(n->src[0]);
        if (has_suffix(wname, ".attn_q.weight")) {
            q = n;
        } else if (has_suffix(wname, ".attn_v.weight")) {
            v = n;
        } else if (has_suffix(wname, ".attn_k.weight")) {
            k = n;
        }
    }
    if (!q || !v || !k) {
        return 0;
    }

    // Avoid false-positive graph fusion: these must be the three independent
    // decode projections over exactly the same activation. The quant pattern is
    // selected separately below so every segment keeps its GGUF representation.
    // Single-token, single-plane decode only: the pipeline constants hardcode
    // ne12 = r2 = r3 = 1, so batched planes fall back to the unfused path (q, v,
    // k share src[1], checked below, so guarding one covers all three).
    if ((q->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        (v->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        (k->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        q->src[1] != v->src[1] || q->src[1] != k->src[1] ||
        q->src[1]->type != GGML_TYPE_F32 ||
        q->src[1]->ne[1] != 1 || v->src[1]->ne[1] != 1 || k->src[1]->ne[1] != 1 ||
        q->src[1]->ne[2] != 1 || q->src[1]->ne[3] != 1 ||
        q->src[0]->ne[0] != q->src[1]->ne[0] ||
        v->src[0]->ne[0] != v->src[1]->ne[0] ||
        k->src[0]->ne[0] != k->src[1]->ne[0] ||
        q->src[0]->ne[2] != 1 || q->src[0]->ne[3] != 1 ||
        v->src[0]->ne[2] != 1 || v->src[0]->ne[3] != 1 ||
        k->src[0]->ne[2] != 1 || k->src[0]->ne[3] != 1) {
        return 0;
    }

    struct qvk_pattern {
        const char * kernel;
        uint32_t nsg;
        uint32_t nr_q;
        uint32_t nr_v;
        uint32_t nr_k;
    } pattern = {};

    const ggml_type tq = q->src[0]->type;
    const ggml_type tv = v->src[0]->type;
    const ggml_type tk = k->src[0]->type;

    const bool qvk_w64 = props_dev->simd_width == 64 && getenv("TOSH_QVK_W64_DISABLE") == nullptr;

    if (tq == GGML_TYPE_Q4_0 && tv == GGML_TYPE_Q4_0 && tk == GGML_TYPE_Q4_0) {
        // the 64-lane twin carries the same decomposition as the standalone matvec, so the
        // fusion keeps that win instead of trading it for the saved dispatches
        pattern = qvk_w64 ? decltype(pattern){ "kernel_mul_mv_qvk_q4_0_q4_0_q4_0_f32_w64", 1, 4, 4, 4 }
                          : decltype(pattern){ "kernel_mul_mv_qvk_q4_0_q4_0_q4_0_f32",     1, 4, 4, 4 };
    } else if (tq == GGML_TYPE_Q2_K && tv == GGML_TYPE_Q4_K && tk == GGML_TYPE_Q2_K) {
        pattern = { "kernel_mul_mv_qvk_q2_K_q4_K_q2_K_f32", 2, 4, 2, 4 };
    } else if (tq == GGML_TYPE_Q4_K && tv == GGML_TYPE_Q6_K && tk == GGML_TYPE_Q4_K) {
        pattern = { "kernel_mul_mv_qvk_q4_K_q6_K_q4_K_f32", 2, 2, 2, 4 };
    } else {
        return 0;
    }

    auto make_args = [](const ggml_tensor * op, int32_t nr0) {
        const ggml_tensor * w = op->src[0];
        const ggml_tensor * x = op->src[1];

        GGML_ASSERT(x->ne[2] % w->ne[2] == 0);
        GGML_ASSERT(x->ne[3] % w->ne[3] == 0);

        return ggml_metal_kargs_mul_mv {
            /*.ne00 =*/ (int32_t) w->ne[0],
            /*.ne01 =*/ (int32_t) w->ne[1],
            /*.ne02 =*/ (int32_t) w->ne[2],
            /*.nb00 =*/ w->nb[0],
            /*.nb01 =*/ w->nb[1],
            /*.nb02 =*/ w->nb[2],
            /*.nb03 =*/ w->nb[3],
            /*.ne10 =*/ (int32_t) x->ne[0],
            /*.ne11 =*/ (int32_t) x->ne[1],
            /*.ne12 =*/ (int32_t) x->ne[2],
            /*.nb10 =*/ x->nb[0],
            /*.nb11 =*/ x->nb[1],
            /*.nb12 =*/ x->nb[2],
            /*.nb13 =*/ x->nb[3],
            /*.ne0  =*/ (int32_t) op->ne[0],
            /*.ne1  =*/ (int32_t) op->ne[1],
            /*.nr0  =*/ nr0,
            /*.r2   =*/ (int16_t) (x->ne[2]/w->ne[2]),
            /*.r3   =*/ (int16_t) (x->ne[3]/w->ne[3]),
        };
    };

    ggml_metal_kargs_mul_mv_qvk args = {
        /*.q     =*/ make_args(q, pattern.nr_q),
        /*.v     =*/ make_args(v, pattern.nr_v),
        /*.k     =*/ make_args(k, pattern.nr_k),
        /*.nwg_q =*/ (uint32_t) ((q->src[0]->ne[1] + pattern.nr_q*pattern.nsg - 1)/(pattern.nr_q*pattern.nsg)),
        /*.nwg_v =*/ (uint32_t) ((v->src[0]->ne[1] + pattern.nr_v*pattern.nsg - 1)/(pattern.nr_v*pattern.nsg)),
    };
    const uint32_t nwg_k = (uint32_t) ((k->src[0]->ne[1] + pattern.nr_k*pattern.nsg - 1)/(pattern.nr_k*pattern.nsg));

    char name[256];
    const int16_t nw = (int16_t) props_dev->simd_width;
    snprintf(name, sizeof(name), "%s_nsg=%u_ne12=1_r2=1_r3=1_nw=%d", pattern.kernel, pattern.nsg, nw);
    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(ctx->lib, name);
    if (!pipeline.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, pattern.nsg, FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 4);
        ggml_metal_cv_set_int16(cv, nw,  FC_MUL_MV + 5);
        pipeline = ggml_metal_library_compile_pipeline(ctx->lib, pattern.kernel, name, cv);
        ggml_metal_cv_free(cv);
    }

    if (!pipeline.pipeline || ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) < nw*(int) pattern.nsg) {
        return 0;
    }

    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes   (ctx->enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(q->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(v->src[0]), 2);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(k->src[0]), 3);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(q->src[1]), 4);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(q),         5);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(v),         6);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(k),         7);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, args.nwg_q + args.nwg_v + nwg_k, 1, 1, nw, pattern.nsg, 1);

    return 3;
}

// K-quants asked for four columns while every other type batches from two, so at two or three
// sequences each column paid its own full pass over the weights
static int kquant_ext_floor() {
    static const int v = []() {
        const char * s = getenv("TOSH_EXT_KQ_FLOOR");
        return s ? atoi(s) : 2;
    }();
    return v;
}

// nine and ten columns only beat the tile with five per group, which needs the two-row variant
// and a cheap dequant: q5_K measures flat there and q6_K loses
static bool ext_wide_group(ggml_type tsrc0) {
    return tsrc0 == GGML_TYPE_Q4_K || tsrc0 == GGML_TYPE_Q2_K || tsrc0 == GGML_TYPE_Q3_K;
}

// above eight columns the matmul tile takes over, but on 64 lanes it costs the same from nine
// columns to thirty-two, so three column groups of the batched kernel are still cheaper
static int ext_ceil(int simd_width, ggml_type tsrc0) {
    static const int v = []() {
        const char * s = getenv("TOSH_EXT_MAX");
        return s ? atoi(s) : 0;
    }();

    if (v) {
        return v;
    }

    if (simd_width == 64) {
        return 12;
    }

    return ext_wide_group(tsrc0) ? 10 : 8;
}

static bool ggml_metal_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// PQ2_0 gate and up mat-vecs of one token followed by their swiglu: one kernel writes the swiglu
// output. The three nodes are adjacent, so writing that output early only needs it apart from
// the activation the kernel reads.
static int ggml_metal_op_mul_mat_swiglu(ggml_metal_op_t ctx, int idx) {
    static const bool disabled = getenv("TOSH_SWIGLU_FUSION_DISABLE") != nullptr;
    if (disabled || !ctx->use_fusion() || idx + 2 >= ctx->n_nodes()) {
        return 0;
    }

    const ggml_tensor * a   = ctx->node(idx);
    const ggml_tensor * b   = ctx->node(idx + 1);
    const ggml_tensor * glu = ctx->node(idx + 2);

    if (a->op != GGML_OP_MUL_MAT || b->op != GGML_OP_MUL_MAT || glu->op != GGML_OP_GLU ||
        ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 || !glu->src[1]) {
        return 0;
    }

    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    if (!((gate == a && up == b) || (gate == b && up == a))) {
        return 0;
    }

    const ggml_cgraph * gf = ctx->graph();
    const ggml_tensor * x  = gate->src[1];

    const bool ok =
        gate->src[0]->type == GGML_TYPE_PQ2_0 && up->src[0]->type == GGML_TYPE_PQ2_0 &&
        ggml_are_same_shape(gate->src[0], up->src[0]) && up->src[1] == x &&
        ggml_get_op_params_i32(gate, 1) == 0 && ggml_get_op_params_i32(up, 1) == 0 &&
        gate->src[0]->ne[2] == 1 && gate->src[0]->ne[3] == 1 &&
        ggml_is_contiguous(gate->src[0]) && ggml_is_contiguous(up->src[0]) &&
        x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && x->ne[1] == 1 && x->ne[2] == 1 && x->ne[3] == 1 &&
        gate->src[0]->ne[0] % 128 == 0 &&
        glu->type == GGML_TYPE_F32 && ggml_is_contiguous(glu) && ggml_nelements(glu) == gate->ne[0] &&
        ggml_metal_use_count(gf, gate) == 1 && ggml_metal_use_count(gf, up) == 1 &&
        !(gate->flags & GGML_TENSOR_FLAG_OUTPUT) && !(up->flags & GGML_TENSOR_FLAG_OUTPUT) &&
        !ggml_metal_ranges_overlap(glu, x);
    if (!ok) {
        return 0;
    }

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_pq2_swiglu(ctx->lib);

    const ggml_tensor * w = gate->src[0];

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ (int32_t) w->ne[0],
        /*.ne01 =*/ (int32_t) w->ne[1],
        /*.ne02 =*/ 1,
        /*.nb00 =*/ w->nb[0],
        /*.nb01 =*/ w->nb[1],
        /*.nb02 =*/ w->nb[2],
        /*.nb03 =*/ w->nb[3],
        /*.ne10 =*/ (int32_t) x->ne[0],
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ x->nb[0],
        /*.nb11 =*/ x->nb[1],
        /*.nb12 =*/ x->nb[2],
        /*.nb13 =*/ x->nb[3],
        /*.ne0  =*/ (int32_t) gate->ne[0],
        /*.ne1  =*/ 1,
        /*.nr0  =*/ pipeline.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(gate->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(up->src[0]),   2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),            3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(glu),          4);

    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;
    ggml_metal_encoder_dispatch_threadgroups(enc, (w->ne[1] + nr0*nsg - 1)/(nr0*nsg), 1, 1, pipeline.nr1, nsg, 1);

    return 3;
}

int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    if (ggml_metal_op_mul_mat_use_fwht(op)) {
        return ggml_metal_op_fwht(ctx, idx);
    }
    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    if (const int n_fuse = ggml_metal_op_mul_mat_qvk(ctx, idx)) {
        return n_fuse;
    }

    if (const int n_fuse = ggml_metal_op_mul_mat_swiglu(ctx, idx)) {
        return n_fuse;
    }

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne00 == ne10);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    // ext kernel derives its row split from the simdgroup width, so it runs on both
    // wave32 and wave64. wave64 rides on wave64_decode (matches the mat-vec gating).
    static const bool ext_disabled = getenv("TOSH_EXT_DISABLE") != nullptr;
    if (!ext_disabled &&
        (props_dev->simd_width == 32 || (props_dev->simd_width == 64 && props_dev->wave64_decode)) &&
        op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
        (
         (
          (
           op->src[0]->type == GGML_TYPE_F32  || // TODO: helper function
           op->src[0]->type == GGML_TYPE_F16  ||
           op->src[0]->type == GGML_TYPE_BF16 ||
           op->src[0]->type == GGML_TYPE_Q1_0 ||
           op->src[0]->type == GGML_TYPE_Q2_0 ||
           op->src[0]->type == GGML_TYPE_Q4_0 ||
           op->src[0]->type == GGML_TYPE_Q4_1 ||
           op->src[0]->type == GGML_TYPE_Q5_0 ||
           op->src[0]->type == GGML_TYPE_Q5_1 ||
           op->src[0]->type == GGML_TYPE_Q8_0 ||
           op->src[0]->type == GGML_TYPE_MXFP4 ||
           op->src[0]->type == GGML_TYPE_IQ4_NL ||
           false) && (ne11 >= 2 && ne11 <= ext_ceil(props_dev->simd_width, op->src[0]->type))
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           op->src[0]->type == GGML_TYPE_Q2_K ||
           op->src[0]->type == GGML_TYPE_Q3_K ||
           false) && (ne11 >= kquant_ext_floor() && ne11 <= ext_ceil(props_dev->simd_width, op->src[0]->type))
         )
        )
       ) {
        // TODO: determine the optimal parameters based on grid utilization
        //       I still don't know why we should not always use the maximum available threads:
        //
        //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
        //
        //       my current hypothesis is that the work grid is not evenly divisible for different nsg
        //       values and there can be some tail effects when nsg is high. need to confirm this
        //
        const int nsg    = 1;                 // num simdgroups per threadgroup

        // num threads along row per simdgroup
        // the thresholds below split a 32-lane simdgroup; TOSH_EXT_NXPSG overrides them so the
        // split can be swept on a 64-lane one, where the row/column balance is not the same
        static const int nxpsg_env = []() {
            const char * s = getenv("TOSH_EXT_NXPSG");
            return s ? atoi(s) : 0;
        }();

        int16_t nxpsg = 0;
        if (nxpsg_env) {
            nxpsg = (int16_t) nxpsg_env;
        } else {
            if (ne00 % 256 == 0 && ne11 < 3) {
                nxpsg = 16;
            } else if (ne00 % 128 == 0) {
                nxpsg = 8;
            } else {
                nxpsg = 4;
            }
            // a 64-lane simdgroup wants twice the reduction width: +9.0% with four requests.
            // TOSH_EXT_NXPSG_X2 tries it on 32 lanes, so that width can be A/B'd before the gate goes
            // only where the contracted dimension is wide enough to keep four-wide loads per thread:
            // at ne00 128 the doubled split leaves too few elements and the result goes wrong
            if (ne00 % 256 == 0 &&
                (props_dev->simd_width == 64 || getenv("TOSH_EXT_NXPSG_X2") != nullptr) &&
                getenv("TOSH_EXT_NXPSG_X2_DISABLE") == nullptr) {
                nxpsg = std::min<int16_t>(16, nxpsg*2);
            }
        }

        const int16_t nypsg  = props_dev->simd_width/nxpsg;   // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
              int16_t r1ptg  = 4;                 // num src1 rows per threadgroup

        // note: not sure how optimal are those across all different hardware. there might be something cleverer
        switch (ne11) {
            case 2:
                r1ptg = 2; break;
            case 3:
            case 6:
                r1ptg = 3; break;
            case 4:
            case 7:
            case 8:
                r1ptg = 4; break;
            case 5:
                r1ptg = 5; break;
            default:
                r1ptg = 4; break;
        };

        // 2 rows per thread: the activation chunk is loaded once and reused across them
        const bool nr0_2 = op->src[0]->type == GGML_TYPE_Q4_K ||
                           op->src[0]->type == GGML_TYPE_Q5_K ||
                           op->src[0]->type == GGML_TYPE_Q6_K ||
                           op->src[0]->type == GGML_TYPE_Q2_K ||
                           op->src[0]->type == GGML_TYPE_Q3_K;

        static const int nr0_env = []() {
            const char * s = getenv("TOSH_EXT_NR0");
            return s ? atoi(s) : 0;
        }();

        // only the K-quants have a two-row variant; forcing the override on the rest asks the
        // library for a kernel that was never instantiated
        // Four rows per thread doubles what a threadgroup covers, so a weight with few
        // rows stops filling the grid. The threshold is per row count, not per quant type.
        static const int64_t nr0_min_rows = []() {
            const char * s = getenv("TOSH_EXT_NR0_MINROWS");
            // off by default: the win depends on the weight's shape and one measured model
            // still regresses at four columns, so it stays opt-in until that is understood
            return s ? (int64_t) atoll(s) : (int64_t) 0;
        }();
        const bool nr0_wide = nr0_min_rows > 0 && props_dev->simd_width == 32 &&
                              op->src[0]->type == GGML_TYPE_Q4_K && ne01 >= nr0_min_rows;

        const int16_t nr0   = nr0_2 ? (nr0_env ? nr0_env : (nr0_wide ? 4 : 2)) : 1;

        // each group makes a full pass over the weights, so the cost is groups by group width:
        // nine and ten columns fit in two groups of five instead of three of four
        if (ext_wide_group(op->src[0]->type) && (ne11 == 9 || ne11 == 10)) {
            r1ptg = 5;
        }

        // the wave64 q6_K variant keeps four rows per simdgroup, so the row split does not apply
        // keep the gate in sync with the pipeline picker in ggml_metal_library_get_pipeline_mul_mv_ext
        static const bool ext_w64_disable = getenv("TOSH_EXT_W64_DISABLE") != nullptr;
        const bool ext_w64_q6k = !ext_w64_disable &&
                                 props_dev->simd_width == 64 && op->src[0]->type == GGML_TYPE_Q6_K;

        // from six src1 rows the eight-column variant reads the weights once instead of twice
        if (ext_w64_q6k && ne11 >= 6) {
            r1ptg = 8;
        }

        const int16_t r0ptg = ext_w64_q6k ? (r1ptg == 8 ? 2 : 4)*nsg : nypsg*nsg*nr0; // num src0 rows per threadgroup

        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_ext(lib, op, nsg, nxpsg, r1ptg, nr0);

        ggml_metal_kargs_mul_mv_ext args = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + r0ptg - 1)/r0ptg), ((ne11 + r1ptg - 1)/r1ptg), ne12*ne13, props_dev->simd_width, nsg, 1);
    // the manual tiled kernels give AMD the same matrix path Apple gets from simdgroup_mm
    } else if (ggml_metal_op_mul_mat_use_mm(op, props_dev->has_simdgroup_mm || props_dev->use_mm_manual)) {
        //GGML_LOG_INFO("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        auto pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, op);

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        const size_t smem = pipeline.smem;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const int sgw = pipeline.sgw ? pipeline.sgw : 32;

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + nr1 - 1) / nr1), ((ne01 + nr0 - 1) / nr0), ne12 * ne13, sgw, nsg, 1);
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const size_t smem = pipeline.smem;

        ggml_metal_kargs_mul_mv args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nr0  =*/ nr0,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        // wave64 decode dispatches a full 64-wide threadgroup so the kernel's lane
        // partitioning and reduction match the hardware simdgroup; 32 everywhere else.
        int nth = 32;
        if (props_dev->wave64_decode &&
            (op->src[0]->type == GGML_TYPE_F16 ||
             op->src[0]->type == GGML_TYPE_F32 ||
             op->src[0]->type == GGML_TYPE_BF16 ||
             op->src[0]->type == GGML_TYPE_Q8_0 ||
             op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q5_K ||
             op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_Q2_K ||
             op->src[0]->type == GGML_TYPE_Q3_K ||
             op->src[0]->type == GGML_TYPE_MXFP4 ||
             op->src[0]->type == GGML_TYPE_IQ2_XXS ||
             op->src[0]->type == GGML_TYPE_IQ2_XS ||
             op->src[0]->type == GGML_TYPE_IQ3_XXS ||
             op->src[0]->type == GGML_TYPE_IQ3_S ||
             op->src[0]->type == GGML_TYPE_IQ2_S ||
             op->src[0]->type == GGML_TYPE_IQ1_S ||
             op->src[0]->type == GGML_TYPE_IQ1_M ||
             op->src[0]->type == GGML_TYPE_IQ4_NL ||
             op->src[0]->type == GGML_TYPE_IQ4_XS ||
             op->src[0]->type == GGML_TYPE_Q4_0 ||
             op->src[0]->type == GGML_TYPE_Q4_1 ||
             op->src[0]->type == GGML_TYPE_Q5_0 ||
             op->src[0]->type == GGML_TYPE_Q5_1 ||
             op->src[0]->type == GGML_TYPE_Q1_0 ||
             op->src[0]->type == GGML_TYPE_Q2_0 ||
             op->src[0]->type == GGML_TYPE_PQ2_0 ||
             op->src[0]->type == GGML_TYPE_PTQ1_0)) {
            nth = props_dev->simd_width;
        }

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/(nr0)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, nth, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, nth, nsg, 1);
        }
    }

    return 1;
}

size_t ggml_metal_op_mul_mat_id_extra_tpe(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    // With a split bank src0 holds the slots, not the experts, and the id map is built over
    // the full expert count. Size this the same way as the ids region below, or map0 writes
    // past it and into the ids that follow.
    const int registered = tosh_moe_experts_of(op->src[0]);
    const int64_t ne02 = registered > 0 ? registered : op->src[0]->ne[2]; // n_expert

    return ggml_type_size(GGML_TYPE_I32)*ne02;
}

size_t ggml_metal_op_mul_mat_id_extra_ids(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int registered = tosh_moe_experts_of(op->src[0]);
    const int64_t ne02 = registered > 0 ? registered : op->src[0]->ne[2]; // n_expert
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    return ggml_type_size(GGML_TYPE_I32)*ne02*ne21;
}

size_t ggml_metal_op_mul_mat_id_extra_amax(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    GGML_UNUSED(op);

    // 2 scaling factors (8 bytes) + N_MM_NPART_AMAX per-threadgroup scales for stage-1
    return 8 + N_MM_NPART_AMAX*sizeof(float);
}

int ggml_metal_op_mul_mat_id(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // src2 = ids
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);

    GGML_ASSERT(!ggml_is_transposed(op->src[0]));
    GGML_ASSERT(!ggml_is_transposed(op->src[1]));

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;
    int32_t ne02_run = ne02;

    // A cached bank: pick the experts and pull in what is missing before the matmul reads the
    // slots. Both run on this encoder, so the routing never goes back to the host.
    // A bank handed over whole, because the batch was too wide for the cache: copy it to the
    // scratch first. Reading experts scattered from host memory is what makes prefill crawl.
    // Only banks of small rows gain: those are read scattered from host and the copy makes them
    // contiguous. A bank of few, large rows already reads well and the copy is pure extra traffic.
    static const size_t stage_row_max = []() {
        const char * s = getenv("TOSH_MOE_STAGE_ROW_KIB");
        return (size_t) (s ? atoi(s) : 1024) << 10;
    }();

    if (const ggml_tensor * stage = (const ggml_tensor *) (tosh_moe_slots_of(op->src[0]) ? tosh_moe_stage() : nullptr)) {
        if (ggml_nbytes(op->src[0]) <= ggml_nbytes(stage) && nb02 <= stage_row_max) {
            const uint32_t n_u4 = (uint32_t) (ggml_nbytes(op->src[0])/16);

            auto ps = ggml_metal_library_compile_pipeline(lib, "tosh_moe_stage", "tosh_moe_stage", nullptr);

            ggml_metal_encoder_set_pipeline(enc, ps);
            ggml_metal_encoder_set_bytes (enc, (void *) &n_u4, sizeof(n_u4), 0);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(stage), 1);
            ggml_metal_encoder_set_buffer(enc, bid_src0, 2);
            ggml_metal_encoder_dispatch_threadgroups(enc, 512, 1, 1, 256, 1, 1);

            ggml_metal_op_concurrency_reset(ctx);

            bid_src0 = ggml_metal_get_buffer_id(stage);
        }
    }

    ggml_metal_buffer_id full_cold_bid = {};
    bool has_full_cold = false;
    ggml_metal_buffer_id staged_ids_bid = {};
    bool has_staged_ids = false;
    {
        std::lock_guard<std::mutex> lock(g_tosh_moe_mutex);
        const auto it_full = g_tosh_moe_full_cold.find(op->src[0]);
        if (it_full != g_tosh_moe_full_cold.end()) {
            full_cold_bid = it_full->second;
            has_full_cold = true;
        }
        const auto it_ids = g_tosh_moe_staged_ids.find(op->src[0]);
        if (it_ids != g_tosh_moe_staged_ids.end()) {
            staged_ids_bid = it_ids->second;
            has_staged_ids = true;
        }
    }

    if (has_full_cold) {
        const ggml_tensor * stage = (const ggml_tensor *) tosh_moe_stage();
        const ggml_tensor * state = (const ggml_tensor *) tosh_moe_state_of(op->src[0]);
        const int n_expert = tosh_moe_experts_of(op->src[0]);
        const int n_slots = (int) ne02;
        const int n_fixed = tosh_moe_fixed_of(op->src[0]);
        const int max_fetch = n_slots - n_fixed;
        GGML_ASSERT(stage && state && n_expert > n_slots && nb02 % 16 == 0);
        // A batch routes to a fraction of the bank, so rebuilding every row is most of the
        // traffic of this path. Mark the routed ones and let the assemble skip the rest; a row
        // nothing routes to is never read, so leaving it stale is safe.
        static const bool compact = getenv("TOSH_MOE_COMPACT_ASSEMBLE_DISABLE") == nullptr;
        ggml_metal_buffer_id ab = ggml_metal_get_buffer_id(state);
        ab.offs += tosh_moe_off_active(n_expert, n_slots, max_fetch)*sizeof(int32_t);
        if (compact) {
            struct { int32_t n_used; int32_t n_tokens; int32_t stride; int32_t n_expert; } ma = {
                (int32_t) ne20, (int32_t) ne21, (int32_t) (nb21/sizeof(int32_t)), n_expert
            };
            auto pm = ggml_metal_library_compile_pipeline(lib, "tosh_moe_mark_active", "tosh_moe_mark_active", nullptr);
            ggml_metal_encoder_set_pipeline(enc, pm);
            ggml_metal_encoder_set_bytes (enc, &ma, sizeof(ma), 0);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 1);
            ggml_metal_encoder_set_buffer(enc, ab, 2);
            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 256, 1, 1);
            ggml_metal_op_concurrency_reset(ctx);
        }
        struct { uint32_t row_u4; uint32_t n_expert; int32_t use_active; } args = {
            (uint32_t) (nb02/16), (uint32_t) n_expert, compact ? 1 : 0
        };
        auto ps = ggml_metal_library_compile_pipeline(lib, "tosh_moe_assemble", "tosh_moe_assemble", nullptr);
        ggml_metal_buffer_id sb = ggml_metal_get_buffer_id(state);
        ggml_metal_encoder_set_pipeline(enc, ps);
        ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(stage), 1);
        ggml_metal_encoder_set_buffer(enc, full_cold_bid, 2);
        ggml_metal_encoder_set_buffer(enc, bid_src0, 3);
        sb.offs = ggml_metal_get_buffer_id(state).offs +
            tosh_moe_off_slot_for_id(n_expert, n_slots, max_fetch)*sizeof(int32_t);
        ggml_metal_encoder_set_buffer(enc, sb, 4);
        sb.offs = ggml_metal_get_buffer_id(state).offs +
            tosh_moe_off_cold_for_id(n_expert, n_slots, max_fetch)*sizeof(int32_t);
        ggml_metal_encoder_set_buffer(enc, sb, 5);
        ggml_metal_encoder_set_buffer(enc, ab, 6);
        const uint64_t total = (uint64_t) args.row_u4*args.n_expert;
        ggml_metal_encoder_dispatch_threadgroups(enc, (total + 255)/256, 1, 1, 256, 1, 1);
        ggml_metal_op_concurrency_reset(ctx);
        bid_src0 = ggml_metal_get_buffer_id(stage);
        ne02_run = n_expert;
    }

    if (has_full_cold) {
        // Full staging preserves the original global expert ids.
    } else if (has_staged_ids) {
        // The bounded host path has already copied only the selected expert rows into these
        // private slots. Its compact shared ID buffer names the physical slots directly.
        bid_src2 = staged_ids_bid;
    } else if (const ggml_tensor * state = (const ggml_tensor *) tosh_moe_state_of(op->src[0])) {
        // the graph only routes a batch here when its remapped ids fit the state
        const int registered_experts = tosh_moe_experts_of(op->src[0]);
        GGML_ASSERT(registered_experts > 0);
        GGML_ASSERT((int64_t) (ne21 - 1)*(int64_t) (nb21/sizeof(int32_t)) + ne20 <=
                tosh_moe_ids_room(registered_experts));
        const ggml_tensor * host = (const ggml_tensor *) tosh_moe_ram_of(op->src[0]);

        const int n_expert  = registered_experts;
        const int n_slots   = (int) ne02;
        const int n_fixed   = tosh_moe_fixed_of(op->src[0]);
        const int max_fetch = n_fixed > 0 ? n_slots - n_fixed : n_slots;

        const ggml_metal_buffer_id bid_state = ggml_metal_get_buffer_id(state);
        const size_t i32 = sizeof(int32_t);

        if (tosh_moe_should_route(op->src[0], idx)) {
            const int n_fixed = tosh_moe_fixed_of(op->src[0]);
            ggml_metal_kargs_moe_lru la = {
                /*.n_used   =*/ (int32_t) ne20,
                /*.n_tokens =*/ (int32_t) ne21,
                /*.stride   =*/ (int32_t) (nb21/sizeof(int32_t)),
                /*.n_expert =*/ n_expert,
                /*.n_slots  =*/ n_slots,
                /*.n_fixed  =*/ n_fixed,
                /*.max_fetch=*/ max_fetch,
            };

            auto pl = ggml_metal_library_compile_pipeline(lib, "tosh_moe_lru", "tosh_moe_lru", nullptr);

            ggml_metal_encoder_set_pipeline(enc, pl);
            ggml_metal_encoder_set_bytes (enc, &la, sizeof(la), 0);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 1);
            ggml_metal_buffer_id b = bid_state;
            b.offs = bid_state.offs + tosh_moe_off_slot_for_id(n_expert, n_slots, max_fetch)*i32;
            ggml_metal_encoder_set_buffer(enc, b, 2);
            b.offs = bid_state.offs + tosh_moe_off_id_of_slot(n_expert, n_slots, max_fetch)*i32;
            ggml_metal_encoder_set_buffer(enc, b, 3);
            b.offs = bid_state.offs + tosh_moe_off_usage(n_expert, n_slots, max_fetch)*i32;
            ggml_metal_encoder_set_buffer(enc, b, 4);
            b.offs = bid_state.offs + tosh_moe_off_step(n_expert, n_slots, max_fetch)*i32;
            ggml_metal_encoder_set_buffer(enc, b, 5);
            b.offs = bid_state.offs + tosh_moe_off_evict(n_expert, n_slots, max_fetch)*i32;
            ggml_metal_encoder_set_buffer(enc, b, 6);
            b.offs = bid_state.offs + tosh_moe_off_src(n_expert, n_slots, max_fetch)*i32;
            ggml_metal_encoder_set_buffer(enc, b, 7);
            b.offs = bid_state.offs + tosh_moe_off_n_indices(n_expert, n_slots, max_fetch)*i32;
            ggml_metal_encoder_set_buffer(enc, b, 8);
            b.offs = bid_state.offs + tosh_moe_off_ids(n_expert, n_slots, max_fetch)*i32;
            ggml_metal_encoder_set_buffer(enc, b, 9);
            b.offs = bid_state.offs + tosh_moe_off_cold_for_id(n_expert, n_slots, max_fetch)*i32;
            ggml_metal_encoder_set_buffer(enc, b, 10);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, (2*n_expert + 512)*i32, 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 256, 1, 1);

            ggml_metal_op_concurrency_reset(ctx);
        }

        // a whole expert is megabytes: one threadgroup per row leaves the device idle
        static const int n_chunks = []() {
            const char * s = getenv("TOSH_MOE_FETCH_CHUNKS");
            const int v = s ? atoi(s) : 16;
            return v > 0 ? v : 1;
        }();

        ggml_metal_kargs_moe_fetch fa = {
            /*.row_u4    =*/ (uint32_t) (nb02/16),
            /*.max_fetch =*/ max_fetch,
            /*.n_chunks  =*/ n_chunks,
        };

        auto pf = ggml_metal_library_compile_pipeline(lib, "tosh_moe_fetch", "tosh_moe_fetch", nullptr);

        ggml_metal_buffer_id b = bid_state;
        ggml_metal_encoder_set_pipeline(enc, pf);
        ggml_metal_encoder_set_bytes (enc, &fa, sizeof(fa), 0);
        ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_host_buffer_id(ctx->dev, host), 2);
        b.offs = bid_state.offs + tosh_moe_off_evict(n_expert, n_slots, max_fetch)*i32;
        ggml_metal_encoder_set_buffer(enc, b, 3);
        b.offs = bid_state.offs + tosh_moe_off_src(n_expert, n_slots, max_fetch)*i32;
        ggml_metal_encoder_set_buffer(enc, b, 4);
        b.offs = bid_state.offs + tosh_moe_off_n_indices(n_expert, n_slots, max_fetch)*i32;
        ggml_metal_encoder_set_buffer(enc, b, 5);
        ggml_metal_encoder_dispatch_threadgroups(enc, max_fetch*n_chunks, 1, 1, 256, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);

        static const bool raw_ids = getenv("TOSH_MOE_RAW_IDS") != nullptr;
        if (!raw_ids) {
            bid_src2 = bid_state;
            bid_src2.offs = bid_state.offs + tosh_moe_off_ids(n_expert, n_slots, max_fetch)*i32;
        }
    }

    // escape hatch: revert wave64 MoE prefill to the mat-vec route
    const bool mm_id_manual_ok = props_dev->simd_width == 32 ||
        getenv("TOSH_W64_MMID_PREFILL_DISABLE") == NULL;

    if (ggml_metal_op_mul_mat_id_use_mm(op, props_dev->has_simdgroup_mm || (props_dev->use_mm_manual && mm_id_manual_ok))) {
        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        // extra buffers for intermediate id mapping
        ggml_metal_buffer_id bid_tpe = bid_dst;
        bid_tpe.offs += ggml_nbytes(op);

        ggml_metal_buffer_id bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        ggml_metal_buffer_id bid_amax = bid_ids;
        bid_amax.offs += ggml_metal_op_mul_mat_id_extra_ids(op);

        // src1 prec [TAG_GGML_PREC]
        const bool use_amax = ggml_get_op_params_i32(op, 3) == GGML_PREC_F32;

        // src1 rescale factors, computed before the matmul
        // ref: https://github.com/ggml-org/llama.cpp/pull/26223
        if (use_amax) {
            ggml_metal_kargs_mul_mm_id_amax args = {
                /*.ne00 =*/ ne10,
                /*.ne01 =*/ ne11,
                /*.ne02 =*/ ne12,
                /*.nb01 =*/ nb11,
                /*.nb02 =*/ nb12,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_amax_part(lib);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_amax, 2);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, N_MM_NPART_AMAX, 1, 1, 256, 1, 1);
        }

        {
            ggml_metal_kargs_mul_mm_id_map0 args = {
                ne02_run,
                ne10,
                ne11, // n_expert_used (bcast)
                nb11,
                nb12,
                ne21, // n_tokens
                ne20, // n_expert_used
                nb21,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02_run, ne20);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(ne02_run <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  2);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02_run, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        if (use_amax) {
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_amax(lib);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_buffer  (enc, bid_amax, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 32, 1, 1);

            // the next kernel has to wait for the amax data
            ggml_metal_op_concurrency_reset(ctx);
        }

        {
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id(lib, op);

            ggml_metal_kargs_mul_mm_id args = {
                /*.ne00  =*/ ne00,
                /*.ne02  =*/ ne02_run,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne11  =*/ ne11, // n_expert_used (bcast)
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne20  =*/ ne20, // n_expert_used
                /*.ne21  =*/ ne21, // n_tokens
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  3);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  4);
            ggml_metal_encoder_set_buffer  (enc, bid_dst,  5);
            ggml_metal_encoder_set_buffer  (enc, bid_amax, 6);

            const size_t smem = pipeline.smem;

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            const int nr1 = pipeline.nr1;

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne21 + nr1 - 1)/nr1, (ne01 + 63)/64, ne02_run, 128, 1, 1);
        }
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id(lib, op);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const size_t smem = pipeline.smem;

        ggml_metal_kargs_mul_mv_id args = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ nb21,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02_run,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ nb1,
            /*.nr0  =*/ nr0,
        };

        if (ggml_is_quantized(op->src[0]->type)) {
            GGML_ASSERT(ne00 >= nsg*nr0);
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst,  3);
        ggml_metal_encoder_set_buffer(enc, bid_src2, 4);

        const int64_t _ne1 = 1;
        const int64_t ne123 = ne20*ne21;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        // wave64 decode: run the MoE mat-vec on a full 64-wide simdgroup (matches the
        // kernel's FC_mul_mv_nw); 32 for wave32 and the CPU-bound types.
        int id_nth = 32;
        if (props_dev->wave64_decode &&
            (op->src[0]->type == GGML_TYPE_F16 ||
             op->src[0]->type == GGML_TYPE_F32 ||
             op->src[0]->type == GGML_TYPE_BF16 ||
             op->src[0]->type == GGML_TYPE_Q8_0 ||
             op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q5_K ||
             op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_Q2_K ||
             op->src[0]->type == GGML_TYPE_Q3_K ||
             op->src[0]->type == GGML_TYPE_MXFP4 ||
             op->src[0]->type == GGML_TYPE_IQ2_XXS ||
             op->src[0]->type == GGML_TYPE_IQ2_XS ||
             op->src[0]->type == GGML_TYPE_IQ3_XXS ||
             op->src[0]->type == GGML_TYPE_IQ3_S ||
             op->src[0]->type == GGML_TYPE_IQ2_S ||
             op->src[0]->type == GGML_TYPE_IQ1_S ||
             op->src[0]->type == GGML_TYPE_IQ1_M ||
             op->src[0]->type == GGML_TYPE_IQ4_NL ||
             op->src[0]->type == GGML_TYPE_IQ4_XS ||
             op->src[0]->type == GGML_TYPE_Q4_0 ||
             op->src[0]->type == GGML_TYPE_Q4_1 ||
             op->src[0]->type == GGML_TYPE_Q5_0 ||
             op->src[0]->type == GGML_TYPE_Q5_1 ||
             op->src[0]->type == GGML_TYPE_Q1_0 ||
             op->src[0]->type == GGML_TYPE_Q2_0 ||
             op->src[0]->type == GGML_TYPE_PQ2_0 ||
             op->src[0]->type == GGML_TYPE_PTQ1_0)) {
            id_nth = props_dev->simd_width;
        }

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123, id_nth, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123, id_nth, nsg, 1);
        }
    }

    return 1;
}

int ggml_metal_op_add_id(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_kargs_add_id args = {
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb11 =*/ nb11,
        /*.nb21 =*/ nb21,
    };

    auto pipeline = ggml_metal_library_get_pipeline_base(lib, GGML_OP_ADD_ID);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, 1, nth, 1, 1);

    return 1;
}

// Measured depth where the 32-way split starts to win for a single-row dk128 query on 64 lanes, per
// local query and KV heads; shapes that were not measured keep the full-die depth.
static int64_t ggml_metal_fa_wg_dk128_min_kv_w64(int64_t n_head, int64_t n_head_kv) {
    static const struct { int64_t n_head, n_head_kv, min_kv; } measured[] = {
        {  8, 4, 1536 },
        { 16, 8, 3584 },
        { 16, 4, 3072 },
        { 16, 2, 3584 },
        { 16, 1, 4096 },
        { 20, 4, 4608 },
    };
    for (const auto & m : measured) {
        if (m.n_head == n_head && m.n_head_kv == n_head_kv) {
            return m.min_kv;
        }
    }
    return 16384;
}

// The same for dk64, where the split only pays with few local query heads; 0 keeps it closed.
static int64_t ggml_metal_fa_wg_dk64_min_kv_w64(int64_t n_head, int64_t n_head_kv) {
    static const struct { int64_t n_head, n_head_kv, min_kv; } measured[] = {
        { 16, 4, 6144 },
    };
    for (const auto & m : measured) {
        if (m.n_head == n_head && m.n_head_kv == n_head_kv) {
            return m.min_kv;
        }
    }
    return 0;
}

// Suffix for the AMD vec FA pipeline name, or nullptr for types we don't instantiate.
static const char * ggml_metal_fa_amd_type_suffix(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_TURBO2_0: return "turbo2";
        case GGML_TYPE_TURBO3_0: return "turbo3";
        case GGML_TYPE_TURBO4_0: return "turbo4";
        default:             return nullptr;
    }
}

bool ggml_metal_op_flash_attn_ext_use_vec(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const int64_t ne00 = op->src[0]->ne[0]; // head size
    const int64_t ne01 = op->src[0]->ne[1]; // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    return (ne01 < 20) && (ne00 % 32 == 0);
}

// ref: https://github.com/ggml-org/llama.cpp/pull/27390
// dequantize the quantized KV cache to F16 before running the F16 flash attention kernels
static bool ggml_metal_op_flash_attn_ext_use_kv_f16(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    // depending on compute/bandwidth ratio, dequant to f16 kv is not always beneficial
    // ref: https://github.com/ggml-org/llama.cpp/pull/27390#issuecomment-5355152767
    // TODO: tune per device
    if (op->src[0]->ne[1] < 32) {
        return false;
    }

    switch (op->src[1]->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

// returns the n_kv_max hint if the sparse path is available for this op, or 0 otherwise
// the mask (src[3]) remains the single source of truth: finite entries are the valid KV positions,
// n_kv_max is only an upper bound on their number per mask row, used to size the index lists
static int ggml_metal_op_flash_attn_ext_n_kv_max_sparse(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    int32_t n_kv_max = 0;
    memcpy(&n_kv_max, ((const int32_t *) op->op_params) + 4, sizeof(n_kv_max));

    if (n_kv_max <= 0) {
        return 0;
    }

    // the sparse indices are gathered from the mask
    if (!op->src[3]) {
        return 0;
    }

    // bound the size of the index lists
    if (n_kv_max > 4096) {
        return 0;
    }

    // vec kernel instantiations exist for these (type, dk, dv) combinations only
    const int64_t dk = op->src[1]->ne[0];
    const int64_t dv = op->src[2]->ne[0];

    const bool dk_dv_ok = (dk == 32  && dv == 32)  ||
                          (dk == 64  && dv == 64)  ||
                          (dk == 96  && dv == 96)  ||
                          (dk == 96  && dv == 64)  ||
                          (dk == 128 && dv == 128) ||
                          (dk == 192 && dv == 128) ||
                          (dk == 192 && dv == 192) ||
                          (dk == 256 && dv == 256) ||
                          (dk == 320 && dv == 256) ||
                          (dk == 512 && dv == 512) ||
                          (dk == 576 && dv == 512);

    if (!dk_dv_ok) {
        return 0;
    }

    switch (op->src[1]->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F32:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            break;
        default:
            return 0;
    }

    return n_kv_max;
}

// in some models (e.g. MLA-based), V is a view of K (the first ne20 elements of each K row);
// the dequantized V is then a view of the dequantized K and does not need its own dequant or scratch
// - ref: https://github.com/ggml-org/llama.cpp/pull/13435
static bool ggml_metal_op_flash_attn_ext_v_is_view_of_k(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = op->src[1];
    const ggml_tensor * V = op->src[2];

    return V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
}

// size of the F16 dequantized K tensor; the dequantized V tensor follows it in the same scratch buffer
static size_t ggml_metal_op_flash_attn_ext_kv_f16_k_size(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    return GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne10*ne11*ne12*ne13, 16);
}

size_t ggml_metal_op_flash_attn_ext_extra_pad(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;
    const bool use_kv_f16 = ggml_metal_op_flash_attn_ext_use_kv_f16(op);

    // when the KV is dequantized to F16, the pad kernel copies the tail chunk from the F16 scratch buffer
    // note: when V is a view of K, the dequantized V is read from the dequantized K with K's row stride
    const bool v_is_view_of_k = use_kv_f16 && ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);
    uint64_t nb11_pad = nb11;
    uint64_t nb21_pad = nb21;

    if (use_kv_f16) {
        nb11_pad = sizeof(ggml_fp16_t)*ne10;
        nb21_pad = sizeof(ggml_fp16_t)*(v_is_view_of_k ? ne10 : ne20);
    }

    // note: the non-vec kernel requires more extra memory, so always reserve for it
    GGML_ASSERT(OP_FLASH_ATTN_EXT_NCPSG >= OP_FLASH_ATTN_EXT_VEC_NCPSG);

    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (false) {
        // note: always reserve the padding space to avoid graph reallocations
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_VEC_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_VEC_NCPSG*(
                nb11_pad*ne12*ne13 +
                nb21_pad*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    } else {
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_NCPSG*(
                nb11_pad*ne12*ne13 +
                nb21_pad*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    }

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_blk(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    if (!has_mask) {
        return res;
    }

    const bool is_vec = ggml_metal_op_flash_attn_ext_use_vec(op);

    // this optimization is not useful for the vector kernels
    // note: always reserve the blk buffer to avoid graph reallocations
    //if (is_vec) {
    //    return res;
    //}

    const int nqptg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NQPSG : OP_FLASH_ATTN_EXT_NQPSG;
    const int ncpsg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NCPSG : OP_FLASH_ATTN_EXT_NCPSG;

    const int64_t ne1 = (ne01 + nqptg - 1)/nqptg;
    const int64_t ne0 = (ne30 + ncpsg - 1)/ncpsg;

    res += GGML_PAD(ggml_type_size(GGML_TYPE_I8)*ne0*ne1*ne32*ne33, 32);

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_tmp(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (true) {
        const int64_t nwg = 32;
        const int64_t ne01_max = std::min(ne01, 32);

        // temp buffer for writing the results from each workgroup
        // - ne20: the size of the Value head
        // -  + 2: the S and M values for each intermediate result
        res += ggml_type_size(GGML_TYPE_F32)*(ne01_max*ne02*ne03*nwg*(ne20 + 2));
    }

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_kv_f16(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (!ggml_metal_op_flash_attn_ext_use_kv_f16(op)) {
    //    return 0;
    //}

    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);

    const size_t k_size = ggml_metal_op_flash_attn_ext_kv_f16_k_size(op);

    // when V is a view of K, the dequantized V is a view of the dequantized K
    const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);
    if (v_is_view_of_k) {
        return k_size;
    }

    const size_t v_size = GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne20*ne21*ne22*ne23, 16);

    return k_size + v_size;
}

// size of the sparse index lists: one list of KV indices per mask row,
// padded with -1 up to a multiple of OP_FLASH_ATTN_EXT_VEC_NCPSG
size_t ggml_metal_op_flash_attn_ext_extra_idx(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);

    const int n_kv_max = ggml_metal_op_flash_attn_ext_n_kv_max_sparse(op);

    if (n_kv_max <= 0) {
        return 0;
    }

    const int n_kv_max_padded = GGML_PAD(n_kv_max, OP_FLASH_ATTN_EXT_VEC_NCPSG);

    return GGML_PAD(sizeof(int32_t)*(size_t) n_kv_max_padded*ne31*ne32*ne33, 16);
}

int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS( int32_t, nb,  op,         nb);

    GGML_ASSERT(ne00 % 4 == 0);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);

    //GGML_ASSERT(ggml_are_same_shape (src1, src2));
    GGML_ASSERT(ne11 == ne21);
    GGML_ASSERT(ne12 == ne22);

    GGML_ASSERT(!op->src[3] || op->src[3]->type == GGML_TYPE_F16);
    GGML_ASSERT(!op->src[3] || op->src[3]->ne[1] >= op->src[0]->ne[1] &&
            "the Flash-Attention Metal kernel requires the mask to be at least n_queries big");

    float scale;
    float max_bias;
    float logit_softcap;

    memcpy(&scale,         ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(logit_softcap));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const bool has_mask  = op->src[3] != NULL;
    const bool has_sinks = op->src[4] != NULL;
    const bool has_bias  = max_bias != 0.0f;
    const bool has_scap  = logit_softcap != 0.0f;

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    GGML_ASSERT(ne01 < 65536);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_src3 = has_mask  ? ggml_metal_get_buffer_id(op->src[3]) : bid_src0;
    ggml_metal_buffer_id bid_src4 = has_sinks ? ggml_metal_get_buffer_id(op->src[4]) : bid_src0;

    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_pad = bid_dst;
    bid_pad.offs += ggml_nbytes(op);

    ggml_metal_buffer_id bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    ggml_metal_buffer_id bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    // AMD discrete GPUs miscompile the templated vec kernel, so route attention (decode
    // and prefill) to the dedicated AMD kernel for the standard K/V pairs it covers.
    // Keeps quantized KV on the GPU instead of the CPU fallback. Gated by TOSH_FA_AMD.
    char fa_amd_pipeline[80] = {0};
    // dk 72 is instantiated for f16 K/V only (vision towers); see the .metal side.
    const bool fa_amd_f16_kv =
        op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16;
    const bool fa_amd_dk72 = ne00 == 72 && fa_amd_f16_kv;
    // UNet diffusion heads, f16 only like dk72: SD 1.5 runs 40 at full resolution,
    // where the score matrix costs 2.7 GB at 768 and does not fit at all at 1024
    const bool fa_amd_unet = (ne00 == 40 || ne00 == 80 || ne00 == 160) && fa_amd_f16_kv;
    // dk 384/640 are instantiated for Turbo KV only; see the .metal side
    const auto fa_amd_is_turbo = [](enum ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
    };
    const bool fa_amd_turbo_kv = fa_amd_is_turbo(op->src[1]->type) || fa_amd_is_turbo(op->src[2]->type);
    if (ne00 == ne20 &&
        (ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 ||
         ((ne00 == 384 || ne00 == 640) && fa_amd_turbo_kv) ||
         fa_amd_dk72 || fa_amd_unet)) {
        const char * ks = ggml_metal_fa_amd_type_suffix(op->src[1]->type);
        const char * vs = ggml_metal_fa_amd_type_suffix(op->src[2]->type);
        if (ks && vs) {
            snprintf(fa_amd_pipeline, sizeof(fa_amd_pipeline), "kernel_flash_attn_ext_vec_amd_dk%d_k%s_v%s%s", (int) ne00, ks, vs,
                     props_dev->simd_width == 64 ? "_w64" : "");
        }
    } else if (((ne00 == 576 && ne20 == 512) || (ne00 == 320 && ne20 == 256)) &&
               op->src[1]->type == op->src[2]->type) {
        const char * ks = ggml_metal_fa_amd_type_suffix(op->src[1]->type);
        const char * vs = ggml_metal_fa_amd_type_suffix(op->src[2]->type);
        if (ks && vs) {
            snprintf(fa_amd_pipeline, sizeof(fa_amd_pipeline), "kernel_flash_attn_ext_vec_amd_dk%d_dv%d_k%s_v%s%s",
                     (int) ne00, (int) ne20, ks, vs, props_dev->simd_width == 64 ? "_w64" : "");
        }
    }
    // wave32 rides on simdgroup_reduction; wave64 uses the _w64 instantiations and
    // is gated by wave64_decode (simdgroup_reduction is forced off there).
    const bool fa_amd_width_ok = props_dev->simd_width == 32
        ? props_dev->has_simdgroup_reduction
        : props_dev->wave64_decode;
    // sparse attention (QSA / DeepSeek-V4): the mask already carries the selection, the
    // index kernel compacts it and the vec kernel walks the list instead of the whole cache
    const int amd_n_kv_max = ggml_metal_op_flash_attn_ext_n_kv_max_sparse(op);
    const bool amd_sparse  = amd_n_kv_max > 0 && ne01 == 1;
    const int amd_n_kv_pad = amd_sparse ? GGML_PAD(amd_n_kv_max, OP_FLASH_ATTN_EXT_VEC_NCPSG) : 0;

    if (getenv("TOSH_FA_AMD") != NULL && fa_amd_pipeline[0] != '\0' &&
        !props_dev->has_simdgroup_mm && fa_amd_width_ok &&
        !has_bias && !has_scap) {

        ggml_metal_kargs_flash_attn_ext_amd args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
            /*.has_sinks     =*/ has_sinks ? 1 : 0,
            /*.has_mask      =*/ has_mask ? 1 : 0,
            /*.n_kv_max_padded =*/ amd_n_kv_pad,
        };

        // queries per threadgroup, matching the instantiation for this head size
        static const bool pf512 = getenv("TOSH_FA_PF_DK512") != nullptr;
        // MLA keeps K wider than V; the tile kernel it used to fall back to measures ~26% slower
        static const bool pfmla = getenv("TOSH_FA_PF_MLA_DISABLE") == nullptr;
        const bool fa_pf_512 = pf512 && ne00 == 512 && ne00 == ne20;
        const bool fa_pf_mla = pfmla && ne00 == 576 && ne20 == 512 &&
                               op->src[2]->type == GGML_TYPE_F16;
        const bool fa_pf_mla320 = ne00 == 320 && ne20 == 256 && op->src[1]->type == op->src[2]->type;
        // BQ must match the instantiation: dispatching in wider blocks than the kernel
        // covers leaves most queries uncomputed
        int fa_pf_bq = (ne00 == 256 || ne00 == 160) ? 32
            : ((fa_pf_512 || fa_pf_mla || fa_pf_mla320) ? 16 : 64);
        const bool fa_pf_head_ok =
            ne00 == 64 || ne00 == 128 || ne00 == 256 || fa_amd_dk72 || fa_amd_unet || fa_pf_512 ||
            fa_pf_mla || fa_pf_mla320;
        // PF always dispatches 256 threads and subdivides them into logical
        // 16-lane query groups, so the same specialization works on wave32
        // (8 physical simdgroups) and wave64 (4 physical simdgroups).
        if (getenv("TOSH_FA_PF_DEBUG") != nullptr) {
            fprintf(stderr, "FA_PF gate dk=%d dv=%d ne01=%d head_ok=%d ktype=%s vtype=%s\n",
                    (int) ne00, (int) ne20, (int) ne01, (int) fa_pf_head_ok,
                    ggml_type_name(op->src[1]->type), ggml_type_name(op->src[2]->type));
        }
        if ((props_dev->simd_width == 32 || props_dev->simd_width == 64) &&
            ne01 >= 32 && fa_pf_head_ok && getenv("TOSH_FA_PF_DISABLE") == NULL) {
            char fa_pf_pipeline[80];
            bool fa_pf_asym = false;
            if (fa_pf_mla) {
                snprintf(fa_pf_pipeline, sizeof(fa_pf_pipeline), "kernel_flash_attn_ext_pf_amd_dk576_dv512_k%s_vf16",
                         ggml_metal_fa_amd_type_suffix(op->src[1]->type));
                fa_pf_asym = true;
            } else if (ne20 != ne00) {
                snprintf(fa_pf_pipeline, sizeof(fa_pf_pipeline), "kernel_flash_attn_ext_pf_amd_dk%d_dv%d_k%s_v%s",
                         (int) ne00, (int) ne20,
                         ggml_metal_fa_amd_type_suffix(op->src[1]->type), ggml_metal_fa_amd_type_suffix(op->src[2]->type));
                fa_pf_asym = true;
            } else {
            snprintf(fa_pf_pipeline, sizeof(fa_pf_pipeline), "kernel_flash_attn_ext_pf_amd_dk%d_k%s_v%s", (int) ne00,
                     ggml_metal_fa_amd_type_suffix(op->src[1]->type), ggml_metal_fa_amd_type_suffix(op->src[2]->type));

                // the 64-lane twin lives in fa_w64.metal, with the lane group as a parameter.
                // Only the symmetric heads have one; MLA and the asymmetric pairs stay above.
                // TOSH_FA_PF_LG picks the group so the three can be compared; 16 reproduces
                // the 32-lane kernel exactly and is the control.
                static const int pf_lg = []() {
                    // the cross sweep left the group at sixteen: wider only wins at depth 0,
                    // where this kernel weighs least, and loses 3 to 37% deeper in
                    const char * s = getenv("TOSH_FA_PF_LG");
                    return s ? atoi(s) : 16;
                }();
                // TOSH_FA_PF_BQ picks the query-block twin of that group, which is what keeps
                // the rows per thread fixed while the group widens
                static const int pf_bq = []() {
                    const char * s = getenv("TOSH_FA_PF_BQ");
                    return s ? atoi(s) : 0;
                }();
                // TOSH_FA_PF_BK picks the key-block twin: the counterpart of BQ on the other
                // side of the K/V reuse, so a bigger block pays the same traffic fewer times
                static const int pf_bk = []() {
                    const char * s = getenv("TOSH_FA_PF_BK");
                    return s ? atoi(s) : 0;
                }();
                if (props_dev->simd_width == 64 && (pf_lg == 16 || pf_lg == 32 || pf_lg == 64) &&
                    (ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512) &&
                    getenv("TOSH_FA_PF_W64_DISABLE") == nullptr) {
                    char w64_name[96];
                    if (pf_bq > 0) {
                        snprintf(w64_name, sizeof(w64_name), "%s_w64_lg%dq%dk%d", fa_pf_pipeline, pf_lg, pf_bq,
                                 pf_bk > 0 ? pf_bk : 64);
                    } else {
                        snprintf(w64_name, sizeof(w64_name), "%s_w64_lg%d", fa_pf_pipeline, pf_lg);
                    }
                    if (ggml_metal_library_get_pipeline(lib, w64_name).pipeline ||
                        ggml_metal_library_compile_pipeline(lib, w64_name, w64_name, nullptr).pipeline) {
                        snprintf(fa_pf_pipeline, sizeof(fa_pf_pipeline), "%s", w64_name);
                        // the grid steps in query blocks, so it has to follow the twin
                        if (pf_bq > 0) {
                            fa_pf_bq = pf_bq;
                        }
                    }
                }
            }

            // the asymmetric heads compose their name in the branches above, so their twin is
            // picked here instead. Same group of sixteen; the fallback is the 32-lane kernel.
            if (fa_pf_asym && props_dev->simd_width == 64 &&
                getenv("TOSH_FA_PF_W64_DISABLE") == nullptr) {
                char w64_name[96];
                // sixteen reproduces the 32-lane kernel exactly, so it is the control; the
                // sweep that closed the symmetric heads there never covered an asymmetric one
                const char * slg = getenv("TOSH_FA_PF_LG");
                snprintf(w64_name, sizeof(w64_name), "%s_w64_lg%d", fa_pf_pipeline, slg ? atoi(slg) : 16);
                if (ggml_metal_library_get_pipeline(lib, w64_name).pipeline ||
                    ggml_metal_library_compile_pipeline(lib, w64_name, w64_name, nullptr).pipeline) {
                    snprintf(fa_pf_pipeline, sizeof(fa_pf_pipeline), "%s", w64_name);
                }
            }

            auto pipeline = ggml_metal_library_get_pipeline(lib, fa_pf_pipeline);
            if (!pipeline.pipeline) {
                pipeline = ggml_metal_library_compile_pipeline(lib, fa_pf_pipeline, fa_pf_pipeline, nullptr);
            }

            // Which twin a head actually lands on cannot be read from the timings: a switch that
            // never reaches the kernel gives the same number on both sides.
            if (getenv("TOSH_FA_PF_DEBUG") != nullptr) {
                fprintf(stderr, "FA_PF dk=%d dv=%d bq=%d -> %s%s\n",
                        (int) ne00, (int) ne20, fa_pf_bq, fa_pf_pipeline,
                        pipeline.pipeline ? "" : "  (NO EXISTE, cae al tile)");
            }

            // it fits in registers with nothing to spare; a compiler bump could
            // push it under the 256 threads it dispatches. A driver that refused to build
            // the kernel leaves the pipeline null, which the tile path below covers: older
            // GCN cards reject these over their local storage limit.
            if (pipeline.pipeline &&
                ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) >= 256) {
                ggml_metal_encoder_set_pipeline(enc, pipeline);
                ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
                ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
                ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
                ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
                ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
                ggml_metal_encoder_set_buffer  (enc, bid_dst,  7);

                ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + fa_pf_bq - 1)/fa_pf_bq, ne02, ne03,
                                                         props_dev->simd_width, 256/props_dev->simd_width, 1);

                return 1;
            }
        }

        // prefill goes to the tile kernel (K/V shared across 16 queries);
        // decode keeps the per-query vec kernel
        if (ne01 >= 16 && getenv("TOSH_FA_TILE_DISABLE") == NULL) {
            char fa_tile_pipeline[80];
            if (ne20 != ne00) {
                // MLA heads carry the value width in the name: 576/512 and 320/256
                snprintf(fa_tile_pipeline, sizeof(fa_tile_pipeline), "kernel_flash_attn_ext_tile_amd_dk%d_dv%d_k%s_v%s%s",
                         (int) ne00, (int) ne20,
                         ggml_metal_fa_amd_type_suffix(op->src[1]->type), ggml_metal_fa_amd_type_suffix(op->src[2]->type),
                         props_dev->simd_width == 64 ? "_w64" : "");
            } else {
                snprintf(fa_tile_pipeline, sizeof(fa_tile_pipeline), "kernel_flash_attn_ext_tile_amd_dk%d_k%s_v%s%s", (int) ne00,
                         ggml_metal_fa_amd_type_suffix(op->src[1]->type), ggml_metal_fa_amd_type_suffix(op->src[2]->type),
                         props_dev->simd_width == 64 ? "_w64" : "");
            }

            auto pipeline = ggml_metal_library_get_pipeline(lib, fa_tile_pipeline);
            if (!pipeline.pipeline) {
                pipeline = ggml_metal_library_compile_pipeline(lib, fa_tile_pipeline, fa_tile_pipeline, nullptr);
            }

            // a missed instantiation leaves a null pipeline; fall through to the vec route
            if (pipeline.pipeline) {
                ggml_metal_encoder_set_pipeline(enc, pipeline);
                ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
                ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
                ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
                ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
                ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
                ggml_metal_encoder_set_buffer  (enc, bid_dst,  7);

                // queries per threadgroup must match the instantiation: 16 on wave32, 8 on wave64
                const int fa_nq = props_dev->simd_width == 64 ? 8 : 16;
                ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + fa_nq - 1)/fa_nq, ne02, ne03, props_dev->simd_width, fa_nq, 1);

                return 1;
            }
        }

        // must match the .metal instantiations: q4_0 dequant caps the pipeline, so those
        // pairs drop to 24 (dk128 q4_0 keys, dk64 subw=8 with q4_0 in either K or V).
        int fa_amd_nsg = ne00 == 128 ? 16 : ne00 <= 128 ? 32 : 16;
        if (props_dev->simd_width == 32) {
            const bool k_q4 = op->src[1]->type == GGML_TYPE_Q4_0;
            const bool v_q4 = op->src[2]->type == GGML_TYPE_Q4_0;
            // the Turbo macro instantiates every head at 16, and pairing q4_0 with it
            // leaves the pipeline at 512 threads, so 24 would overrun what it admits
            if (!fa_amd_turbo_kv && ((ne00 == 128 && k_q4) || (ne00 == 64 && (k_q4 || v_q4)))) {
                fa_amd_nsg = 24;
            }
        }
        if (props_dev->simd_width == 64) {
            fa_amd_nsg /= 2;
        }
        // the simdgroup count per head was never swept on either width; the pipeline caps it,
        // so an override that overruns is dropped below rather than dispatched
        // one value for every head size overruns the pipelines that cap lower, so the
        // override is per head size: TOSH_FA_AMD_NSG_64, _128, _WIDE
        static const int nsg_env_64 = []() {
            const char * s = getenv("TOSH_FA_AMD_NSG_64");
            return s ? atoi(s) : 0;
        }();
        static const int nsg_env_128 = []() {
            const char * s = getenv("TOSH_FA_AMD_NSG_128");
            return s ? atoi(s) : 0;
        }();
        static const int nsg_env_wide = []() {
            const char * s = getenv("TOSH_FA_AMD_NSG_WIDE");
            return s ? atoi(s) : 0;
        }();
        const int nsg_env = ne00 == 64 ? nsg_env_64 : ne00 == 128 ? nsg_env_128 : nsg_env_wide;
        if (nsg_env > 0) {
            fa_amd_nsg = nsg_env;
        }

        // one threadgroup per query head leaves most of the GPU idle on models with
        // few heads; split the cache across 32 and let upstream's reduce merge them
        // Head 128 needs a deeper cache than the wide heads before the 32-way reduce pays
        // for itself, so it carries its own threshold. The depth it turns at is not the same
        // on both widths, so each keeps its own.
        static const int64_t wg_dk128_env = []() {
            const char * s = getenv("TOSH_FA_WG_DK128_MIN_KV");
            return s ? (int64_t) atoll(s) : (int64_t) -1;
        }();
        const bool wg_wide_head =
            ne00 == 256 || ne00 == 512 || (ne00 == 576 && ne20 == 512) ||
            (ne00 == 320 && ne20 == 256);
        const int64_t wg_dk128_min_kv = wg_dk128_env >= 0 ? wg_dk128_env
                                      : (props_dev->simd_width == 64 ? ggml_metal_fa_wg_dk128_min_kv_w64(ne02, ne12) : 24576);
        const bool wg_dk128_width_ok = props_dev->simd_width == 32 || props_dev->simd_width == 64;
        // TOSH_FA_WG_DK64=<kv> sets the head-64 depth for every shape, 0 closes it
        static const int64_t wg_dk64_env = []() {
            const char * s = getenv("TOSH_FA_WG_DK64");
            return s ? (int64_t) atoll(s) : (int64_t) -1;
        }();
        const int64_t wg_dk64_min_kv = wg_dk64_env >= 0 ? wg_dk64_env
                                     : (props_dev->simd_width == 64 ? ggml_metal_fa_wg_dk64_min_kv_w64(ne02, ne12) : 0);
        const bool wg_dk64 = ne00 == 64 && ne20 == 64 && wg_dk128_width_ok &&
                             wg_dk64_min_kv > 0 && ne11 >= wg_dk64_min_kv;
        const bool wg_dk128 = (ne00 == 128 && ne20 == 128 && wg_dk128_width_ok &&
                               wg_dk128_min_kv > 0 && ne11 >= wg_dk128_min_kv) || wg_dk64;
        const int fa_amd_nwg =
            (wg_wide_head || wg_dk128) && ne01 == 1 && !has_sinks &&
            (props_dev->simd_width == 32 || props_dev->simd_width == 64) &&
            (wg_dk128 || ne11 >= 1024) &&
            getenv("TOSH_FA_WG_DISABLE") == NULL ? 32 : 1;

        char fa_amd_wg_pipeline[96];
        if (fa_amd_nwg > 1) {
            snprintf(fa_amd_wg_pipeline, sizeof(fa_amd_wg_pipeline), "%s_wg32", fa_amd_pipeline);
        }

        // sparse: build the compacted index list first, on the same encoder
        ggml_metal_buffer_id bid_amd_idx = bid_tmp;
        if (amd_sparse) {
            bid_amd_idx.offs += ggml_metal_op_flash_attn_ext_extra_tmp(op)
                              + ggml_metal_op_flash_attn_ext_extra_kv_f16(op);

            ggml_metal_kargs_flash_attn_ext_vec_idx argsi = {
                /*.ne30            =*/ ne30,
                /*.ne31            =*/ ne31,
                /*.ne32            =*/ ne32,
                /*.ne33            =*/ ne33,
                /*.nb31            =*/ nb31,
                /*.nb32            =*/ nb32,
                /*.nb33            =*/ nb33,
                /*.n_kv_max        =*/ amd_n_kv_max,
                /*.n_kv_max_padded =*/ amd_n_kv_pad,
            };

            auto pipei = ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(lib, op);

            ggml_metal_encoder_set_pipeline(enc, pipei);
            ggml_metal_encoder_set_bytes   (enc, &argsi, sizeof(argsi), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3,    1);
            ggml_metal_encoder_set_buffer  (enc, bid_amd_idx, 2);

            int nthi = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipei), 256);
            // whole simdgroups only: a partial one loses its slice of the count
            const int nwi = props_dev->simd_width;
            nthi = std::max(nwi, (nthi/nwi)*nwi);

            ggml_metal_encoder_dispatch_threadgroups(enc, ne31, ne32, ne33, nthi, 1, 1);
            ggml_metal_op_concurrency_reset(ctx);
        }

        auto pipeline = ggml_metal_library_get_pipeline(lib, fa_amd_nwg > 1 ? fa_amd_wg_pipeline : fa_amd_pipeline);
        if (!pipeline.pipeline) {
            const char * nm = fa_amd_nwg > 1 ? fa_amd_wg_pipeline : fa_amd_pipeline;
            pipeline = ggml_metal_library_compile_pipeline(lib, nm, nm, nullptr);
        }

        // dispatching more threads than the pipeline admits hangs the GPU, and how many
        // it admits depends on the K type's dequant; fall through rather than risk it
        if (!pipeline.pipeline) {
            // a missed instantiation leaves a null pipeline; fall through rather than crash
            GGML_LOG_WARN("%s: %s is not instantiated; using the generic kernels\n",
                          __func__, fa_amd_nwg > 1 ? fa_amd_wg_pipeline : fa_amd_pipeline);
        } else if (ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) < fa_amd_nsg*props_dev->simd_width) {
            GGML_LOG_WARN("%s: %s wants %d threads, pipeline allows %d; using the generic kernels\n",
                          __func__, fa_amd_pipeline, fa_amd_nsg*props_dev->simd_width,
                          ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        } else {

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
        ggml_metal_encoder_set_buffer  (enc, fa_amd_nwg > 1 ? bid_tmp : bid_dst, 7);
        ggml_metal_encoder_set_buffer  (enc, bid_amd_idx, 8);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03*fa_amd_nwg, props_dev->simd_width, fa_amd_nsg, 1);

        if (fa_amd_nwg > 1) {
            ggml_metal_op_concurrency_reset(ctx);

            ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = { (int32_t) (ne1*ne2*ne3) };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, fa_amd_nwg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, ne1*ne2*ne3, 1, 1, 32*fa_amd_nwg, 1, 1);
        }

        return 1;
        }
    }

    // The generic Metal kernels below need K and V to share a type.
    GGML_ASSERT(op->src[1]->type == op->src[2]->type);

    ggml_metal_buffer_id bid_kv_f16 = bid_tmp;
    bid_kv_f16.offs += ggml_metal_op_flash_attn_ext_extra_tmp(op);

    // sparse path: gather the finite mask entries into index lists and run the vec kernels over them
    const int n_kv_max_sparse = ggml_metal_op_flash_attn_ext_n_kv_max_sparse(op);
    const bool use_sparse = n_kv_max_sparse > 0;
    const int n_kv_max_padded = use_sparse ? GGML_PAD(n_kv_max_sparse, OP_FLASH_ATTN_EXT_VEC_NCPSG) : 0;

    // the vec kernels dequantize the KV inline; no need for the F16 dequant pass in the sparse path
    const bool use_kv_f16 = !use_sparse && ggml_metal_op_flash_attn_ext_use_kv_f16(op);

    ggml_metal_buffer_id bid_idx = bid_kv_f16;
    bid_idx.offs += ggml_metal_op_flash_attn_ext_extra_kv_f16(op);

    ggml_metal_buffer_id bid_k = bid_src1;
    ggml_metal_buffer_id bid_v = bid_src2;

    uint64_t nb10_attn = nb10;
    uint64_t nb11_attn = nb11;
    uint64_t nb12_attn = nb12;
    uint64_t nb13_attn = nb13;
    uint64_t nb20_attn = nb20;
    uint64_t nb21_attn = nb21;
    uint64_t nb22_attn = nb22;
    uint64_t nb23_attn = nb23;

    if (use_kv_f16) {
        assert(ggml_metal_op_flash_attn_ext_extra_kv_f16(op) != 0);

        const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);

        const int64_t nblocks1_64 = (ne10/ggml_blck_size(op->src[1]->type))*(int64_t) ne11*ne12*ne13;
        GGML_ASSERT(nblocks1_64 <= INT32_MAX);
        const int32_t nblocks1 = nblocks1_64;

        ggml_metal_buffer_id bid_v_f16 = bid_kv_f16;
        bid_v_f16.offs += ggml_metal_op_flash_attn_ext_kv_f16_k_size(op);

        auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(lib, op);
        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline0), 256);

        // K
        ggml_metal_kargs_flash_attn_ext_kv_f16 args_k = {
            /*.ne0    =*/ ne10,
            /*.ne1    =*/ ne11,
            /*.ne2    =*/ ne12,
            /*.ne3    =*/ ne13,
            /*.nb0    =*/ nb10,
            /*.nb1    =*/ nb11,
            /*.nb2    =*/ nb12,
            /*.nb3    =*/ nb13,
            /*.nblocks =*/ nblocks1,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline0);
        ggml_metal_encoder_set_bytes   (enc, &args_k, sizeof(args_k), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src1,        1);
        ggml_metal_encoder_set_buffer  (enc, bid_kv_f16, 2);

        ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks1 + nth - 1)/nth, 1, 1, nth, 1, 1);

        // V (skip when V is a view of K: the dequantized V is a view of the dequantized K)
        if (!v_is_view_of_k) {
            const int64_t nblocks2_64 = (ne20/ggml_blck_size(op->src[2]->type))*(int64_t) ne21*ne22*ne23;
            GGML_ASSERT(nblocks2_64 <= INT32_MAX);
            const int32_t nblocks2 = nblocks2_64;

            ggml_metal_kargs_flash_attn_ext_kv_f16 args_v = {
                /*.ne0    =*/ ne20,
                /*.ne1    =*/ ne21,
                /*.ne2    =*/ ne22,
                /*.ne3    =*/ ne23,
                /*.nb0    =*/ nb20,
                /*.nb1    =*/ nb21,
                /*.nb2    =*/ nb22,
                /*.nb3    =*/ nb23,
                /*.nblocks =*/ nblocks2,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args_v, sizeof(args_v), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2,        1);
            ggml_metal_encoder_set_buffer  (enc, bid_v_f16,       2);

            ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks2 + nth - 1)/nth, 1, 1, nth, 1, 1);
        }

        // the pad and attention kernels read the dequantized KV
        ggml_metal_op_concurrency_reset(ctx);

        bid_k = bid_kv_f16;
        bid_v = v_is_view_of_k ? bid_k : bid_v_f16;

        // contiguous F16 layout of the dequantized K
        nb10_attn = sizeof(ggml_fp16_t);
        nb11_attn = nb10_attn*ne10;
        nb12_attn = nb11_attn*ne11;
        nb13_attn = nb12_attn*ne12;

        // if V is a view of K, the dequantized V is read from the dequantized K with K's strides
        if (v_is_view_of_k) {
            nb20_attn = nb10_attn;
            nb21_attn = nb11_attn;
            nb22_attn = nb12_attn;
            nb23_attn = nb13_attn;
        } else {
            // contiguous F16 layout of the dequantized V
            nb20_attn = sizeof(ggml_fp16_t);
            nb21_attn = nb20_attn*ne20;
            nb22_attn = nb21_attn*ne21;
            nb23_attn = nb22_attn*ne22;
        }
    }

    if (!use_sparse && !ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        const int nqptg = OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 8  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = ne11 % ncpsg != 0;

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11_attn,
                /*.nb12    =*/nb12_attn,
                /*.nb13    =*/nb13_attn,
                /*.nb21    =*/nb21_attn,
                /*.nb22    =*/nb22_attn,
                /*.nb23    =*/nb23_attn,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_k,    1);
            ggml_metal_encoder_set_buffer  (enc, bid_v,    2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (has_mask) {
            assert(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

            ggml_metal_kargs_flash_attn_ext_blk args0 = {
                /*.ne01 =*/ ne01,
                /*.ne30 =*/ ne30,
                /*.ne31 =*/ ne31,
                /*.ne32 =*/ ne32,
                /*.ne33 =*/ ne33,
                /*.nb31 =*/ nb31,
                /*.nb32 =*/ nb32,
                /*.nb33 =*/ nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);

            const int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            const int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        const int is_q = !use_kv_f16 && ggml_is_quantized(op->src[1]->type) ? 1 : 0;

        // shared memory layout (halfs unless noted):
        //   queries/attn/result: Q*(DK + 2*PAD2(DV,64) + 4*C)
        //   quantized KV scratch: 16*32*NSG (only when is_q)
        const int64_t dv_pad = GGML_PAD(ne20, 64);

        auto fa_smem = [&](int32_t nsg) -> size_t {
            const size_t smem_half = nqptg*(ne00 + 2*dv_pad + 4*ncpsg) + is_q*(16*32*nsg);
            return GGML_PAD(smem_half*sizeof(ggml_fp16_t), 16);
        };

        // simdgroups per threadgroup (a.k.a. warps)
        int32_t nsg = ne00 >= 512 ? 8 : 4;

        const size_t smem = fa_smem(nsg);

        const int32_t ns10 = nb11_attn/nb10_attn;
        const int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg, use_kv_f16, ns10, ns20);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
        ggml_metal_encoder_set_buffer  (enc, bid_blk,  7);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  8);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
    } else {
        // half4x4 kernel
        // sparse: the index lists are per query row, so a threadgroup can share KV with Q == 1 only
        auto cfg = use_sparse
                ? ggml_metal_tuning::fa_vec_baseline_cfg((int) ne00, (int) ne20)
                : ggml_metal_tuning::fa_vec_pick(
                          props_dev->gpu_family,
                          (int) op->src[1]->type,
                          (int) ne00, (int) ne20,   // dk, dv (ne00 == dk for FA)
                          ne11, ne01);

        int nqptg = cfg.Q; // queries per threadgroup

        const int ncpsg = OP_FLASH_ATTN_EXT_VEC_NCPSG; // cache values per simdgroup !! sync with kernel template arguments !!
        const int nhptg = 1;                           // heads per threadgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg == 1 || nqptg == 2 || nqptg == 4);  // only instantiated Q values
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = !use_sparse && ne11 % ncpsg != 0;

        if (use_sparse) {
            assert(ggml_metal_op_flash_attn_ext_extra_idx(op) != 0);

            GGML_ASSERT(ne30 == ne11);

            ggml_metal_kargs_flash_attn_ext_vec_idx args0 = {
                /*.ne30              =*/ ne30,
                /*.ne31              =*/ ne31,
                /*.ne32              =*/ ne32,
                /*.ne33              =*/ ne33,
                /*.nb31              =*/ nb31,
                /*.nb32              =*/ nb32,
                /*.nb33              =*/ nb33,
                /*.n_kv_max          =*/ n_kv_max_sparse,
                /*.n_kv_max_padded   =*/ n_kv_max_padded,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(lib, op);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_idx,  2);

            int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline0), 256);
            // whole simdgroups only: a partial one loses its slice of the count
            const int nw0 = props_dev->simd_width;
            nth = std::max(nw0, (nth/nw0)*nw0);

            ggml_metal_encoder_dispatch_threadgroups(enc, ne31, ne32, ne33, nth, 1, 1);

            need_sync = true;
        }

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11_attn,
                /*.nb12    =*/nb12_attn,
                /*.nb13    =*/nb13_attn,
                /*.nb21    =*/nb21_attn,
                /*.nb22    =*/nb22_attn,
                /*.nb23    =*/nb23_attn,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_k,    1);
            ggml_metal_encoder_set_buffer  (enc, bid_v,    2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        // note: for simplicity assume the K is larger or equal than V
        GGML_ASSERT(ne10 >= ne20);

        // shared memory layout (halfs unless noted):
        //   queries:      Q*NSG*PAD2(ne00, 128)
        //   attn + mask:  NSG*4*Q*C
        //   results:      2*NSG*Q*PAD2(ne20, 128)
        //   sparse idx:   NSG*C ints (only when use_sparse)
        const int64_t dk_pad = GGML_PAD(ne00, 128);
        const int64_t dv_pad = GGML_PAD(ne20, 128);

        auto fa_vec_smem = [&](int64_t nsg, int32_t nqptg) -> size_t {
            const size_t smem_half = (size_t) (dk_pad + 4*ncpsg + 2*dv_pad)*nqptg*nsg;
            return GGML_PAD(smem_half*sizeof(ggml_fp16_t) + (use_sparse ? (size_t) nsg*ncpsg*sizeof(int) : 0), 16);
        };

        int64_t nsg = 1;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        int32_t nwg = 1;
        if (use_sparse) {
            if (ne01 > 32) {
                // large sparse batch
                nwg = 1;
                nsg = 1;
                if (n_kv_max_padded == 640) {
                    nsg = 4; // 640 % (4*32) == 0
                } else {
                    while (2*nwg*nsg*ncpsg < n_kv_max_padded && nsg < 4) {
                        nsg *= 2;
                    }
                }
            } else {
                // small sparse batch
                nwg = 32;
                nsg = 1;
                while (2*nwg*nsg*ncpsg < n_kv_max_padded && nsg < 4) {
                    nsg *= 2;
                }
            }
        } else {
            nwg = 32;
            nsg = 1;
            while (2*nwg*nsg*ncpsg < ne11 && nsg < 4) {
                nsg *= 2;
            }
        }

        // fall back to baseline (Q=1) if the tuned config exceeds threadgroup memory
        if (fa_vec_smem(nsg, nqptg) > props_dev->max_theadgroup_memory_size) {
            cfg   = ggml_metal_tuning::fa_vec_baseline_cfg((int) ne00, (int) ne20);
            nqptg = cfg.Q;  // = 1
        }

        const int32_t ns10 = nb11_attn/nb10_attn;
        const int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext_vec args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ use_sparse ? n_kv_max_padded : ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
            /*.n_kv_max_padded =*/ n_kv_max_padded,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, use_sparse, nqptg, cfg.NE, nsg, nwg, use_kv_f16, ns10, ns20);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, use_sparse ? bid_idx : bid_src0, 8);

        const size_t smem = fa_vec_smem(nsg, nqptg);

        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        if (nwg == 1) {
            // using 1 workgroup -> write the result directly into dst
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_dst, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);
        } else {
            // sanity checks
            assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);

            GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
            GGML_ASSERT((uint64_t)ne1*ne2*ne3 <= (1u << 31));

            // write the results from each workgroup into a temp buffer
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_tmp, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);

            // sync the 2 kernels
            ggml_metal_op_concurrency_reset(ctx);

            // reduce the results from the workgroups
            {
                const int32_t nrows = ne1*ne2*ne3;

                ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                    nrows,
                };

                auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

                ggml_metal_encoder_set_pipeline(enc, pipeline0);
                ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

                ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
            }
        }
    }

    return 1;
}

int ggml_metal_op_bin(ggml_metal_op_t ctx, int idx) {
    int n_fuse = 1;
    const ggml_metal_fusion * fusion = nullptr;

    if (ctx->use_fusion()) {
        int n = 1;
        fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        n_fuse = n;

        // snake activation autofuse: mul -> sin -> sqr -> mul -> add
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SNAKE) {
            ctx->count_fusions(fusion);
            return ggml_metal_op_snake_fused(ctx, idx);
        }

        // MoE output reduction: experts * weights -> weighted sum
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_MOE_REDUCE) {
            ctx->count_fusions(fusion);
            return ggml_metal_op_moe_reduce(ctx, idx);
        }
    }

    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion();

    const int debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_src1.offs },
    };

    // c[0] = add(a,    b[0])
    // c[1] = add(c[0], b[1])
    // c[2] = add(c[1], b[2])
    // ...
    if (use_fusion && fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_ADD_CHAIN) {
        // the offsets of the fused addends are relative to the start of the src1 buffer
        for (int i = 1; i < n_fuse; i++) {
            args.o1[i] = ggml_metal_get_buffer_id(ctx->node(idx + i)->src[1]).offs;
        }

        ctx->count_fusions(fusion);

        if (debug_fusion > 1) {
            GGML_LOG_DEBUG("%s: fuse: ADD x %d\n", __func__, n_fuse);
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    struct ggml_metal_pipeline_with_params pipeline;

    pipeline = ggml_metal_library_get_pipeline_bin(lib, op, n_fuse);

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne10 = ne10/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

    if (pipeline.cnt) {
        ggml_metal_encoder_dispatch_threadgroups(enc, args.ne0, ggml_nrows(op), 1, 1, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        int nth = 1;

        while (2*nth < args.ne0 && nth < nth_max) {
            nth *= 2;
        }

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return n_fuse;
}

int ggml_metal_op_silu_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    auto pipeline = ggml_metal_library_get_pipeline_silu_back(lib, op);

    const int64_t ne = ggml_nelements(op);

    ggml_metal_kargs_silu_back args = {
        /*.ne =*/ ne,
    };

    int arg_idx{0};

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), arg_idx++);

    const int nth = std::min<int64_t>(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne);
    const int64_t n = (ne + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_l2_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_kargs_l2_norm args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.eps   =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_l2_norm(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_group_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t ngrp = ((const int32_t *) op->op_params)[0];

    float eps;
    memcpy(&eps, op->op_params + 1, sizeof(float));

    ggml_metal_kargs_group_norm args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ngrp =*/ ngrp,
        /*.eps  =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_group_norm(lib, op);

    int nth = 32; // SIMD width
    //while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
    //    nth *= 2;
    //}

    //nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    //nth = std::min(nth, ne00/4);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion();

    const int debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_t =*/ ne00 % 4 == 0 ? ne00/4 : ne00,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
        /*.nef1   =*/ { ne01 },
        /*.nef2   =*/ { ne02 },
        /*.nef3   =*/ { ne03 },
        /*.nbf1   =*/ { nb01 },
        /*.nbf2   =*/ { nb02 },
        /*.nbf3   =*/ { nb03 },
        /*.scale =*/ 1.0f,
    };

    int n_fuse = 1;
    bool fused_norm_scale = false;

    ggml_metal_buffer_id bid_fuse[2] = { bid_src0, bid_src0 };

    // d[0] = norm(a)
    // d[1] = mul(d[0], b) or scale(d[0])
    // d[2] = add(d[1], c)
    if (use_fusion) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);

        if (fusion && (ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL || ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL_ADD)) {
            n_fuse = n;

            ctx->count_fusions(fusion);

            for (int i = 1; i < n_fuse; i++) {
                const ggml_tensor * fn = ctx->node(idx + i);

                bid_fuse[i - 1] = ggml_metal_get_buffer_id(fn->src[1]);

                args.nef1[i] = fn->src[1]->ne[1];
                args.nef2[i] = fn->src[1]->ne[2];
                args.nef3[i] = fn->src[1]->ne[3];

                args.nbf1[i] = fn->src[1]->nb[1];
                args.nbf2[i] = fn->src[1]->nb[2];
                args.nbf3[i] = fn->src[1]->nb[3];
            }

            if (debug_fusion > 1) {
                if (n_fuse == 2) {
                    GGML_LOG_DEBUG("%s: fuse: %s + MUL\n", __func__, ggml_op_name(op->op));
                }
                if (n_fuse == 3) {
                    GGML_LOG_DEBUG("%s: fuse: %s + MUL + ADD\n", __func__, ggml_op_name(op->op));
                }
            }
        }

        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_SCALE) {
            n_fuse = n;
            fused_norm_scale = true;

            ctx->count_fusions(fusion);

            const ggml_tensor * scale_node = ctx->node(idx + 1);
            args.scale = ggml_get_op_params_f32(scale_node, 0);

            if (debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: %s + SCALE\n", __func__, ggml_op_name(op->op));
            }
        }
    }

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    auto pipeline = fused_norm_scale ?
        ggml_metal_library_get_pipeline_norm_scale(lib, op) :
        ggml_metal_library_get_pipeline_norm(lib, op, n_fuse);

    int nth = 32; // SIMD width

    while (nth < args.ne00_t && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (args.ne00_t + 31)/32*32);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0,    1);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[0], 2);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[1], 3);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,     4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return n_fuse;
}

int ggml_metal_op_rope(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // make sure we have one or more position id(ne10) per token(ne02)
    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    const int nth = std::min(1024, ne00);

    const int n_past     = ((const int32_t *) op->op_params)[0];
    const int n_dims     = ((const int32_t *) op->op_params)[1];
  //const int mode       = ((const int32_t *) op->op_params)[2];
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    const int n_ctx_orig = ((const int32_t *) op->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) op->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) op->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) op->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) op->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) op->op_params + 10, sizeof(float));

    // mrope
    const int sect_0 = ((const int32_t *) op->op_params)[11];
    const int sect_1 = ((const int32_t *) op->op_params)[12];
    const int sect_2 = ((const int32_t *) op->op_params)[13];
    const int sect_3 = ((const int32_t *) op->op_params)[14];

    const int n_offs = ((const int32_t *) op->op_params)[15];

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const bool inplace = op->data == op->src[0]->data;

    ggml_metal_kargs_rope args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ nb00,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne0         =*/ ne0,
        /*.ne1         =*/ ne1,
        /*.ne2         =*/ ne2,
        /*.ne3         =*/ ne3,
        /*.nb0         =*/ nb0,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.n_past      =*/ n_past,
        /*.n_dims      =*/ n_dims,
        /*.n_offs      =*/ n_offs,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /* sect_0      =*/ sect_0,
        /* sect_1      =*/ sect_1,
        /* sect_2      =*/ sect_2,
        /* sect_3      =*/ sect_3,
        /* src2        =*/ op->src[2] != nullptr,
        /* inplace     =*/ inplace,
    };

    auto pipeline = ggml_metal_library_get_pipeline_rope(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_im2col(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];
    const int32_t p1 = ((const int32_t *)(op->op_params))[3];
    const int32_t d0 = ((const int32_t *)(op->op_params))[4];
    const int32_t d1 = ((const int32_t *)(op->op_params))[5];

    const bool is_2D = ((const int32_t *)(op->op_params))[6] == 1;

    const int32_t N  = op->src[1]->ne[is_2D ? 3 : 2];
    const int32_t IC = op->src[1]->ne[is_2D ? 2 : 1];
    const int32_t IH = is_2D ? op->src[1]->ne[1] : 1;
    const int32_t IW =         op->src[1]->ne[0];

    const int32_t KH = is_2D ? op->src[0]->ne[1] : 1;
    const int32_t KW =         op->src[0]->ne[0];

    const int32_t OH = is_2D ? op->ne[2] : 1;
    const int32_t OW =         op->ne[1];

    const int32_t CHW = IC * KH * KW;

    const uint64_t ofs0 = op->src[1]->nb[is_2D ? 3 : 2] / 4;
    const uint64_t ofs1 = op->src[1]->nb[is_2D ? 2 : 1] / 4;

    ggml_metal_kargs_im2col args = {
        /*.ofs0 =*/ ofs0,
        /*.ofs1 =*/ ofs1,
        /*.IW   =*/ IW,
        /*.IH   =*/ IH,
        /*.CHW  =*/ CHW,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
        /*.N    =*/ N,
        /*.KH   =*/ KH,
        /*.KW   =*/ KW,
        /*.KHW  =*/ KH * KW,
    };

    auto pipeline = ggml_metal_library_get_pipeline_im2col(lib, op);

    if (KH*KW <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        const uint64_t ntptg0 = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)/(KH*KW), N);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, ntptg0, KH, KW);
    } else {
        const uint64_t n_threads = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), N);
        const int64_t  quotient  = N / n_threads + (N % n_threads > 0 ? 1 : 0);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, quotient * CHW, OH, OW, n_threads, 1, 1);
    }

    return 1;
}

int ggml_metal_op_conv_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    const int32_t s1 = ((const int32_t *) op->op_params)[1];
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    const int32_t d0 = ((const int32_t *) op->op_params)[4];
    const int32_t d1 = ((const int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d args = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.IC   =*/ ne02,
        /*.OC   =*/ ne03,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d(lib, op);

    int nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    const uint64_t n_out = ggml_nelements(op);

    uint64_t tg = (n_out + nth - 1)/nth;
    tg = std::max<uint64_t>(tg, 1);
    tg = std::min<uint64_t>(tg, (uint64_t) std::numeric_limits<int>::max());

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_conv_2d_dw(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    const int32_t s1 = ((const int32_t *) op->op_params)[1];
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    const int32_t d0 = ((const int32_t *) op->op_params)[4];
    const int32_t d1 = ((const int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d_dw args = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.C    =*/ ne12,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne13,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    const bool use_tiled = (nb12 < nb10);

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d_dw(lib, op, use_tiled);

    int nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    const int32_t OW = ne0;
    const int32_t OH = ne1;
    const int32_t C  = ne12;
    const int32_t N  = ne13;

    const int tg_x = use_tiled ? (C + nth - 1) / nth : (OW + nth - 1) / nth;
    const int tg_y = OH;
    const int tg_z = use_tiled ? OW * N : C * N;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg_x, tg_y, tg_z, nth, 1, 1);

    return 1;
}

int ggml_metal_op_conv_3d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    // 1. Extract standard dimensions and byte strides
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // 2. Extract hyperparams from op_params
    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t s2 = ((const int32_t *)(op->op_params))[2];
    const int32_t p0 = ((const int32_t *)(op->op_params))[3];
    const int32_t p1 = ((const int32_t *)(op->op_params))[4];
    const int32_t p2 = ((const int32_t *)(op->op_params))[5];
    const int32_t d0 = ((const int32_t *)(op->op_params))[6];
    const int32_t d1 = ((const int32_t *)(op->op_params))[7];
    const int32_t d2 = ((const int32_t *)(op->op_params))[8];
    const int32_t IC = ((const int32_t *)(op->op_params))[9];
    const int32_t N  = ((const int32_t *)(op->op_params))[10];
    const int32_t OC = ((const int32_t *)(op->op_params))[11];

    // 3. Build the parameter struct using the macro-generated variables
    ggml_metal_kargs_conv_3d args = {
        /*.IW =*/ (int32_t)op->src[1]->ne[0],
        /*.IH =*/ (int32_t)op->src[1]->ne[1],
        /*.ID =*/ (int32_t)op->src[1]->ne[2],
        /*.OW =*/ (int32_t)op->ne[0],
        /*.OH =*/ (int32_t)op->ne[1],
        /*.OD =*/ (int32_t)op->ne[2],
        /*.KW =*/ (int32_t)op->src[0]->ne[0],
        /*.KH =*/ (int32_t)op->src[0]->ne[1],
        /*.KD =*/ (int32_t)op->src[0]->ne[2],
        s0, s1, s2,
        p0, p1, p2,
        d0, d1, d2,
        IC, N, OC,
        nb00, nb01, nb02, nb03, // Weight strides
        nb10, nb11, nb12, nb13, // Input strides
        nb0,  nb1,  nb2,  nb3   // Output strides
    };

    // 4. Fetch the JIT pipeline
    auto pipeline = ggml_metal_library_get_pipeline_conv_3d(lib, op);

    // 5. Grid mapping
    int nth0 = 32; // Standard SIMD width for Apple Silicon
    int nth1 = 1;
    int nth2 = 1;

    int64_t spatial_volume = args.OW * args.OH * args.OD;

    int ntg0 = (spatial_volume + nth0 - 1) / nth0;
    int ntg1 = args.OC;
    int ntg2 = args.N;

    // 6. Bind and Dispatch via the ggml C wrapper
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg0, ntg1, ntg2, nth0, nth1, nth2);

    return 1;
}

int ggml_metal_op_conv_transpose_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[1];
    const int32_t IL = op->src[1]->ne[0];

    const int32_t K  = op->src[0]->ne[0];

    const int32_t OL = op->ne[0];
    const int32_t OC = op->ne[1];

    ggml_metal_kargs_conv_transpose_1d args = {
        /*.IC  =*/ IC,
        /*.IL  =*/ IL,
        /*.K   =*/ K,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_1d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_col2im_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t OC = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];

    const int32_t K_OC  = (int32_t) op->src[0]->ne[0];
    const int32_t T_in  = (int32_t) op->src[0]->ne[1];
    const int32_t K     = K_OC / OC;
    const int32_t T_out = (int32_t) op->ne[0];

    ggml_metal_kargs_col2im_1d args = {
        /*.T_in  =*/ T_in,
        /*.T_out =*/ T_out,
        /*.OC    =*/ OC,
        /*.K     =*/ K,
        /*.K_OC  =*/ K_OC,
        /*.s0    =*/ s0,
        /*.p0    =*/ p0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_col2im_1d(lib, op);

    const int total = T_out * OC;
    const int nth   = 256;
    const int ntg   = (total + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

// Dispatch the fused snake kernel from the matched mul -> sin -> sqr -> mul -> add chain.
// idx points at the leading mul. The caller has validated the chain.
int ggml_metal_op_snake_fused(ggml_metal_op_t ctx, int idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * mul0 = ctx->node(idx + 0);
    const ggml_tensor * sqr  = ctx->node(idx + 2);
    const ggml_tensor * mul1 = ctx->node(idx + 3);
    ggml_tensor *       add  = ctx->node(idx + 4);

    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    const int T     = (int) x->ne[0];
    const int C     = (int) x->ne[1];
    const int total = T * C;

    // the encode loop pre-checked the leading mul only, check the rest of the chain
    for (int i = 1; i < 5; ++i) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);

            break;
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline_snake(lib, x->type);

    ggml_metal_kargs_snake args = {
        /*.T =*/ T,
        /*.C =*/ C,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),     1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(a),     2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(inv_b), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(add),   4);

    const int nth = 256;
    const int ntg = (total + nth - 1) / nth;
    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 5;
}

int ggml_metal_op_conv_transpose_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[2];
    const int32_t IH = op->src[1]->ne[1];
    const int32_t IW = op->src[1]->ne[0];

    const int32_t KH = op->src[0]->ne[1];
    const int32_t KW = op->src[0]->ne[0];

    const int32_t OW = op->ne[0];
    const int32_t OH = op->ne[1];
    const int32_t OC = op->ne[2];
    const int32_t N  = op->src[1]->ne[3];

    ggml_metal_kargs_conv_transpose_2d args = {
        /*.IC  =*/ IC,
        /*.IH  =*/ IH,
        /*.IW  =*/ IW,
        /*.KH  =*/ KH,
        /*.KW  =*/ KW,
        /*.OC  =*/ OC,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
        /*.nb2 =*/ nb2,
        /*.nb3 =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_2d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    // Metal requires buffer size to be multiple of 16 bytes
    const size_t smem = GGML_PAD(KW * KH * sizeof(float), 16);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC * N, KW, KH, 1);

    return 1;
}

int ggml_metal_op_upscale(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float sf0 = (float)ne0/op->src[0]->ne[0];
    float sf1 = (float)ne1/op->src[0]->ne[1];
    float sf2 = (float)ne2/op->src[0]->ne[2];
    float sf3 = (float)ne3/op->src[0]->ne[3];

    const int32_t mode_flags = ggml_get_op_params_i32(op, 0);

    float poffs = 0.5f;

    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        poffs = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    ggml_metal_kargs_upscale args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.sf0   =*/ sf0,
        /*.sf1   =*/ sf1,
        /*.sf2   =*/ sf2,
        /*.sf3   =*/ sf3,
        /*.poffs =*/ poffs,
    };

    auto pipeline = ggml_metal_library_get_pipeline_upscale(lib, op);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_roll(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ggml_get_op_params_i32(op, 0);
    const int32_t s1 = ggml_get_op_params_i32(op, 1);
    const int32_t s2 = ggml_get_op_params_i32(op, 2);
    const int32_t s3 = ggml_get_op_params_i32(op, 3);

    ggml_metal_kargs_roll args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.s2   =*/ s2,
        /*.s3   =*/ s3
    };

    auto pipeline = ggml_metal_library_get_pipeline_roll(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    const int nth_max = MIN(64, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int nth = MIN(args.ne0, nth_max);
    const int nk0 = (args.ne0 + 1024 - 1)/1024; // note: 1024 is hardcoded in the kernel!

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad_reflect_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad_reflect_1d args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.p0 =*/ ((const int32_t *)(op->op_params))[0],
        /*.p1 =*/ ((const int32_t *)(op->op_params))[1]
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad_reflect_1d(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_arange(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float start;
    float step;

    memcpy(&start, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&step,  ((const int32_t *) op->op_params) + 2, sizeof(float));

    ggml_metal_kargs_arange args = {
        /*.ne0   =*/ ne0,
        /*.start =*/ start,
        /*.step  =*/ step
    };

    const int nth = std::min(1024, ne0);

    auto pipeline = ggml_metal_library_get_pipeline_arange(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 1);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int dim        = op->op_params[0];
    const int max_period = op->op_params[1];

    ggml_metal_kargs_timestep_embedding args = {
        /*.nb1 =*/ nb1,
        /*.dim =*/ dim,
        /*.max_period =*/ max_period,
    };

    auto pipeline = ggml_metal_library_get_pipeline_timestep_embedding(lib, op);

    const int nth = std::max(1, std::min(1024, dim/2));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne00, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argmax(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_argmax args = {
        /*.ne00 = */ ne00,
        /*.nb01 = */ nb01,
    };

    auto pipeline = ggml_metal_library_get_pipeline_argmax(lib, op);

    const int64_t nrows = ggml_nrows(op->src[0]);

    int nth = ggml_metal_device_get_props(ctx->dev)->simd_width;
    while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
        nth *= 2;
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}


// the MoE router emits softmax -> argsort -> get_rows [-> sum_rows -> clamp -> div].
// ggml_can_fuse_ext cannot express it: the argsort output is also read by the later
// mul_mat_id, so it has two uses. Match it by hand like the CUDA backend does.
struct ggml_metal_topk_moe_plan {
    int n_fuse = 0;
    bool with_norm = false;
    float clamp = 0.0f;
    ggml_tensor * logits = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * weights = nullptr;
};

static ggml_tensor * ggml_metal_view_base(ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t;
}

static bool ggml_metal_topk_moe_same(ggml_tensor * a, ggml_tensor * b) {
    return a && b && ggml_metal_view_base(a) == ggml_metal_view_base(b);
}

static bool ggml_metal_topk_moe_fusion(ggml_metal_op_t ctx, int idx, ggml_metal_topk_moe_plan & plan) {
    // measured neutral on RDNA2 (pp512 +0.1%, tg128 -1%, both inside the noise), so the
    // dispatch saving does not pay for itself here; opt-in until a device shows a gain
    static const bool enabled = getenv("GGML_METAL_TOPK_MOE_ENABLE") != nullptr;
    if (!enabled || !ctx->use_fusion()) {
        return false;
    }

    if (idx + 2 >= ctx->n_nodes()) {
        return false;
    }

    ggml_tensor * softmax = ctx->node(idx + 0);
    ggml_tensor * argsort = ctx->node(idx + 1);
    ggml_tensor * getrows = ctx->node(idx + 2);

    if (softmax->op != GGML_OP_SOFT_MAX || argsort->op != GGML_OP_ARGSORT || getrows->op != GGML_OP_GET_ROWS) {
        return false;
    }

    // a mask or ALiBi slope is not part of the router chain
    if (softmax->src[1] || softmax->src[2]) {
        return false;
    }

    if (softmax->type != GGML_TYPE_F32 || softmax->src[0]->type != GGML_TYPE_F32 ||
        argsort->type != GGML_TYPE_I32 || getrows->type != GGML_TYPE_F32) {
        return false;
    }

    float scale = 1.0f;
    float bias  = 0.0f;
    memcpy(&scale, ((const int32_t *) softmax->op_params) + 0, sizeof(float));
    memcpy(&bias,  ((const int32_t *) softmax->op_params) + 1, sizeof(float));
    if (scale != 1.0f || bias != 0.0f) {
        return false;
    }

    if (!ggml_metal_topk_moe_same(argsort->src[0], softmax) ||
        !ggml_metal_topk_moe_same(getrows->src[0], softmax) ||
        !ggml_metal_topk_moe_same(getrows->src[1], argsort)) {
        return false;
    }

    if (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(softmax) ||
        !ggml_is_contiguous(argsort) || !ggml_is_contiguous(getrows)) {
        return false;
    }

    // one use means no normalisation follows and get_rows already is the weights;
    // the normalising router reads it twice, from sum_rows and from div
    if (ctx->has_n_uses(idx + 2, 1)) {
        plan.n_fuse  = 3;
        plan.weights = getrows;
    } else if (idx + 5 < ctx->n_nodes()) {
        ggml_tensor * sumrows = ctx->node(idx + 3);
        ggml_tensor * clamp   = ctx->node(idx + 4);
        ggml_tensor * div     = ctx->node(idx + 5);

        if (sumrows->op != GGML_OP_SUM_ROWS || clamp->op != GGML_OP_CLAMP || div->op != GGML_OP_DIV ||
            !ggml_metal_topk_moe_same(sumrows->src[0], getrows) ||
            !ggml_metal_topk_moe_same(clamp->src[0], sumrows) ||
            !ggml_metal_topk_moe_same(div->src[0], getrows) ||
            !ggml_metal_topk_moe_same(div->src[1], clamp) ||
            !ggml_is_contiguous(div) ||
            !ctx->has_n_uses(idx + 3, 1) ||
            !ctx->has_n_uses(idx + 4, 1)) {
            return false;
        }

        float cmin = 0.0f;
        float cmax = 0.0f;
        memcpy(&cmin, ((const int32_t *) clamp->op_params) + 0, sizeof(float));
        memcpy(&cmax, ((const int32_t *) clamp->op_params) + 1, sizeof(float));
        if (!std::isinf(cmax) || cmax < 0.0f) {
            return false;
        }

        plan.n_fuse    = 6;
        plan.with_norm = true;
        plan.clamp     = cmin;
        plan.weights   = div;
    } else {
        return false;
    }

    const int64_t n_expert = softmax->ne[0];
    if (n_expert > 512 || n_expert < 1) {
        return false;
    }

    plan.logits = softmax->src[0];
    plan.ids    = argsort;
    return true;
}

int ggml_metal_op_topk_moe(ggml_metal_op_t ctx, int idx) {
    ggml_metal_topk_moe_plan plan;
    if (!ggml_metal_topk_moe_fusion(ctx, idx, plan)) {
        return 0;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    auto pipeline = ggml_metal_library_get_pipeline_topk_moe(lib);
    if (!pipeline.pipeline) {
        return 0;
    }

    const int32_t n_expert      = plan.logits->ne[0];
    const int32_t n_expert_used = plan.weights->ne[1];
    const int32_t nrows         = ggml_nrows(plan.logits);

    int nth = 32;
    while (2*nth <= n_expert && 2*nth <= (int) ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    const size_t smem_f = GGML_PAD((2*n_expert + nth)*sizeof(float), 16);
    const size_t smem_i = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_kargs_topk_moe args = {
        /*.n_expert        =*/ n_expert,
        /*.n_expert_used   =*/ n_expert_used,
        /*.nrows           =*/ nrows,
        /*.mode            =*/ 0,
        /*.with_norm       =*/ plan.with_norm ? 1 : 0,
        /*.delayed_softmax =*/ 0,
        /*.has_bias        =*/ 0,
        /*.clamp           =*/ plan.clamp,
        /*.scale           =*/ 1.0f,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(plan.logits),  1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(plan.logits),  2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(plan.weights), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(plan.ids),     4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_f, 0);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_i, 1);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    // two outputs leave the fused range, and only the first node's dst is tracked
    ggml_metal_op_concurrency_reset(ctx);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 0) {
        GGML_LOG_DEBUG("%s: fused %d nodes into topk_moe (norm = %d)\n", __func__, plan.n_fuse, (int) plan.with_norm);
    }

    return plan.n_fuse;
}

int ggml_metal_op_argsort(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_argsort(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    const int npr = (ne00 + nth - 1)/nth;

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ nth,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_argsort_merge(lib, op);

    int len = nth;

    while (len < ne00) {
        ggml_metal_op_concurrency_reset(ctx);

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ ne00,
            /*.len   =*/ len,
        };

        // merges per row
        const int nm = (ne00 + 2*len - 1) / (2*len);

        const int nth = std::min(512, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge));

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }

    return 1;
}

// bitonic-sort + merge fallback: efficient when k is small and there are few rows,
// where the single-workgroup-per-row radix-select cannot reach enough parallelism
static void ggml_metal_op_top_k_bitonic(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_top_k(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    // blocks per row
    const int npr = (ne00 + nth - 1)/nth;

    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += sizeof(int32_t)*ggml_nelements(op->src[0]);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    const int top_k = ne0;

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ std::min(nth, top_k), // for each block, keep just the top_k indices
    };

    if (npr > 1) {
        args.ne0 = (npr - 1)*args.top_k + std::min(ne00 - (npr - 1)*nth, args.top_k);
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_top_k_merge(lib, op);

    int len = args.top_k;

    while (len < args.ne0) {
        ggml_metal_op_concurrency_reset(ctx);

        // merges per row
        const int nm = (args.ne0 + 2*len - 1) / (2*len);

        const int nth = std::min(512, std::min(len, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge)));

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ args.ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ nm == 1 ? top_k : args.ne0, // the final merge outputs top_k elements
            /*.len   =*/ len,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }
}

// radix-select: one workgroup per row. Maps each float to an order-preserving unsigned
// key, finds the k-th largest via 4 radix-8 histogram passes, then compacts the top-k
// indices. Fast for large k and/or many rows.
static void ggml_metal_op_top_k_radix(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);

    auto pipeline = ggml_metal_library_get_pipeline_top_k_radix(lib, op);

    // one workgroup per row; radix-select the k-th largest value
    const int nth = std::min(1024, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_kargs_top_k args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.top_k =*/ (int32_t) op->ne[0],
    };

    // shared memory: 256-entry histogram + bucket/above scalars + output counter
    const size_t smem_histo  = GGML_PAD(256*sizeof(uint32_t), 16);
    const size_t smem_bucket = GGML_PAD(    sizeof(uint32_t), 16);
    const size_t smem_above  = GGML_PAD(    sizeof(uint32_t), 16);
    const size_t smem_out    = GGML_PAD(    sizeof(uint32_t), 16);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_histo,  0);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_bucket, 1);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_above,  2);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_out,    3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
}

int ggml_metal_op_moe_reduce(ggml_metal_op_t ctx, int idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    int n_fuse = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
    if (!fusion || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_MOE_REDUCE) {
        return 1;
    }

    ggml_tensor * mul     = ctx->node(idx);
    ggml_tensor * experts = mul->src[0];
    ggml_tensor * weights = mul->src[1];
    ggml_tensor * dst     = ctx->node(idx + n_fuse - 1);

    ggml_metal_kargs_moe_reduce args = {
        /*.ne00 =*/ (int32_t) experts->ne[0],
        /*.ne02 =*/ (int32_t) experts->ne[2],
    };

    auto pipeline = ggml_metal_library_get_pipeline_moe_reduce(lib, (int32_t) experts->ne[1]);

    const int nth = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int n_col_tiles = (args.ne00 + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(experts), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),     3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (uint32_t) args.ne02, (uint32_t) n_col_tiles, 1, nth, 1, 1);

    ctx->count_fusions(fusion);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: MOE_REDUCE\n", __func__);
    }

    return n_fuse;
}


int ggml_metal_op_top_k(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    // radix-select has a fixed single-workgroup-per-row cost (~50-60us) that is only
    // amortized for long rows, many rows, or a large k; otherwise the bitonic path wins
    const int ncols = op->src[0]->ne[0];
    const int k     = op->ne[0];
    const int nrows = ggml_nrows(op->src[0]);

    const bool use_radix =
        ncols > 2048 && (k > 64 || (nrows > 4 && ncols >= 8192));

    if (use_radix) {
        ggml_metal_op_top_k_radix(ctx, idx);
    } else {
        ggml_metal_op_top_k_bitonic(ctx, idx);
    }

    return 1;
}

int ggml_metal_op_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_tri args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_tri(lib, op);

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_adamw(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_adamw(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_adamw args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_sgd(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_sgd(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_sgd args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_count_equal(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);

    {
        ggml_metal_kargs_memset args = { /*.val =*/ 0 };

        auto pipeline = ggml_metal_library_get_pipeline_memset(lib, op);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 1);

        ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);
    }

    ggml_metal_op_concurrency_reset(ctx);

    {
        ggml_metal_kargs_count_equal args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
        };

        auto pipeline = ggml_metal_library_get_pipeline_count_equal(lib, op);

        const size_t smem = pipeline.smem;

        const int nth = ggml_metal_device_get_props(ctx->dev)->simd_width*pipeline.nsg;

        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return 1;
}
