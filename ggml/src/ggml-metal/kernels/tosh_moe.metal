#include <metal_stdlib>
using namespace metal;

// Timestamp LRU over a bank of slots, one threadgroup, no host involvement: the routing
// arrives as expert ids and leaves as slot ids, or -1 for an expert this step will not
// fetch. Fixed shapes, so it can sit inside a captured graph.
struct tosh_lru_args {
    int32_t n_used;      // experts per token
    int32_t n_tokens;
    int32_t stride;      // ints between one token's ids and the next, which is not n_used
    int32_t n_expert;
    int32_t n_slots;
    int32_t n_fixed;
    int32_t max_fetch;   // cap per step; the rest stay non-resident
};

kernel void tosh_moe_lru(
        constant tosh_lru_args & args         [[buffer(0)]],
        device const int32_t   * ids          [[buffer(1)]],   // the routing, untouched
        device   int32_t       * ids_out      [[buffer(9)]],   // the same, as slots
        device const int32_t   * cold_for_id  [[buffer(10)]],
        device   int32_t       * slot_for_id  [[buffer(2)]],
        device   int32_t       * id_of_slot   [[buffer(3)]],
        device   int32_t       * usage        [[buffer(4)]],
        device   int32_t       * step         [[buffer(5)]],
        device   int32_t       * evict_slots  [[buffer(6)]],
        device   int32_t       * src_indices  [[buffer(7)]],
        device   int32_t       * n_indices    [[buffer(8)]],
        threadgroup int32_t    * shm          [[threadgroup(0)]],
        uint tid  [[thread_position_in_threadgroup]],
        uint ntg  [[threads_per_threadgroup]]) {

    threadgroup int32_t * active  = shm;                    // n_expert
    threadgroup int32_t * missing = shm + args.n_expert;     // n_expert
    threadgroup int32_t * red     = missing + args.n_expert; // 2*ntg, for the argmin

    int32_t now = 0;
    if (tid == 0) {
        now = *step + 1;
        *step = now;
        *n_indices = 0;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    now = *step;

    // which experts this step wants, and which of those are not here
    for (uint e = tid; e < (uint) args.n_expert; e += ntg) {
        active[e]  = 0;
        missing[e] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_active = (uint) (args.n_used*args.n_tokens);
    for (uint i = tid; i < n_active; i += ntg) {
        const uint t = i / (uint) args.n_used;
        const uint k = i % (uint) args.n_used;
        const int32_t e = ids[t*(uint) args.stride + k];
        if (e >= 0 && e < args.n_expert) {
            active[e] = 1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e < (uint) args.n_expert; e += ntg) {
        if (active[e] == 0) {
            continue;
        }
        const int32_t s = slot_for_id[e];
        if (s >= 0) {
            usage[s] = now;    // a hit keeps its place
        } else {
            missing[e] = 1;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // Serial over the fetches, which depend on each other, but the search for the least
    // recently used slot is a threadgroup reduction: on a big bank it is thousands of
    // comparisons and one thread would walk them alone.
    threadgroup int32_t * best_u = red;              // ntg
    threadgroup int32_t * best_s = red + 256;        // ntg

    int32_t fetched = 0;
    for (int32_t e = 0; e < args.n_expert && fetched < args.max_fetch; e++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (missing[e] == 0) {
            continue;
        }

        int32_t mine_u = INT_MAX;
        int32_t mine_s = -1;
        for (uint sl = (uint) args.n_fixed + tid; sl < (uint) args.n_slots; sl += ntg) {
            const int32_t owner = id_of_slot[sl];
            if (owner >= 0 && owner < args.n_expert && active[owner] != 0) {
                continue;          // in use this step
            }
            const int32_t u = usage[sl];
            if (mine_s < 0 || u < mine_u) {
                mine_u = u;
                mine_s = (int32_t) sl;
            }
        }
        best_u[tid] = mine_u;
        best_s[tid] = mine_s;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint span = ntg/2; span > 0; span >>= 1) {
            if (tid < span) {
                const int32_t os = best_s[tid + span];
                // ties go to the lower slot, so the result does not depend on the split
                if (os >= 0 && (best_s[tid] < 0 || best_u[tid + span] < best_u[tid] ||
                        (best_u[tid + span] == best_u[tid] && os < best_s[tid]))) {
                    best_u[tid] = best_u[tid + span];
                    best_s[tid] = os;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const int32_t victim = best_s[0];
        if (victim < 0) {
            break;                 // every slot is needed this step
        }

        if (tid == 0) {
            const int32_t old = id_of_slot[victim];
            if (old >= 0 && old < args.n_expert) {
                slot_for_id[old] = -1;
            }

            id_of_slot[victim]   = e;
            slot_for_id[e]       = victim;
            usage[victim]        = now;
            evict_slots[fetched] = victim;
            src_indices[fetched] = cold_for_id[e];
        }
        fetched++;
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        *n_indices = fetched;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // the routing is left alone; the slots go to their own place, because the ids are also
    // what the router's weights are gathered with
    for (uint i = tid; i < n_active; i += ntg) {
        const uint t = i / (uint) args.n_used;
        const uint k = i % (uint) args.n_used;
        const uint o = t*(uint) args.stride + k;
        const int32_t e = ids[o];
        ids_out[o] = (e >= 0 && e < args.n_expert) ? slot_for_id[e] : -1;
    }

}

// Brings the rows the LRU just scheduled from the host bank into their slots. Only the
// missing ones travel, so the window the device touches stays small no matter how big the
// bank is.
struct tosh_fetch_args {
    uint32_t row_u4;     // row size in uint4 units
    int32_t  max_fetch;
    int32_t  n_chunks;   // threadgroups per row: one is far too few for a whole expert
};

kernel void tosh_moe_fetch(
        constant tosh_fetch_args & args        [[buffer(0)]],
        device       uint4       * slots       [[buffer(1)]],
        device const uint4       * host        [[buffer(2)]],
        device const int32_t     * evict_slots [[buffer(3)]],
        device const int32_t     * src_indices [[buffer(4)]],
        device const int32_t     * n_indices   [[buffer(5)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tid   [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {

    const int32_t i     = (int32_t) (tgpig / (uint) args.n_chunks);
    const uint    chunk = tgpig % (uint) args.n_chunks;
    if (i >= *n_indices) {
        return;
    }

    const int32_t dst = evict_slots[i];
    const int32_t src = src_indices[i];
    if (dst < 0 || src < 0) {
        return;
    }

    device       uint4 * d = slots + (uint64_t) dst*args.row_u4;
    device const uint4 * s = host  + (uint64_t) src*args.row_u4;

    for (uint j = chunk*ntg + tid; j < args.row_u4; j += ntg*(uint) args.n_chunks) {
        d[j] = s[j];
    }
}

// Copies a whole bank from the host into the scratch in one contiguous pass, which is what a
// batch too wide for the cache needs: reading the experts scattered from host costs far more.
kernel void tosh_moe_stage(
        constant uint32_t    & n_u4  [[buffer(0)]],
        device       uint4   * dst   [[buffer(1)]],
        device const uint4   * src   [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tid   [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]],
        uint ntgs  [[threadgroups_per_grid]]) {

    for (uint i = tgpig*ntg + tid; i < n_u4; i += ntg*ntgs) {
        dst[i] = src[i];
    }
}

// Reconstructs one logical full bank in private scratch without ever wrapping the complete
// CPU bank as a Metal resource. Hot rows come from private resident slots; cold rows arrive
// packed in a bounded shared buffer.
struct tosh_assemble_args {
    uint32_t row_u4;
    uint32_t n_expert;
    int32_t  use_active;
};

kernel void tosh_moe_assemble(
        constant tosh_assemble_args & args        [[buffer(0)]],
        device       uint4          * dst         [[buffer(1)]],
        device const uint4          * cold        [[buffer(2)]],
        device const uint4          * slots       [[buffer(3)]],
        device const int32_t        * slot_for_id [[buffer(4)]],
        device const int32_t        * cold_for_id [[buffer(5)]],
        device const int32_t        * active      [[buffer(6)]],
        uint gid [[thread_position_in_grid]]) {
    const uint64_t total = (uint64_t) args.n_expert*args.row_u4;
    if ((uint64_t) gid >= total) return;
    const uint32_t expert = gid/args.row_u4;
    // A batch routes to far fewer experts than the bank holds, and a row nothing routes to is
    // never read: leaving it stale costs nothing and skipping it is most of the traffic.
    if (args.use_active != 0 && active[expert] == 0) return;
    const uint32_t col = gid - expert*args.row_u4;
    const int32_t slot = slot_for_id[expert];
    const int32_t ci = cold_for_id[expert];
    dst[gid] = slot >= 0 ? slots[(uint64_t) slot*args.row_u4 + col]
                         : cold[(uint64_t) ci*args.row_u4 + col];
}

// Marks the experts one batch routes to, so the assemble can skip the rest.
struct tosh_mark_args {
    int32_t n_used;
    int32_t n_tokens;
    int32_t stride;
    int32_t n_expert;
};

// One threadgroup: the clear has to be visible to the marking, and a barrier only spans a
// threadgroup. The work is a few thousand ids, so one is plenty.
kernel void tosh_moe_mark_active(
        constant tosh_mark_args & args   [[buffer(0)]],
        device const int32_t    * ids    [[buffer(1)]],
        device       int32_t    * active [[buffer(2)]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    for (uint e = tid; e < (uint) args.n_expert; e += ntg) {
        active[e] = 0;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    const uint n_active = (uint) (args.n_used*args.n_tokens);
    for (uint i = tid; i < n_active; i += ntg) {
        const uint t = i / (uint) args.n_used;
        const uint k = i % (uint) args.n_used;
        const int32_t e = ids[t*(uint) args.stride + k];
        if (e >= 0 && e < args.n_expert) {
            active[e] = 1;
        }
    }
}
