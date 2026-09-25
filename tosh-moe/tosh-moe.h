// Experimental MoE expert cache: maps an expert to a physical slot, nothing else.
// RAM keeps every expert, so an eviction only drops a copy. The unit is a whole expert of a
// layer, all of its tensors together, because a partial one cannot run.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct tosh_moe_cache;

// n_slots is per layer.
struct tosh_moe_cache * tosh_moe_cache_init(int n_layer, int n_expert, int n_slots);
struct tosh_moe_cache * tosh_moe_cache_init_fixed(int n_layer, int n_expert, int n_slots, int n_fixed);
void                    tosh_moe_cache_set_fixed(struct tosh_moe_cache * c, int il, const int32_t * experts, int n);
void                    tosh_moe_cache_free(struct tosh_moe_cache * c);

// Slot of an expert, or -1. Leaves the state alone so a caller can plan before committing.
int tosh_moe_cache_find(const struct tosh_moe_cache * c, int il, int expert);

// Pin a resident expert for this step: not evictable, and its recency moves to now.
int tosh_moe_cache_hold(struct tosh_moe_cache * c, int il, int expert);

// Give an expert a slot, evicting the least recently used one that is not held, and hold it.
// Returns -1 when every slot of the layer is held; the caller has to run it elsewhere.
int tosh_moe_cache_admit(struct tosh_moe_cache * c, int il, int expert, int * out_evicted);

// Ends the step: releases every hold and advances the recency clock.
void tosh_moe_cache_step(struct tosh_moe_cache * c);

// False on the first broken invariant, naming it in `why`.
bool tosh_moe_cache_check(const struct tosh_moe_cache * c, const char ** why);

struct tosh_moe_cache_stats {
    uint64_t steps;
    uint64_t lookups;
    uint64_t hits;
    uint64_t admits;
    uint64_t evictions;
    uint64_t starved;   // refused because every slot was held
};

void tosh_moe_cache_stats_get(const struct tosh_moe_cache * c, struct tosh_moe_cache_stats * out);

enum tosh_moe_mode {
    TOSH_MOE_OFF = 0,
    TOSH_MOE_CACHE,    // every expert on the device, a miss is filled before it runs
    TOSH_MOE_HYBRID,   // the device runs what it holds, the host runs the rest
};

enum tosh_moe_mode tosh_moe_mode(void);

// The backend registers which of its buffer types keeps weights in host memory, so the
// loader can tell a bank the user placed there from one that landed in VRAM.
void         tosh_moe_set_host_buft(const void * buft);
const void * tosh_moe_host_buft(void);

// Layout of the per-layer cache state, packed in one i32 tensor so the kernels can bind it
// at offsets instead of needing seven buffers of their own.
// Widest batch whose remapped routing still fits the state. The ids are padded to one entry
// per expert, so the room needed is tokens times experts, not tokens times experts used.
#define TOSH_MOE_MAX_TOKENS 512

static inline int tosh_moe_ids_room(int n_expert) {
    return TOSH_MOE_MAX_TOKENS*n_expert;
}

static inline int tosh_moe_state_ints(int n_expert, int n_slots, int max_fetch) {
    return 2*n_expert + 2*n_slots + 2 + 2*max_fetch + tosh_moe_ids_room(n_expert) + n_expert;
}
static inline int tosh_moe_off_slot_for_id(int n_expert, int n_slots, int max_fetch) { (void) n_expert; (void) n_slots; (void) max_fetch; return 0; }
static inline int tosh_moe_off_cold_for_id(int n_expert, int n_slots, int max_fetch) { (void) n_slots; (void) max_fetch; return n_expert; }
static inline int tosh_moe_off_id_of_slot (int n_expert, int n_slots, int max_fetch) { (void) n_slots; (void) max_fetch; return 2*n_expert; }
static inline int tosh_moe_off_usage      (int n_expert, int n_slots, int max_fetch) { (void) max_fetch; return 2*n_expert + n_slots; }
static inline int tosh_moe_off_step       (int n_expert, int n_slots, int max_fetch) { (void) max_fetch; return 2*n_expert + 2*n_slots; }
static inline int tosh_moe_off_n_indices  (int n_expert, int n_slots, int max_fetch) { (void) max_fetch; return 2*n_expert + 2*n_slots + 1; }
static inline int tosh_moe_off_evict      (int n_expert, int n_slots, int max_fetch) { (void) max_fetch; return 2*n_expert + 2*n_slots + 2; }
static inline int tosh_moe_off_src        (int n_expert, int n_slots, int max_fetch) { return 2*n_expert + 2*n_slots + 2 + max_fetch; }
static inline int tosh_moe_off_ids        (int n_expert, int n_slots, int max_fetch) { return 2*n_expert + 2*n_slots + 2 + 2*max_fetch; }
// Which experts the current batch routes to. Last, so every offset above keeps its place.
static inline int tosh_moe_off_active     (int n_expert, int n_slots, int max_fetch) { return 2*n_expert + 2*n_slots + 2 + 2*max_fetch + tosh_moe_ids_room(n_expert); }

// Slots per expert bank, 0 when the cache is off.
int tosh_moe_slots_want(void);

// Ties a bank of VRAM slots to the weights that stay in RAM. Keys are ggml tensors, kept as
// void so this stays independent of ggml.
void         tosh_moe_bind(const void * slots, const void * ram);
void         tosh_moe_bind_experts(const void * slots, int n_expert);
int          tosh_moe_experts_of(const void * slots);
void         tosh_moe_bind_fixed(const void * slots, int n_fixed);
int          tosh_moe_fixed_of(const void * slots);
const void * tosh_moe_ram_of(const void * slots);

// The other way round, for the paths that are handed the bank itself.
const void * tosh_moe_slots_of(const void * bank);

// One scratch in VRAM, the size of a single layer's bank, reused by every layer: a wide batch
// copies the bank there in one contiguous read and computes from the device.
void         tosh_moe_set_stage(const void * stage);
const void * tosh_moe_stage(void);

// The bank itself when the tensor is a set of slots, otherwise the tensor unchanged.
static inline void * tosh_moe_bank_ptr(void * slots) {
    const void * host = tosh_moe_ram_of(slots);
#ifdef __cplusplus
    return host ? const_cast<void *>(host) : slots;
#else
    return host ? (void *) host : slots;
#endif
}
void         tosh_moe_unbind_all(void);

void tosh_moe_layer_set(const void * tensor, int il, int n_expert);
int  tosh_moe_layer_of(const void * tensor);
void tosh_moe_hot_set(int il, const int32_t * experts, int n);
const int32_t * tosh_moe_hot_of(int il, int * n);
void tosh_moe_cold_set(int il, const int32_t * cold_for, int n);
int  tosh_moe_cold_of(int il, int expert);
void tosh_moe_observe(const void * tensor, const int32_t * ids, int n);
void tosh_moe_seen_seed(int il, const int32_t * experts, const uint32_t * counts, int n);
void tosh_moe_seen_dump(void);
const uint32_t * tosh_moe_seen_of(int il, int * n);
void tosh_moe_seen_clear(int il);

void         tosh_moe_bind_state(const void * slots, const void * state);

// The banks of a layer share one routing, so the pick runs once per layer and the fetch once
// per bank. True on the first bank of the layer this pass reaches: node indices climb within
// a pass, so one that does not climb means a new pass has started.
bool tosh_moe_should_route(const void * slots, int node_idx);
const void * tosh_moe_state_of(const void * slots);

int  tosh_moe_bind_count(void);
void tosh_moe_bind_at(int i, const void ** slots, const void ** ram);

// Residency of one bank: table[expert] is its slot, or -1 when it is not on the device. The
// device path computes the experts that have a slot and the host path the rest, so the two
// together cover every expert exactly once. Keyed by either tensor of the bank.
// Seeds every bank once the weights are in: the first slots take the first experts and the
// last slot is left at zero, which is where the table sends an expert the device does not hold.
void tosh_moe_seed_banks(void);


const int32_t * tosh_moe_table_of(const void * tensor, int * n_expert);
void            tosh_moe_table_set(const void * slots, const int32_t * table, int n_expert);

#ifdef __cplusplus
}
#endif
