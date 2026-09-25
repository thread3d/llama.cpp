#include "tosh-moe.h"

#include <cstdlib>
#include <cstring>
#include <vector>

// Allocated once: a step must not allocate, or a long session creeps upwards.
struct tosh_moe_cache {
    int n_layer = 0;
    int n_expert = 0;
    int n_slots = 0;
    int n_fixed = 0;

    uint64_t tick = 1;   // advances once per step

    std::vector<int32_t>  slot_of;    // [layer][expert] -> slot, or -1
    std::vector<int32_t>  expert_of;  // [layer][slot]   -> expert, or -1
    std::vector<uint64_t> used_at;    // [layer][slot]   -> tick of last use
    std::vector<uint64_t> held_at;    // [layer][slot]   -> tick it is held for, 0 when free

    tosh_moe_cache_stats stats = {};
};

static inline size_t idx_e(const tosh_moe_cache * c, int il, int expert) {
    return (size_t) il*c->n_expert + expert;
}

static inline size_t idx_s(const tosh_moe_cache * c, int il, int slot) {
    return (size_t) il*c->n_slots + slot;
}

struct tosh_moe_cache * tosh_moe_cache_init_fixed(int n_layer, int n_expert, int n_slots, int n_fixed) {
    if (n_layer <= 0 || n_expert <= 0 || n_slots <= 0 || n_slots > n_expert ||
        n_fixed < 0 || n_fixed > n_slots) {
        return nullptr;
    }

    auto * c = new tosh_moe_cache();

    c->n_layer  = n_layer;
    c->n_expert = n_expert;
    c->n_slots  = n_slots;
    c->n_fixed  = n_fixed;

    c->slot_of  .assign((size_t) n_layer*n_expert, -1);
    c->expert_of.assign((size_t) n_layer*n_slots,  -1);
    c->used_at  .assign((size_t) n_layer*n_slots,   0);
    c->held_at  .assign((size_t) n_layer*n_slots,   0);

    // Split banks place the first K experts permanently in the first K GPU slots.
    // Seed that mapping once and keep only the trailing ring eligible for eviction.
    for (int il = 0; il < n_layer; ++il) {
        for (int e = 0; e < n_fixed; ++e) {
            c->slot_of[idx_e(c, il, e)] = e;
            c->expert_of[idx_s(c, il, e)] = e;
            c->used_at[idx_s(c, il, e)] = c->tick;
        }
    }

    return c;
}

struct tosh_moe_cache * tosh_moe_cache_init(int n_layer, int n_expert, int n_slots) {
    return tosh_moe_cache_init_fixed(n_layer, n_expert, n_slots, 0);
}

void tosh_moe_cache_set_fixed(struct tosh_moe_cache * c, int il, const int32_t * experts, int n) {
    if (!c || il < 0 || il >= c->n_layer || !experts || n != c->n_fixed) return;
    for (int e = 0; e < c->n_expert; ++e) c->slot_of[idx_e(c, il, e)] = -1;
    for (int s = 0; s < c->n_fixed; ++s) {
        c->expert_of[idx_s(c, il, s)] = -1;
        const int e = experts[s];
        if (e < 0 || e >= c->n_expert) continue;
        c->slot_of[idx_e(c, il, e)] = s;
        c->expert_of[idx_s(c, il, s)] = e;
        c->used_at[idx_s(c, il, s)] = c->tick;
    }
}

void tosh_moe_cache_free(struct tosh_moe_cache * c) {
    delete c;
}

int tosh_moe_cache_find(const struct tosh_moe_cache * c, int il, int expert) {
    if (!c || il < 0 || il >= c->n_layer || expert < 0 || expert >= c->n_expert) {
        return -1;
    }

    return c->slot_of[idx_e(c, il, expert)];
}

int tosh_moe_cache_hold(struct tosh_moe_cache * c, int il, int expert) {
    if (!c || il < 0 || il >= c->n_layer || expert < 0 || expert >= c->n_expert) {
        return -1;
    }

    c->stats.lookups++;

    const int slot = c->slot_of[idx_e(c, il, expert)];
    if (slot < 0) {
        return -1;
    }

    c->stats.hits++;

    c->used_at[idx_s(c, il, slot)] = c->tick;
    c->held_at[idx_s(c, il, slot)] = c->tick;

    return slot;
}

int tosh_moe_cache_admit(struct tosh_moe_cache * c, int il, int expert, int * out_evicted) {
    if (out_evicted) {
        *out_evicted = -1;
    }

    if (!c || il < 0 || il >= c->n_layer || expert < 0 || expert >= c->n_expert) {
        return -1;
    }

    // admitting a resident expert again would give it a second slot
    const int cur = c->slot_of[idx_e(c, il, expert)];
    if (cur >= 0) {
        return tosh_moe_cache_hold(c, il, expert);
    }

    // a free slot first, then the least recently used one that nothing holds
    int victim = -1;
    for (int s = c->n_fixed; s < c->n_slots; s++) {
        if (c->expert_of[idx_s(c, il, s)] < 0) {
            victim = s;
            break;
        }
    }

    if (victim < 0) {
        for (int s = c->n_fixed; s < c->n_slots; s++) {
            if (c->held_at[idx_s(c, il, s)] == c->tick) {
                continue;   // in use this step
            }
            if (victim < 0 || c->used_at[idx_s(c, il, s)] < c->used_at[idx_s(c, il, victim)]) {
                victim = s;
            }
        }
    }

    if (victim < 0) {
        c->stats.starved++;
        return -1;
    }

    const int old = c->expert_of[idx_s(c, il, victim)];
    if (old >= 0) {
        c->slot_of[idx_e(c, il, old)] = -1;
        c->stats.evictions++;
        if (out_evicted) {
            *out_evicted = old;
        }
    }

    c->expert_of[idx_s(c, il, victim)] = expert;
    c->slot_of  [idx_e(c, il, expert)] = victim;
    c->used_at  [idx_s(c, il, victim)] = c->tick;
    c->held_at  [idx_s(c, il, victim)] = c->tick;

    c->stats.admits++;

    return victim;
}

void tosh_moe_cache_step(struct tosh_moe_cache * c) {
    if (!c) {
        return;
    }

    // holds carry the tick they were taken on, so the clock releases them all at once
    c->tick++;
    c->stats.steps++;
}

bool tosh_moe_cache_check(const struct tosh_moe_cache * c, const char ** why) {
    const auto fail = [&](const char * msg) {
        if (why) {
            *why = msg;
        }
        return false;
    };

    if (!c) {
        return fail("no cache");
    }

    for (int il = 0; il < c->n_layer; il++) {
        for (int e = 0; e < c->n_expert; e++) {
            const int s = c->slot_of[idx_e(c, il, e)];
            if (s == -1) {
                continue;
            }
            if (s < 0 || s >= c->n_slots) {
                return fail("an expert points outside the slot range");
            }
            if (c->expert_of[idx_s(c, il, s)] != e) {
                return fail("an expert points at a slot that does not point back");
            }
        }

        for (int s = 0; s < c->n_slots; s++) {
            const int e = c->expert_of[idx_s(c, il, s)];
            if (e == -1) {
                continue;
            }
            if (e < 0 || e >= c->n_expert) {
                return fail("a slot points outside the expert range");
            }
            if (c->slot_of[idx_e(c, il, e)] != s) {
                return fail("a slot holds an expert that lives somewhere else");
            }
            if (c->held_at[idx_s(c, il, s)] > c->tick) {
                return fail("a slot is held for a step that has not happened");
            }
        }
    }

    return true;
}

void tosh_moe_cache_stats_get(const struct tosh_moe_cache * c, struct tosh_moe_cache_stats * out) {
    if (!c || !out) {
        return;
    }

    *out = c->stats;
}

#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_map>

enum tosh_moe_mode tosh_moe_mode(void) {
    static const enum tosh_moe_mode m = []() {
        const char * s = getenv("TOSH_MOE_MODE");
        if (!s || strcmp(s, "off") == 0) {
            return TOSH_MOE_OFF;
        }
        if (strcmp(s, "cache") == 0) {
            return TOSH_MOE_CACHE;
        }
        if (strcmp(s, "hybrid") == 0) {
            return TOSH_MOE_HYBRID;
        }
        return TOSH_MOE_OFF;
    }();

    return m;
}

int tosh_moe_slots_want(void) {
    static const int n = []() {
        if (tosh_moe_mode() == TOSH_MOE_OFF) {
            return 0;
        }
        const char * s = getenv("TOSH_MOE_SLOTS");
        const int v = s ? atoi(s) : 0;
        return v > 0 ? v : 0;
    }();

    return n;
}

// keyed by pointer, so it has to be dropped when a model goes away or a later load can hit a
// freed tensor that happens to land on the same address
static std::unordered_map<const void *, const void *> g_bind;
static std::unordered_map<const void *, const void *> g_slots_of;
static std::unordered_map<const void *, int> g_experts;
static std::unordered_map<const void *, int> g_fixed;
static const void * g_stage;
static std::unordered_map<const void *, int> g_layer_of;
static std::unordered_map<int, std::vector<uint32_t>> g_seen;   // layer -> uses per expert
static std::unordered_map<int, std::vector<int32_t>> g_hot;
static std::unordered_map<int, std::vector<int32_t>> g_cold_for;

static void tosh_moe_tables_clear(void);
static void tosh_moe_layers_clear(void);
static void tosh_moe_rounds_clear(void);
static void tosh_moe_state_clear(void);

void tosh_moe_bind(const void * slots, const void * ram) {
    g_bind[slots] = ram;
    g_slots_of[ram] = slots;
}

const void * tosh_moe_ram_of(const void * slots) {
    const auto it = g_bind.find(slots);
    return it == g_bind.end() ? nullptr : it->second;
}

void tosh_moe_bind_experts(const void * slots, int n_expert) {
    g_experts[slots] = n_expert;
}

int tosh_moe_experts_of(const void * slots) {
    const auto it = g_experts.find(slots);
    return it == g_experts.end() ? 0 : it->second;
}

void tosh_moe_bind_fixed(const void * slots, int n_fixed) {
    g_fixed[slots] = n_fixed;
}

int tosh_moe_fixed_of(const void * slots) {
    const auto it = g_fixed.find(slots);
    return it == g_fixed.end() ? 0 : it->second;
}

void tosh_moe_seen_dump(void) {
    if (const char * path = getenv("TOSH_MOE_HOT_MAP_OUT")) {
        const std::string temporary = std::string(path) + ".tmp";
        FILE * fp = fopen(temporary.c_str(), "w");
        if (fp) {
            const int want = std::max(1, getenv("TOSH_MOE_HOT_MAP_K") ? atoi(getenv("TOSH_MOE_HOT_MAP_K")) : 75);
            std::vector<int> order;
            for (const auto & kv : g_seen) {
                order.resize(kv.second.size());
                for (int e = 0; e < (int) order.size(); ++e) order[e] = e;
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                    return kv.second[a] > kv.second[b];
                });
                fprintf(fp, "%d", kv.first);
                for (int i = 0; i < want && i < (int) order.size(); ++i) {
                    const int e = order[i];
                    fprintf(fp, " %d:%u", e, kv.second[e]);
                }
                fputc('\n', fp);
            }
            const bool flushed = fflush(fp) == 0;
            const bool closed = fclose(fp) == 0;
            const bool ok = flushed && closed;
            if (ok) {
                if (rename(temporary.c_str(), path) != 0) remove(temporary.c_str());
            } else {
                remove(temporary.c_str());
            }
        }
    }
}

void tosh_moe_seen_seed(int il, const int32_t * experts, const uint32_t * counts, int n) {
    auto & seen = g_seen[il];
    if (seen.empty() || !std::all_of(seen.begin(), seen.end(), [](uint32_t value) { return value == 0; })) {
        return;
    }

    uint32_t peak = 1;
    for (int i = 0; i < n; ++i) peak = std::max(peak, counts[i]);

    // Carry history forward with bounded weight. New routing can replace stale habits
    // within a normal session, while short sessions no longer erase the accumulated map.
    constexpr uint32_t history_peak = 32;
    for (int i = 0; i < n; ++i) {
        const int e = experts[i];
        if (e < 0 || e >= (int) seen.size()) continue;
        const uint64_t scaled = (uint64_t) counts[i]*history_peak + peak - 1;
        seen[e] = std::max(1u, (uint32_t) (scaled/peak));
    }
}

void tosh_moe_unbind_all(void) {
    tosh_moe_seen_dump();
    g_bind.clear();
    g_slots_of.clear();
    g_experts.clear();
    g_fixed.clear();
    g_stage = nullptr;
    tosh_moe_state_clear();
    tosh_moe_layers_clear();
    tosh_moe_tables_clear();
}

struct tosh_moe_table {
    std::vector<int32_t> slot_of;
};

static std::unordered_map<const void *, tosh_moe_table> g_tables;

void tosh_moe_table_set(const void * slots, const int32_t * table, int n_expert) {
    auto & t = g_tables[slots];
    t.slot_of.assign(table, table + n_expert);

    // reachable from the RAM twin as well, so the host path finds it without another lookup
    const void * ram = tosh_moe_ram_of(slots);
    if (ram) {
        g_tables[ram].slot_of = t.slot_of;
    }
}

const int32_t * tosh_moe_table_of(const void * tensor, int * n_expert) {
    const auto it = g_tables.find(tensor);
    if (it == g_tables.end() || it->second.slot_of.empty()) {
        return nullptr;
    }

    if (n_expert) {
        *n_expert = (int) it->second.slot_of.size();
    }

    return it->second.slot_of.data();
}

static void tosh_moe_tables_clear(void) {
    g_tables.clear();
}

// A bank belongs to a layer, and the three of a layer move together: an expert that is only
// half resident cannot run.
void tosh_moe_layer_set(const void * tensor, int il, int n_expert) {
    g_layer_of[tensor] = il;

    auto & v = g_seen[il];
    if ((int) v.size() < n_expert) {
        v.assign(n_expert, 0);
    }
}

int tosh_moe_layer_of(const void * tensor) {
    const auto it = g_layer_of.find(tensor);
    return it == g_layer_of.end() ? -1 : it->second;
}

void tosh_moe_hot_set(int il, const int32_t * experts, int n) {
    g_hot[il].assign(experts, experts + n);

    // Preserve a small prior for the resident set loaded from the previous profile.
    // Real routing observations overtake it quickly, but a one-token session cannot
    // erase the whole ranking by leaving every unused expert tied at zero.
    auto & seen = g_seen[il];
    const bool empty = std::all_of(seen.begin(), seen.end(), [](uint32_t value) { return value == 0; });
    if (empty) {
        for (int i = 0; i < n; ++i) {
            const int e = experts[i];
            if (e >= 0 && e < (int) seen.size()) seen[e] = 2;
        }
    }
}

const int32_t * tosh_moe_hot_of(int il, int * n) {
    const auto it = g_hot.find(il);
    if (it == g_hot.end()) return nullptr;
    if (n) *n = (int) it->second.size();
    return it->second.data();
}

void tosh_moe_cold_set(int il, const int32_t * cold_for, int n) {
    g_cold_for[il].assign(cold_for, cold_for + n);
}

int tosh_moe_cold_of(int il, int expert) {
    const auto it = g_cold_for.find(il);
    if (it == g_cold_for.end() || expert < 0 || expert >= (int) it->second.size()) return -1;
    return it->second[expert];
}

void tosh_moe_observe(const void * tensor, const int32_t * ids, int n) {
    const int il = tosh_moe_layer_of(tensor);
    if (il < 0) {
        return;
    }

    auto & v = g_seen[il];
    for (int i = 0; i < n; i++) {
        const int32_t e = ids[i];
        if (e >= 0 && e < (int32_t) v.size()) {
            v[e]++;
        }
    }
}

const uint32_t * tosh_moe_seen_of(int il, int * n) {
    const auto it = g_seen.find(il);
    if (it == g_seen.end()) {
        return nullptr;
    }
    if (n) {
        *n = (int) it->second.size();
    }
    return it->second.data();
}

void tosh_moe_seen_clear(int il) {
    const auto it = g_seen.find(il);
    if (it != g_seen.end()) {
        std::fill(it->second.begin(), it->second.end(), 0u);
    }
}

int tosh_moe_bind_count(void) {
    return (int) g_bind.size();
}

void tosh_moe_bind_at(int i, const void ** slots, const void ** ram) {
    int k = 0;
    for (const auto & kv : g_bind) {
        if (k++ != i) {
            continue;
        }
        if (slots) { *slots = kv.first;  }
        if (ram)   { *ram   = kv.second; }
        return;
    }
}


static void tosh_moe_layers_clear(void) {
    g_layer_of.clear();
    g_seen.clear();
    g_hot.clear();
    g_cold_for.clear();
    tosh_moe_rounds_clear();
}

static const void * g_host_buft = nullptr;

void tosh_moe_set_host_buft(const void * buft) {
    g_host_buft = buft;
}

const void * tosh_moe_host_buft(void) {
    return g_host_buft;
}

static std::unordered_map<const void *, const void *> g_state;

void tosh_moe_bind_state(const void * slots, const void * state) {
    g_state[slots] = state;
}

const void * tosh_moe_state_of(const void * slots) {
    const auto it = g_state.find(slots);
    return it == g_state.end() ? nullptr : it->second;
}


static void tosh_moe_state_clear(void) {
    g_state.clear();
}

static std::unordered_map<int, int> g_round;      // layer -> last node index seen

bool tosh_moe_should_route(const void * slots, int node_idx) {
    const int il = tosh_moe_layer_of(slots);
    if (il < 0) {
        return false;
    }

    const auto it = g_round.find(il);
    const bool first = it == g_round.end() || node_idx <= it->second;

    g_round[il] = node_idx;

    return first;
}


static void tosh_moe_rounds_clear(void) {
    g_round.clear();
}

const void * tosh_moe_slots_of(const void * bank) {
    const auto it = g_slots_of.find(bank);
    return it == g_slots_of.end() ? nullptr : it->second;
}

void tosh_moe_set_stage(const void * stage) {
    g_stage = stage;
}

const void * tosh_moe_stage(void) {
    return g_stage;
}
