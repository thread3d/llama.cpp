#include "tosh-moe.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <vector>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>

struct hot_history {
    std::vector<int32_t> ids;
    std::vector<uint32_t> counts;
};

static std::unordered_map<int, hot_history> load_hot_map() {
    std::unordered_map<int, hot_history> out;
    const char * path = getenv("TOSH_MOE_HOT_MAP");
    if (!path) return out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        int il = -1;
        if (!(row >> il)) continue;
        std::string token;
        bool legacy_ranking = true;
        while (row >> token) {
            const size_t colon = token.find(':');
            legacy_ranking = legacy_ranking && colon == std::string::npos;
            try {
                const int e = std::stoi(token.substr(0, colon));
                const uint32_t count = colon == std::string::npos ? 1u
                    : (uint32_t) std::max(1, std::stoi(token.substr(colon + 1)));
                out[il].ids.push_back(e);
                out[il].counts.push_back(count);
            } catch (...) {
                out.erase(il);
                break;
            }
        }
        const auto found = out.find(il);
        if (legacy_ranking && found != out.end()) {
            auto & counts = found->second.counts;
            for (size_t i = 0; i < counts.size(); ++i) counts[i] = (uint32_t) (counts.size() - i);
        }
    }
    return out;
}

// Every slot starts empty: the kernels expect -1 for "nowhere" and a zero clock.
void tosh_moe_seed_banks(void) {
    const int n = tosh_moe_bind_count();

    std::vector<int32_t> init;
    const auto hot_map = load_hot_map();

    for (int i = 0; i < n; i++) {
        const void * sv = nullptr;
        const void * hv = nullptr;
        tosh_moe_bind_at(i, &sv, &hv);

        auto * slots = (ggml_tensor *) sv;
        auto * host  = (ggml_tensor *) hv;
        auto * state = (ggml_tensor *) tosh_moe_state_of(sv);
        if (!slots || !host || !state || !state->buffer) {
            GGML_LOG_WARN("%s: an expert bank has no state, the cache stays off for it\n", __func__);
            continue;
        }

        const int n_expert  = tosh_moe_experts_of(slots);
        const int n_slots   = (int) slots->ne[2];
        const int n_fixed   = tosh_moe_fixed_of(slots);
        const int max_fetch = n_fixed > 0 ? n_slots - n_fixed : n_slots;
        const bool split_bank = host->ne[2] < n_expert;

        std::vector<int32_t> fixed_ids(n_fixed);
        std::iota(fixed_ids.begin(), fixed_ids.end(), 0);
        const int il = tosh_moe_layer_of(slots);
        const auto hit = hot_map.find(il);
        if (split_bank && hit != hot_map.end() && (int) hit->second.ids.size() >= n_fixed) {
            std::vector<uint8_t> used(n_expert, 0);
            bool valid = true;
            for (int s = 0; s < n_fixed; ++s) {
                const int e = hit->second.ids[s];
                if (e < 0 || e >= n_expert || used[e]) { valid = false; break; }
                used[e] = 1;
                fixed_ids[s] = e;
            }
            if (valid) {
                const size_t row = host->nb[2];
                std::vector<uint8_t> old_fixed((size_t) n_fixed*row);
                std::vector<uint8_t> old_cold((size_t) (n_expert - n_fixed)*row);
                std::vector<uint8_t> new_fixed(old_fixed.size());
                std::vector<uint8_t> new_cold(old_cold.size());
                ggml_backend_tensor_get(slots, old_fixed.data(), 0, old_fixed.size());
                memcpy(old_cold.data(), host->data, old_cold.size());
                const auto source = [&](int e) -> const uint8_t * {
                    return e < n_fixed ? old_fixed.data() + (size_t) e*row
                                       : old_cold.data() + (size_t) (e - n_fixed)*row;
                };
                for (int s = 0; s < n_fixed; ++s) {
                    memcpy(new_fixed.data() + (size_t) s*row, source(fixed_ids[s]), row);
                }
                int cold = 0;
                for (int e = 0; e < n_expert; ++e) {
                    if (!used[e]) memcpy(new_cold.data() + (size_t) cold++*row, source(e), row);
                }
                ggml_backend_tensor_set(slots, new_fixed.data(), 0, new_fixed.size());
                memcpy(host->data, new_cold.data(), new_cold.size());
            }
        }

        std::vector<int32_t> cold_for(n_expert, -1);
        std::vector<uint8_t> is_fixed(n_expert, 0);
        for (int s = 0; s < n_fixed; ++s) is_fixed[fixed_ids[s]] = 1;
        int cold = 0;
        for (int e = 0; e < n_expert; ++e) if (!is_fixed[e]) cold_for[e] = cold++;
        if (split_bank && il >= 0) {
            if (hit != hot_map.end()) {
                tosh_moe_seen_seed(il, hit->second.ids.data(), hit->second.counts.data(), (int) hit->second.ids.size());
            }
            tosh_moe_hot_set(il, fixed_ids.data(), n_fixed);
            tosh_moe_cold_set(il, cold_for.data(), n_expert);
        }

        init.assign(tosh_moe_state_ints(n_expert, n_slots, max_fetch), 0);

        for (int e = 0; e < n_expert; e++) {
            init[tosh_moe_off_slot_for_id(n_expert, n_slots, max_fetch) + e] = -1;
            init[tosh_moe_off_cold_for_id(n_expert, n_slots, max_fetch) + e] = split_bank ? cold_for[e] : e;
        }
        for (int s = 0; s < n_slots; s++) {
            init[tosh_moe_off_id_of_slot(n_expert, n_slots, max_fetch) + s] = -1;
        }

        // A CPU-owned bank is consumed through the scheduler's prefetch path during wide
        // batches. Do not seed it through Metal's partial tensor-set path: the first decode
        // token fills all K slots anyway, and avoiding the eager seed also avoids invalid
        // range lookups on discrete-GPU model buffers.
        const bool cpu_bank = getenv("TOSH_MOE_CPU_BANK") != nullptr;

        // start with the first experts already in, so a fresh cache is not one big miss
        if (split_bank) {
            for (int s = 0; s < n_fixed; s++) {
                const int e = fixed_ids[s];
                init[tosh_moe_off_slot_for_id(n_expert, n_slots, max_fetch) + e] = s;
                init[tosh_moe_off_cold_for_id(n_expert, n_slots, max_fetch) + e] = -1;
                init[tosh_moe_off_id_of_slot(n_expert, n_slots, max_fetch) + s] = e;
            }
        } else if (!cpu_bank && getenv("TOSH_MOE_NO_SEED") == nullptr) {
            const size_t row = host->nb[2];
            for (int s = 0; s < n_slots && s < n_expert; s++) {
                ggml_backend_tensor_set(slots, (const char *) host->data + (size_t) s*row, (size_t) s*row, row);
                init[tosh_moe_off_slot_for_id(n_expert, n_slots, max_fetch) + s] = s;
                init[tosh_moe_off_id_of_slot(n_expert, n_slots, max_fetch) + s] = s;
            }
        }

        ggml_backend_tensor_set(state, init.data(), 0, init.size()*sizeof(int32_t));
    }
}
