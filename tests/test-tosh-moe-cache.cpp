#include "tosh-moe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

#include <mach/mach.h>

// resident bytes, to catch a cache that allocates per step
static size_t rss_now() {
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &info, &count) != KERN_SUCCESS) {
        return 0;
    }
    return info.resident_size;
}

static int g_fail = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        printf("  FAIL %s\n", what);
        g_fail++;
    }
}

// Favours a few experts: uniform routing hides the cache, one hot expert flatters it.
struct zipf_router {
    std::mt19937 rng;
    std::vector<double> cdf;

    zipf_router(int n_expert, double s, uint32_t seed) : rng(seed) {
        double sum = 0.0;
        cdf.resize(n_expert);
        for (int i = 0; i < n_expert; i++) {
            sum += 1.0/pow(i + 1, s);
            cdf[i] = sum;
        }
        for (auto & v : cdf) {
            v /= sum;
        }
    }

    int draw() {
        const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        for (size_t i = 0; i < cdf.size(); i++) {
            if (u <= cdf[i]) {
                return (int) i;
            }
        }
        return (int) cdf.size() - 1;
    }

    void draw_set(int k, std::vector<int> & out) {
        out.clear();
        while ((int) out.size() < k) {
            const int e = draw();
            bool dup = false;
            for (int x : out) {
                dup = dup || x == e;
            }
            if (!dup) {
                out.push_back(e);
            }
        }
    }
};

static void test_rejects_nonsense() {
    printf("argumentos invalidos\n");

    check(tosh_moe_cache_init( 0, 32,  8) == nullptr, "acepta 0 capas");
    check(tosh_moe_cache_init(24,  0,  8) == nullptr, "acepta 0 expertos");
    check(tosh_moe_cache_init(24, 32,  0) == nullptr, "acepta 0 ranuras");
    check(tosh_moe_cache_init(24, 32, 33) == nullptr, "acepta mas ranuras que expertos");

    tosh_moe_cache * c = tosh_moe_cache_init(2, 8, 4);
    check(c != nullptr, "rechaza argumentos validos");

    check(tosh_moe_cache_find(c, -1, 0) == -1, "acepta una capa negativa");
    check(tosh_moe_cache_find(c,  2, 0) == -1, "acepta una capa fuera de rango");
    check(tosh_moe_cache_find(c,  0, 8) == -1, "acepta un experto fuera de rango");
    check(tosh_moe_cache_hold(c,  0, 8) == -1, "fija un experto fuera de rango");
    check(tosh_moe_cache_admit(c, 0, 8, nullptr) == -1, "admite un experto fuera de rango");

    tosh_moe_cache_free(c);
}

static void test_one_slot_one_expert() {
    printf("un experto, una ranura\n");

    const int n_layer = 3, n_expert = 16, n_slots = 4;
    tosh_moe_cache * c = tosh_moe_cache_init(n_layer, n_expert, n_slots);

    // admitting the same expert twice must not give it a second slot
    const int a = tosh_moe_cache_admit(c, 0, 5, nullptr);
    const int b = tosh_moe_cache_admit(c, 0, 5, nullptr);
    check(a >= 0 && a == b, "un experto ocupa dos ranuras");

    // the same expert id in another layer is a different expert
    const int d = tosh_moe_cache_admit(c, 1, 5, nullptr);
    check(d >= 0, "no admite el mismo id en otra capa");
    check(tosh_moe_cache_find(c, 0, 5) == a, "la capa 1 movio a la capa 0");

    const char * why = nullptr;
    check(tosh_moe_cache_check(c, &why), why ? why : "invariantes rotas");

    tosh_moe_cache_free(c);
}

static void test_held_slots_survive() {
    printf("una ranura en uso no se expulsa\n");

    const int n_slots = 4;
    tosh_moe_cache * c = tosh_moe_cache_init(1, 32, n_slots);

    // fill every slot and hold all of them within one step
    std::vector<int> slots;
    for (int e = 0; e < n_slots; e++) {
        slots.push_back(tosh_moe_cache_admit(c, 0, e, nullptr));
    }

    // a NEW expert cannot displace anything: every slot is in use this step
    int evicted = -2;
    const int s = tosh_moe_cache_admit(c, 0, 20, &evicted);
    check(s == -1, "expulso una ranura en uso");
    check(evicted == -1, "informo de una expulsion que no ocurrio");

    for (int e = 0; e < n_slots; e++) {
        check(tosh_moe_cache_find(c, 0, e) == slots[e], "un experto fijado cambio de ranura");
    }

    // once the step ends the holds are gone and the same admission works
    tosh_moe_cache_step(c);
    check(tosh_moe_cache_admit(c, 0, 31, &evicted) >= 0, "no admite tras terminar el paso");
    check(evicted >= 0, "no informo de la expulsion");

    const char * why = nullptr;
    check(tosh_moe_cache_check(c, &why), why ? why : "invariantes rotas");

    tosh_moe_cache_free(c);
}

static void test_evicts_least_recent() {
    printf("expulsa la menos usada\n");

    tosh_moe_cache * c = tosh_moe_cache_init(1, 8, 3);

    tosh_moe_cache_admit(c, 0, 0, nullptr); tosh_moe_cache_step(c);
    tosh_moe_cache_admit(c, 0, 1, nullptr); tosh_moe_cache_step(c);
    tosh_moe_cache_admit(c, 0, 2, nullptr); tosh_moe_cache_step(c);

    // touch 0 so that 1 becomes the oldest
    tosh_moe_cache_hold(c, 0, 0);
    tosh_moe_cache_step(c);

    int evicted = -1;
    tosh_moe_cache_admit(c, 0, 3, &evicted);
    check(evicted == 1, "expulso la equivocada");

    tosh_moe_cache_free(c);
}

// Long session of routing churn, invariants checked along the way.
static void test_long_session() {
    printf("sesion larga con ruteo realista\n");

    const int n_layer = 24, n_expert = 32, n_used = 4, n_slots = 8;
    const int n_steps = 20000;

    tosh_moe_cache * c = tosh_moe_cache_init(n_layer, n_expert, n_slots);
    zipf_router router(n_expert, 1.1, 1234);

    std::vector<int> pick;
    std::vector<int> slots_this_step;

    // let the vectors reach their final size before the baseline
    for (int warm = 0; warm < 64; warm++) {
        for (int il = 0; il < n_layer; il++) {
            router.draw_set(n_used, pick);
            for (int e : pick) {
                if (tosh_moe_cache_hold(c, il, e) < 0) {
                    tosh_moe_cache_admit(c, il, e, nullptr);
                }
            }
        }
        tosh_moe_cache_step(c);
    }
    const size_t rss0 = rss_now();

    for (int step = 0; step < n_steps; step++) {
        for (int il = 0; il < n_layer; il++) {
            router.draw_set(n_used, pick);

            slots_this_step.clear();

            for (int e : pick) {
                int slot = tosh_moe_cache_hold(c, il, e);
                if (slot < 0) {
                    slot = tosh_moe_cache_admit(c, il, e, nullptr);
                }

                // n_slots >= n_used, so a step can always place every expert it selected
                if (slot < 0) {
                    check(false, "se quedo sin ranuras con n_slots >= n_used");
                    break;
                }

                // two experts of the same step must never share a slot
                for (int s : slots_this_step) {
                    if (s == slot) {
                        check(false, "dos expertos del mismo paso comparten ranura");
                    }
                }
                slots_this_step.push_back(slot);

                check(tosh_moe_cache_find(c, il, e) == slot, "la tabla no coincide con lo devuelto");
            }
        }

        tosh_moe_cache_step(c);

        if ((step % 97) == 0) {
            const char * why = nullptr;
            if (!tosh_moe_cache_check(c, &why)) {
                check(false, why ? why : "invariantes rotas");
                break;
            }
        }
    }

    const char * why = nullptr;
    check(tosh_moe_cache_check(c, &why), why ? why : "invariantes rotas al final");

    const size_t rss1 = rss_now();
    printf("  memoria residente: %.2f MiB antes, %.2f MiB despues de %d pasos\n",
            rss0/1048576.0, rss1/1048576.0, n_steps);
    check(rss0 == 0 || rss1 <= rss0 + (256u << 10), "la memoria crece con los pasos");

    tosh_moe_cache_stats stats = {};
    tosh_moe_cache_stats_get(c, &stats);

    const double hit = stats.lookups ? 100.0*stats.hits/stats.lookups : 0.0;
    printf("  %llu pasos, %llu consultas, acierto %.1f%%, %llu admisiones, %llu expulsiones, %llu sin ranura\n",
            (unsigned long long) stats.steps, (unsigned long long) stats.lookups, hit,
            (unsigned long long) stats.admits, (unsigned long long) stats.evictions,
            (unsigned long long) stats.starved);

    check(stats.starved == 0, "hubo pasos sin ranura disponible");
    check(stats.evictions + (uint64_t) n_layer*n_slots >= stats.admits, "mas admisiones que ranuras y expulsiones");

    tosh_moe_cache_free(c);
}

// A cache the size of the bank must never evict.
static void test_full_cache_never_evicts() {
    printf("caché del tamaño del banco\n");

    const int n_expert = 16;
    tosh_moe_cache * c = tosh_moe_cache_init(1, n_expert, n_expert);
    zipf_router router(n_expert, 0.8, 99);

    std::vector<int> pick;
    for (int step = 0; step < 5000; step++) {
        router.draw_set(4, pick);
        for (int e : pick) {
            if (tosh_moe_cache_hold(c, 0, e) < 0) {
                tosh_moe_cache_admit(c, 0, e, nullptr);
            }
        }
        tosh_moe_cache_step(c);
    }

    tosh_moe_cache_stats stats = {};
    tosh_moe_cache_stats_get(c, &stats);
    check(stats.evictions == 0, "expulso con sitio de sobra");
    check(stats.admits <= (uint64_t) n_expert, "admitio un experto dos veces");

    tosh_moe_cache_free(c);
}

int main() {
    printf("tosh-moe: expert cache\n\n");

    test_rejects_nonsense();
    test_one_slot_one_expert();
    test_held_slots_survive();
    test_evicts_least_recent();
    test_full_cache_never_evicts();
    test_long_session();

    printf("\n%s\n", g_fail == 0 ? "todo correcto" : "HAY FALLOS");

    return g_fail == 0 ? 0 : 1;
}
