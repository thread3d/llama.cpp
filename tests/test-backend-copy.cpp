// concurrent copies between two devices of one backend
//
// a copy between two devices stages through host memory owned by the source device. that
//   memory was shared by every context on the device and used without a lock, so copies that
//   ran at the same time overwrote each other and the wrong data reached the destination
// each thread here has its own backends, as separate llama contexts do, and copies its own
//   pattern, so a copy that arrives with another thread's data is detected

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstdio>
#include <functional>
#include <thread>
#include <vector>

static const int    n_threads = 4;
static const int    n_copies  = 200;
static const size_t n_elem    = 16*1024; // 64 KiB, large enough to be staged

struct worker_state {
    ggml_backend_t backend_src = nullptr;
    ggml_backend_t backend_dst = nullptr;

    ggml_context_ptr ctx_src;
    ggml_context_ptr ctx_dst;

    ggml_backend_buffer_ptr buf_src;
    ggml_backend_buffer_ptr buf_dst;

    ggml_tensor * src = nullptr;
    ggml_tensor * dst = nullptr;

    float value = 0.0f;

    int n_bad = 0;
};

static ggml_tensor * new_tensor(ggml_context_ptr & ctx, ggml_backend_t backend, ggml_backend_buffer_ptr & buf) {
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ctx.reset(ggml_init(params));
    if (!ctx) {
        return nullptr;
    }

    ggml_tensor * t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_elem);

    buf.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) {
        return nullptr;
    }

    return t;
}

static bool worker_init(worker_state & st, ggml_backend_dev_t dev_src, ggml_backend_dev_t dev_dst, float value) {
    st.backend_src = ggml_backend_dev_init(dev_src, nullptr);
    st.backend_dst = ggml_backend_dev_init(dev_dst, nullptr);

    if (st.backend_src == nullptr || st.backend_dst == nullptr) {
        fprintf(stderr, "failed to init the backends\n");
        return false;
    }

    st.src = new_tensor(st.ctx_src, st.backend_src, st.buf_src);
    st.dst = new_tensor(st.ctx_dst, st.backend_dst, st.buf_dst);

    if (st.src == nullptr || st.dst == nullptr) {
        fprintf(stderr, "failed to allocate the tensors\n");
        return false;
    }

    st.value = value;

    std::vector<float> data(n_elem, value);
    ggml_backend_tensor_set(st.src, data.data(), 0, ggml_nbytes(st.src));

    return true;
}

static void worker_run(worker_state & st) {
    std::vector<float> data(n_elem);

    for (int i = 0; i < n_copies; i++) {
        ggml_backend_tensor_copy(st.src, st.dst);
        ggml_backend_tensor_get(st.dst, data.data(), 0, ggml_nbytes(st.dst));

        for (float v : data) {
            if (v != st.value) {
                st.n_bad++;
                break;
            }
        }
    }
}

static bool same_backend(ggml_backend_dev_t a, ggml_backend_dev_t b) {
    return ggml_backend_dev_backend_reg(a) == ggml_backend_dev_backend_reg(b) &&
           ggml_backend_dev_buffer_type(a) != ggml_backend_dev_buffer_type(b);
}

int main() {
    ggml_backend_load_all();

    // two devices of one backend: the copy between them is the one that stages in host memory
    ggml_backend_dev_t dev_src = nullptr;
    ggml_backend_dev_t dev_dst = nullptr;

    const size_t n_dev = ggml_backend_dev_count();

    for (size_t i = 0; i < n_dev && dev_src == nullptr; i++) {
        for (size_t j = i + 1; j < n_dev; j++) {
            ggml_backend_dev_t a = ggml_backend_dev_get(i);
            ggml_backend_dev_t b = ggml_backend_dev_get(j);

            if (same_backend(a, b)) {
                dev_src = a;
                dev_dst = b;
                break;
            }
        }
    }

    if (dev_src == nullptr) {
        printf("SKIPPED: no backend has two devices\n");
        return 0;
    }

    printf("%s -> %s, %d threads x %d copies of %zu KiB\n",
            ggml_backend_dev_name(dev_src), ggml_backend_dev_name(dev_dst),
            n_threads, n_copies, n_elem*sizeof(float)/1024);

    std::vector<worker_state> states(n_threads);

    for (int t = 0; t < n_threads; t++) {
        if (!worker_init(states[t], dev_src, dev_dst, (float) (t + 1))) {
            return 1;
        }
    }

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back(worker_run, std::ref(states[t]));
    }
    for (auto & th : threads) {
        th.join();
    }

    int n_bad = 0;
    for (const auto & st : states) {
        n_bad += st.n_bad;
    }

    const int n_total = n_threads*n_copies;

    printf("%d/%d copies arrived intact\n", n_total - n_bad, n_total);

    for (auto & st : states) {
        ggml_backend_buffer_free(st.buf_src.release());
        ggml_backend_buffer_free(st.buf_dst.release());
        ggml_backend_free(st.backend_src);
        ggml_backend_free(st.backend_dst);
    }

    return n_bad == 0 ? 0 : 1;
}
