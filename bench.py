#!/usr/bin/env python3
"""
Decode/prefill benchmark for the phase-aware memory performance model.

Measures, on one CUDA GPU:
  1. micro   - effective bandwidth (copy, GEMV) and FP16 GEMM throughput, plus a
               GEMV batch sweep that traces the dense-layer roofline;
  2. attn    - a single decode-attention layer over a KV cache of length S
               (isolates KV-cache traffic, used to estimate the reuse factor rho);
  3. decode  - full decode step latency for synthetic transformer models over a
               grid of (batch B, context S), captured in CUDA graphs so that
               Python/kernel-launch overhead is removed;
  4. eager   - the same decode step without CUDA graphs (framework overhead);
  5. prefill - prefill latency versus prompt length.

Weights and caches are random FP16 tensors: latency depends on tensor shapes,
not on trained values, so no model download is needed.

Usage:
  python bench.py --smoke          # ~2 minutes, sanity check
  python bench.py                  # full run (~20-40 minutes)
Results go to results/<timestamp>/ (CSV + JSON). Send that folder back.
"""
import argparse, csv, datetime, json, math, os, platform, subprocess, sys, time

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Model geometries (embedding / LM head excluded, as in the analytical model)
# --------------------------------------------------------------------------
MODELS = {
    # name:        L,   D,    H,  Hkv, Dh,  ffn ratio
    "M1-MHA":  dict(L=24, D=1024, H=16, Hkv=16, Dh=64,  r=4),   # paper baseline
    "M1-GQA4": dict(L=24, D=1024, H=16, Hkv=4,  Dh=64,  r=4),   # same size, 4 KV heads
    "M1-MQA":  dict(L=24, D=1024, H=16, Hkv=1,  Dh=64,  r=4),   # same size, 1 KV head
    "M2-GQA4": dict(L=24, D=2048, H=16, Hkv=4,  Dh=128, r=4),   # ~1.1B params, GQA
}
CONTEXTS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BATCHES = [1, 4, 16, 64, 128, 256]
DTYPE = torch.float16
BYTES = 2


def weight_elems_per_layer(m):
    D, Hkv, Dh, r = m["D"], m["Hkv"], m["Dh"], m["r"]
    return 2 * D * D + 2 * D * Hkv * Dh + 2 * r * D * D   # Wq, Wo, Wk, Wv, W1, W2


def kv_bytes_per_token(m):
    return 2 * m["L"] * m["Hkv"] * m["Dh"] * BYTES


# --------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------
def sync():
    torch.cuda.synchronize()


def time_fn(fn, iters, trials=5, warmup=3):
    """Median over trials of mean ms per call (CUDA events)."""
    for _ in range(warmup):
        fn()
    sync()
    res = []
    for _ in range(trials):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        sync()
        res.append(s.elapsed_time(e) / iters)
    res.sort()
    return res[len(res) // 2], res[0], res[-1]


def time_auto(fn, target_ms=60.0, trials=5):
    """Pick the iteration count so each trial lasts about target_ms."""
    for _ in range(3):
        fn()
    sync()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); sync()
    one = max(s.elapsed_time(e), 1e-3)
    return time_fn(fn, iters=max(3, min(500, int(target_ms / one))), trials=trials, warmup=1)


def capture_graph(fn):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    sync()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    sync()
    return g


def free_bytes():
    f, _ = torch.cuda.mem_get_info()
    return f


def iters_for(ms_guess):
    return max(5, min(200, int(200.0 / max(ms_guess, 1e-3))))


# --------------------------------------------------------------------------
# Synthetic transformer
# --------------------------------------------------------------------------
class Model:
    def __init__(self, m, dev):
        self.m = m
        D, H, Hkv, Dh, r, L = m["D"], m["H"], m["Hkv"], m["Dh"], m["r"], m["L"]
        assert H * Dh == D and H % Hkv == 0
        g = lambda *s: (torch.randn(*s, device=dev, dtype=DTYPE) * 0.02)
        self.layers = []
        for _ in range(L):
            self.layers.append(dict(
                wq=g(D, D), wk=g(Hkv * Dh, D), wv=g(Hkv * Dh, D), wo=g(D, D),
                w1=g(r * D, D), w2=g(D, r * D),
                ln1=torch.ones(D, device=dev, dtype=DTYPE),
                ln2=torch.ones(D, device=dev, dtype=DTYPE)))
        self.scale = 1.0 / math.sqrt(Dh)

    # ---- one decode step: x (B, D), caches K/V lists of (B, Hkv, Smax, Dh)
    def decode_step(self, x, Kc, Vc, pos, S):
        m = self.m
        B, D, H, Hkv, Dh = x.shape[0], m["D"], m["H"], m["Hkv"], m["Dh"]
        G = H // Hkv
        for l, w in enumerate(self.layers):
            h = F.layer_norm(x, (D,), w["ln1"])
            q = F.linear(h, w["wq"]).view(B, Hkv, G, Dh)
            k = F.linear(h, w["wk"]).view(B, Hkv, Dh)
            v = F.linear(h, w["wv"]).view(B, Hkv, Dh)
            Kc[l][:, :, pos].copy_(k)
            Vc[l][:, :, pos].copy_(v)
            Kl = Kc[l][:, :, :S]                      # (B, Hkv, S, Dh), reads K once
            Vl = Vc[l][:, :, :S]
            sc = torch.matmul(q, Kl.transpose(-1, -2)) * self.scale   # (B,Hkv,G,S)
            p = torch.softmax(sc, dim=-1)
            o = torch.matmul(p, Vl).reshape(B, D)     # reads V once
            x = x + F.linear(o, w["wo"])
            h = F.layer_norm(x, (D,), w["ln2"])
            x = x + F.linear(F.gelu(F.linear(h, w["w1"])), w["w2"])
        return x

    # ---- decode step with attention removed (dense operators only)
    def dense_step(self, x):
        m = self.m
        B, D = x.shape[0], m["D"]
        for w in self.layers:
            h = F.layer_norm(x, (D,), w["ln1"])
            q = F.linear(h, w["wq"])
            F.linear(h, w["wk"])
            F.linear(h, w["wv"])
            x = x + F.linear(q, w["wo"])          # q stands in for the attention output
            h = F.layer_norm(x, (D,), w["ln2"])
            x = x + F.linear(F.gelu(F.linear(h, w["w1"])), w["w2"])
        return x

    # ---- attention operators only, all L layers (cache write + QK^T + softmax + PV)
    def attn_step(self, q, k, v, Kc, Vc, pos, S):
        m = self.m
        B, D = q.shape[0], m["D"]
        outs = []
        for l in range(m["L"]):
            Kc[l][:, :, pos].copy_(k)
            Vc[l][:, :, pos].copy_(v)
            sc = torch.matmul(q, Kc[l][:, :, :S].transpose(-1, -2)) * self.scale
            outs.append(torch.matmul(torch.softmax(sc, dim=-1), Vc[l][:, :, :S]).reshape(B, D))
        return outs[-1]

    # ---- prefill: x (B, S, D)
    def prefill(self, x):
        m = self.m
        B, S, D = x.shape
        H, Hkv, Dh = m["H"], m["Hkv"], m["Dh"]
        for w in self.layers:
            h = F.layer_norm(x, (D,), w["ln1"])
            q = F.linear(h, w["wq"]).view(B, S, H, Dh).transpose(1, 2)
            k = F.linear(h, w["wk"]).view(B, S, Hkv, Dh).transpose(1, 2)
            v = F.linear(h, w["wv"]).view(B, S, Hkv, Dh).transpose(1, 2)
            if Hkv != H:
                k = k.repeat_interleave(H // Hkv, dim=1)
                v = v.repeat_interleave(H // Hkv, dim=1)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + F.linear(o.transpose(1, 2).reshape(B, S, D), w["wo"])
            h = F.layer_norm(x, (D,), w["ln2"])
            x = x + F.linear(F.gelu(F.linear(h, w["w1"])), w["w2"])
        return x


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
class Writer:
    def __init__(self, path, fields):
        self.path, self.fields = path, fields
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        if new:
            self.w.writeheader()

    def row(self, **kw):
        self.w.writerow(kw)
        self.f.flush()


def nvsmi():
    q = "name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu,pstate"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return dict(zip(q.split(","), [s.strip() for s in out.split(",")]))
    except Exception as e:
        return {"error": str(e)}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def run_micro(out, smoke):
    dev = "cuda"
    res = {}
    n = (1 << 28) if not smoke else (1 << 26)          # 512 MiB (64 MiB smoke) fp16
    a = torch.empty(n, device=dev, dtype=DTYPE).normal_()
    b = torch.empty_like(a)
    ms, lo, hi = time_fn(lambda: b.copy_(a), iters=20)
    res["copy_GBps"] = 2 * n * BYTES / (ms * 1e-3) / 1e9
    del a, b

    N = 16384 if not smoke else 8192
    W = torch.randn(N, N, device=dev, dtype=DTYPE) * 0.01
    sweep = []
    for Bt in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        if smoke and Bt > 64:
            break
        x = torch.randn(Bt, N, device=dev, dtype=DTYPE)
        ms, lo, hi = time_fn(lambda: F.linear(x, W), iters=20)
        macs = Bt * N * N
        byts = (N * N + 2 * Bt * N) * BYTES
        sweep.append(dict(B=Bt, ms=ms, TMACs=macs / (ms * 1e-3) / 1e12,
                          GBps=byts / (ms * 1e-3) / 1e9))
        log(f"  GEMV/GEMM N={N} B={Bt:5d}: {ms:.3f} ms  {sweep[-1]['TMACs']:.2f} TMAC/s  {sweep[-1]['GBps']:.0f} GB/s")
    res["gemv_sweep"] = sweep
    res["gemv_GBps"] = sweep[0]["GBps"]
    del W

    best = 0.0
    for M in ([4096, 8192] if not smoke else [4096]):
        A = torch.randn(M, M, device=dev, dtype=DTYPE)
        Bm = torch.randn(M, M, device=dev, dtype=DTYPE)
        ms, lo, hi = time_fn(lambda: A @ Bm, iters=10)
        t = M ** 3 / (ms * 1e-3) / 1e12
        res[f"gemm{M}_TMACs"] = t
        best = max(best, t)
        del A, Bm
    res["gemm_TMACs"] = best
    torch.cuda.empty_cache()
    log(f"micro: copy {res['copy_GBps']:.0f} GB/s, GEMV {res['gemv_GBps']:.0f} GB/s, GEMM {best:.1f} TMAC/s")
    json.dump(res, open(os.path.join(out, "micro.json"), "w"), indent=2)
    return res


def run_attn(out, smoke):
    """Single-layer decode attention (QK^T, softmax, PV) over a cache of length S."""
    dev = "cuda"
    wr = Writer(os.path.join(out, "attn.csv"),
                ["Hkv", "Dh", "H", "B", "S", "ms", "ms_min", "ms_max", "kv_bytes", "GBps"])
    H, Dh = 16, 64
    ctxs = [512, 2048, 8192, 32768] if not smoke else [512, 2048]
    for Hkv in [16, 4, 1]:
        G = H // Hkv
        for B in [1, 16]:
            for S in ctxs:
                kvb = 2 * B * Hkv * S * Dh * BYTES
                if kvb > 0.6 * free_bytes():
                    continue
                K = torch.randn(B, Hkv, S, Dh, device=dev, dtype=DTYPE)
                V = torch.randn_like(K)
                q = torch.randn(B, Hkv, G, Dh, device=dev, dtype=DTYPE)

                def f():
                    p = torch.softmax(torch.matmul(q, K.transpose(-1, -2)) * 0.125, dim=-1)
                    return torch.matmul(p, V)
                g = capture_graph(f)
                ms, lo, hi = time_auto(g.replay)
                wr.row(Hkv=Hkv, Dh=Dh, H=H, B=B, S=S, ms=ms, ms_min=lo, ms_max=hi,
                       kv_bytes=kvb, GBps=kvb / (ms * 1e-3) / 1e9)
                log(f"  attn Hkv={Hkv:2d} B={B:3d} S={S:6d}: {ms:.4f} ms  {kvb / (ms * 1e-3) / 1e9:.0f} GB/s")
                del g, K, V, q
                torch.cuda.empty_cache()


def run_decode(out, smoke, models, eager_only=False):
    dev = "cuda"
    wr = Writer(os.path.join(out, "eager.csv" if eager_only else "decode.csv"),
                ["model", "L", "D", "H", "Hkv", "Dh", "r", "B", "S", "ms", "ms_min", "ms_max", "mode"])
    ctxs = CONTEXTS if not smoke else [128, 1024]
    bats = BATCHES if not smoke else [1, 16]
    if eager_only:
        bats = [1]
        ctxs = [128, 1024, 4096] if not smoke else [128]
    for name in models:
        m = MODELS[name]
        log(f"model {name}: {m}  clocks: {nvsmi().get('clocks.sm', '?')} MHz")
        model = Model(m, dev)
        sync()
        for B in bats:
            budget = free_bytes() - 1.0e9                    # keep 1 GB headroom
            per_tok = kv_bytes_per_token(m) * B
            ok = [S for S in ctxs if (S + 1) * per_tok < 0.92 * budget]
            if not ok:
                continue
            x = torch.randn(B, m["D"], device=dev, dtype=DTYPE)
            for S in ok:
                # Cache holds exactly S+1 positions: S past tokens plus the new one,
                # so attention reads contiguous K/V tensors of length S+1.
                try:
                    Kc = [torch.randn(B, m["Hkv"], S + 1, m["Dh"], device=dev, dtype=DTYPE) for _ in range(m["L"])]
                    Vc = [torch.randn(B, m["Hkv"], S + 1, m["Dh"], device=dev, dtype=DTYPE) for _ in range(m["L"])]
                    fn = lambda: model.decode_step(x, Kc, Vc, S, S + 1)
                    if eager_only:
                        ms, lo, hi = time_auto(fn)
                    else:
                        g = capture_graph(fn)
                        ms, lo, hi = time_auto(g.replay)
                        del g
                except torch.cuda.OutOfMemoryError:
                    Kc = Vc = None
                    torch.cuda.empty_cache()
                    log(f"  OOM B={B} S={S}; skipped")
                    continue
                wr.row(model=name, B=B, S=S, ms=ms, ms_min=lo, ms_max=hi,
                       mode="eager" if eager_only else "graph", **m)
                log(f"  {'eager' if eager_only else 'graph'} {name} B={B:3d} S={S:6d}: {ms:.3f} ms")
                del Kc, Vc
                torch.cuda.empty_cache()
            del x
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()


def run_ops(out, smoke, models):
    """Operator-level calibration: dense-only and attention-only stacks of all
    L layers, over the same (B, S) grid as the decode sweep."""
    dev = "cuda"
    wd = Writer(os.path.join(out, "ops_dense.csv"),
                ["model", "L", "D", "H", "Hkv", "Dh", "r", "B", "ms", "ms_min", "ms_max"])
    wa = Writer(os.path.join(out, "ops_attn.csv"),
                ["model", "L", "D", "H", "Hkv", "Dh", "r", "B", "S", "kv_bytes", "ms", "ms_min", "ms_max"])
    bats = BATCHES if not smoke else [1, 16]
    ctxs = [256, 1024, 4096, 16384] if not smoke else [256, 1024]
    for name in models:
        m = MODELS[name]
        log(f"ops {name}: clocks {nvsmi().get('clocks.sm', '?')} MHz")
        model = Model(m, dev)
        G = m["H"] // m["Hkv"]
        for B in bats:
            x = torch.randn(B, m["D"], device=dev, dtype=DTYPE)
            g = capture_graph(lambda: model.dense_step(x))
            ms, lo, hi = time_auto(g.replay)
            del g
            wd.row(model=name, B=B, ms=ms, ms_min=lo, ms_max=hi, **m)
            log(f"  dense {name} B={B:3d}: {ms:.3f} ms")
            q = torch.randn(B, m["Hkv"], G, m["Dh"], device=dev, dtype=DTYPE)
            k = torch.randn(B, m["Hkv"], m["Dh"], device=dev, dtype=DTYPE)
            v = torch.randn_like(k)
            budget = free_bytes() - 1.0e9
            for S in ctxs:
                kvb = (S + 1) * kv_bytes_per_token(m) * B
                if kvb > 0.9 * budget:
                    continue
                try:
                    Kc = [torch.randn(B, m["Hkv"], S + 1, m["Dh"], device=dev, dtype=DTYPE) for _ in range(m["L"])]
                    Vc = [torch.randn(B, m["Hkv"], S + 1, m["Dh"], device=dev, dtype=DTYPE) for _ in range(m["L"])]
                    g = capture_graph(lambda: model.attn_step(q, k, v, Kc, Vc, S, S + 1))
                    ms, lo, hi = time_auto(g.replay)
                    del g
                except torch.cuda.OutOfMemoryError:
                    Kc = Vc = None
                    torch.cuda.empty_cache()
                    continue
                wa.row(model=name, B=B, S=S, kv_bytes=kvb, ms=ms, ms_min=lo, ms_max=hi, **m)
                log(f"  attn  {name} B={B:3d} S={S:6d}: {ms:.3f} ms  {kvb / (ms * 1e-3) / 1e9:.0f} GB/s")
                del Kc, Vc
                torch.cuda.empty_cache()
            del x, q, k, v
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()


def run_prefill(out, smoke, models):
    dev = "cuda"
    wr = Writer(os.path.join(out, "prefill.csv"),
                ["model", "L", "D", "H", "Hkv", "Dh", "r", "B", "S", "ms", "ms_min", "ms_max", "sdpa"])
    ctxs = [128, 256, 512, 1024, 2048, 4096, 8192] if not smoke else [256, 1024]
    for name in models:
        m = MODELS[name]
        model = Model(m, dev)
        for S in ctxs:
            try:
                x = torch.randn(1, S, m["D"], device=dev, dtype=DTYPE)
                with torch.no_grad():
                    ms, lo, hi = time_fn(lambda: model.prefill(x), iters=3, trials=5, warmup=2)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"  prefill OOM {name} S={S}")
                continue
            wr.row(model=name, B=1, S=S, ms=ms, ms_min=lo, ms_max=hi, sdpa="default", **m)
            log(f"  prefill {name} S={S:5d}: {ms:.2f} ms")
            del x
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="short sanity run")
    ap.add_argument("--only", default="micro,attn,ops,decode,eager,prefill",
                    help="comma list of experiments to run")
    ap.add_argument("--models", default=",".join(MODELS), help="comma list of models")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not available to PyTorch. Install a CUDA build of PyTorch (see README).")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.set_grad_enabled(False)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ("-smoke" if a.smoke else "")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", stamp)
    os.makedirs(out, exist_ok=True)
    env = dict(time=stamp, python=sys.version, torch=torch.__version__,
               cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               device=torch.cuda.get_device_name(0),
               capability=torch.cuda.get_device_capability(0),
               total_mem=torch.cuda.get_device_properties(0).total_memory,
               sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
               os=platform.platform(), smoke=a.smoke, nvsmi_start=nvsmi())
    json.dump(env, open(os.path.join(out, "env.json"), "w"), indent=2, default=str)
    log(f"device: {env['device']}  torch {env['torch']}  CUDA {env['cuda']}  -> {out}")

    models = [s for s in a.models.split(",") if s]
    todo = a.only.split(",")
    t0 = time.time()
    for step in todo:
        log(f"=== {step} ===")
        try:
            if step == "micro":
                run_micro(out, a.smoke)
            elif step == "attn":
                run_attn(out, a.smoke)
            elif step == "decode":
                run_decode(out, a.smoke, models)
            elif step == "eager":
                run_decode(out, a.smoke, models[:2], eager_only=True)
            elif step == "ops":
                run_ops(out, a.smoke, models)
            elif step == "prefill":
                run_prefill(out, a.smoke, models)
        except Exception as e:                       # keep going; record the failure
            import traceback
            traceback.print_exc()
            with open(os.path.join(out, "errors.txt"), "a") as f:
                f.write(f"{step}: {e!r}\n{traceback.format_exc()}\n")
        torch.cuda.empty_cache()
    env["nvsmi_end"] = nvsmi()
    env["elapsed_s"] = time.time() - t0
    json.dump(env, open(os.path.join(out, "env.json"), "w"), indent=2, default=str)
    log(f"done in {env['elapsed_s'] / 60:.1f} min. Results in {out}")


if __name__ == "__main__":
    main()
