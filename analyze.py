#!/usr/bin/env python3
"""
Compare measured latencies (bench.py output) with the phase-aware model.

    python analyze.py results/<run-folder> [more run folders ...]

Calibration / validation protocol (kept strictly separate):
  * P_eff  <- large FP16 GEMM throughput           (micro.json)
  * B_eff  <- single-vector GEMV bandwidth         (micro.json)
  * rho    <- slope of decode-attention latency vs KV bytes      (attn.csv)
  * t0     <- fixed per-step overhead, fitted on M1-MHA at B=1 only (decode.csv)
  Everything else in decode.csv / prefill.csv is a held-out test of the model.

Writes figures to ../figures (meas_*.pdf, balance.pdf), LaTeX tables to
../tables, and summary.json into the (first) run folder.
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

b = 2  # bytes per FP16 element
HERE = os.path.dirname(os.path.abspath(__file__))

# Published peak specifications used only for the cross-platform projection.
SPEC_3080L_GBPS = 448.0        # RTX 3080 Laptop GPU, 256-bit GDDR6 @ 14 Gbps
PLATFORMS = [                  # name, MAC/s, bytes/s   (FP16 tensor, dense)
    ("A100 80GB SXM (spec)", 312e12 / 2, 2039e9),
    ("H100 SXM (spec)", 989e12 / 2, 3350e9),
    ("Compute-light accel.", 5e12, 900e9),
]

plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e3e3",
                     "grid.linewidth": 0.6, "axes.edgecolor": "#555", "legend.frameon": False,
                     "savefig.bbox": "tight"})
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
MARK = ["o", "s", "^", "D", "v", "P"]
INK = "#333333"
from matplotlib.ticker import NullFormatter, LogLocator, ScalarFormatter


def tidy(ax):
    """Major ticks only, plain numbers, on any log axis."""
    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale == "log":
            axis.set_minor_formatter(NullFormatter())
            f = ScalarFormatter(); f.set_scientific(False)
            axis.set_major_formatter(f)


# --------------------------------------------------------------------------
def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v) if any(c in v for c in ".eE") else int(v)
            except (ValueError, TypeError):
                d[k] = v
        out.append(d)
    return out


def merge_runs(dirs):
    """Average repeated runs (median of medians) keyed by configuration."""
    data = {}
    for name, keys in [("decode", ("model", "B", "S")), ("eager", ("model", "B", "S")),
                       ("prefill", ("model", "S")), ("attn", ("Hkv", "B", "S"))]:
        groups = defaultdict(list)
        for d in dirs:
            for r in read_csv(os.path.join(d, f"{name}.csv")):
                groups[tuple(r[k] for k in keys)].append(r)
        rows = []
        for k, rs in groups.items():
            r = dict(rs[0])
            ms = [x["ms"] for x in rs]
            r["ms"] = float(np.median(ms))
            r["ms_runs"] = ms
            r["n_runs"] = len(ms)
            rows.append(r)
        data[name] = rows
    micros = [json.load(open(os.path.join(d, "micro.json"))) for d in dirs
              if os.path.exists(os.path.join(d, "micro.json"))]
    if not micros:
        sys.exit("micro.json missing: run bench.py with the 'micro' experiment.")
    micro = dict(micros[0])
    for k in ("copy_GBps", "gemv_GBps", "gemm_TMACs"):
        micro[k] = float(np.median([m[k] for m in micros]))
        micro[k + "_runs"] = [m[k] for m in micros]
    data["micro"] = micro
    data["env"] = [json.load(open(os.path.join(d, "env.json"))) for d in dirs
                   if os.path.exists(os.path.join(d, "env.json"))]
    return data


# --------------------------------------------------------------------------
# The analytical model (Section 4 of the paper)
# --------------------------------------------------------------------------
def W_layer(r):
    D, Hkv, Dh, ff = r["D"], r["Hkv"], r["Dh"], r["r"]
    return 2 * D * D + 2 * D * Hkv * Dh + 2 * ff * D * D


def t_comp(r, B, S, P):
    return B * r["L"] * (W_layer(r) + 2 * S * r["D"]) / P


def t_mem(r, B, S, Bw, rho):
    return r["L"] * (W_layer(r) * b + B * rho * 2 * r["Hkv"] * r["Dh"] * S * b) / Bw


def pred_decode(r, P, Bw, rho, t0=0.0, mode="max"):
    S = r["S"] + 1                     # the new token attends to S+1 positions
    tc, tm = t_comp(r, r["B"], S, P), t_mem(r, r["B"], S, Bw, rho)
    return (max(tc, tm) if mode == "max" else tc + tm) + t0


def pred_prefill(r, P):
    S, D, L = r["S"], r["D"], r["L"]
    # causal attention: half of the 2 S^2 D score/value work is needed
    return r["B"] * L * (W_layer(r) * S + S * S * D) / P


def s_star(D, Hkv, Dh, P, Bw, rho, Wl=None):
    Wl = 12 * D * D if Wl is None else Wl
    Rinf = rho * Hkv * Dh * b * P / (D * Bw)
    return (Wl / (2 * D * (Rinf - 1)) if Rinf > 1 else np.inf), Rinf


def mape(pred, meas):
    pred, meas = np.asarray(pred), np.asarray(meas)
    return float(np.mean(np.abs(pred - meas) / meas) * 100)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--figdir", default=os.path.join(HERE, "..", "figures"))
    ap.add_argument("--tabdir", default=os.path.join(HERE, "..", "tables"))
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    os.makedirs(a.tabdir, exist_ok=True)
    data = merge_runs(a.runs)
    micro, dec, pre, att, eag = data["micro"], data["decode"], data["prefill"], data["attn"], data["eager"]
    S_ = {}

    # ---------------- calibration -------------------------------------
    P = micro["gemm_TMACs"] * 1e12
    Bw = micro["gemv_GBps"] * 1e9
    S_["P_eff_TMACs"], S_["B_eff_GBps"] = P / 1e12, Bw / 1e9
    S_["copy_GBps"] = micro["copy_GBps"]
    S_["B_eff_frac_of_spec"] = Bw / 1e9 / SPEC_3080L_GBPS
    S_["balance_MAC_per_byte"] = P / Bw
    S_["B_c"] = b * P / Bw
    S_["device"] = data["env"][0].get("device", "?") if data["env"] else "?"
    S_["n_runs"] = len(a.runs)

    # rho from attention microbenchmark: t = a + kv_bytes * rho / Bw
    rho_fits = {}
    for (Hkv, B) in sorted({(r["Hkv"], r["B"]) for r in att}):
        rs = sorted([r for r in att if r["Hkv"] == Hkv and r["B"] == B], key=lambda r: r["S"])
        if len(rs) >= 2:
            x = np.array([r["kv_bytes"] for r in rs]); y = np.array([r["ms"] * 1e-3 for r in rs])
            c, a0 = np.polyfit(x, y, 1)
            rho_fits[f"Hkv{Hkv}_B{B}"] = dict(rho=float(c * Bw), intercept_us=float(a0 * 1e6),
                                              peak_GBps=float(max(r["GBps"] for r in rs)))
    rho = float(np.median([v["rho"] for v in rho_fits.values()])) if rho_fits else 1.0
    S_["rho_hat"], S_["rho_fits"] = rho, rho_fits

    # t0 from M1-MHA, B=1 only (calibration set)
    cal = [r for r in dec if r["model"] == "M1-MHA" and r["B"] == 1]
    t0 = float(np.median([r["ms"] * 1e-3 - pred_decode(r, P, Bw, rho) for r in cal])) if cal else 0.0
    t0 = max(t0, 0.0)
    S_["t0_ms"] = t0 * 1e3

    # ---------------- decode validation --------------------------------
    for r in dec:
        r["p_raw"] = pred_decode(r, P, Bw, 1.0) * 1e3              # rho = 1, no overhead
        r["p_cal"] = pred_decode(r, P, Bw, rho, t0) * 1e3           # calibrated max model
        r["p_sum"] = pred_decode(r, P, Bw, rho, t0, mode="sum") * 1e3
        r["bound"] = "memory" if t_mem(r, r["B"], r["S"] + 1, Bw, rho) > t_comp(r, r["B"], r["S"] + 1, P) else "compute"
        r["heldout"] = not (r["model"] == "M1-MHA" and r["B"] == 1)
    models = sorted({r["model"] for r in dec}, key=lambda m: ["M1-MHA", "M1-GQA4", "M1-MQA", "M2-GQA4"].index(m)
                    if m in ["M1-MHA", "M1-GQA4", "M1-MQA", "M2-GQA4"] else 9)
    err = []
    for m in models + ["All held-out"]:
        rs = [r for r in dec if (r["model"] == m if m != "All held-out" else r["heldout"])]
        if not rs:
            continue
        meas = [r["ms"] for r in rs]
        e = dict(model=m, n=len(rs), raw=mape([r["p_raw"] for r in rs], meas),
                 cal=mape([r["p_cal"] for r in rs], meas), sum=mape([r["p_sum"] for r in rs], meas),
                 worst=float(max(abs(r["p_cal"] - r["ms"]) / r["ms"] * 100 for r in rs)))
        err.append(e)
    S_["decode_error"] = err

    # slopes at B=1: t(S) = a + c S
    slopes = []
    for m in models:
        rs = sorted([r for r in dec if r["model"] == m and r["B"] == 1], key=lambda r: r["S"])
        if len(rs) < 3:
            continue
        x = np.array([r["S"] + 1 for r in rs], float); y = np.array([r["ms"] for r in rs])
        c, a0 = np.polyfit(x, y, 1)
        r0 = rs[0]
        c_pred = r0["L"] * rho * 2 * r0["Hkv"] * r0["Dh"] * b / Bw * 1e3
        a_pred = r0["L"] * W_layer(r0) * b / Bw * 1e3 + t0 * 1e3
        kvbw = r0["L"] * 2 * r0["Hkv"] * r0["Dh"] * b / (c * 1e-3) / 1e9
        Sw_pred = W_layer(r0) / (2 * rho * r0["Hkv"] * r0["Dh"])
        slopes.append(dict(model=m, slope_us_per_ktok=c * 1e6, slope_pred=c_pred * 1e6,
                           icpt_ms=a0, icpt_pred=a_pred, kv_GBps=kvbw,
                           S_cross_meas=(a0 - t0 * 1e3) / c, S_cross_pred=Sw_pred))
    S_["slopes_b1"] = slopes

    # compute->memory transition at batch B (closed form S*_B = S*(1 - Bc/B))
    trans = []
    for m in models:
        r0 = next(r for r in dec if r["model"] == m)
        Ss, Rinf = s_star(r0["D"], r0["Hkv"], r0["Dh"], P, Bw, rho, W_layer(r0))
        for B in sorted({r["B"] for r in dec if r["model"] == m}):
            SB = Ss * (1 - S_["B_c"] / B) if (B > S_["B_c"] and np.isfinite(Ss)) else None
            trans.append(dict(model=m, B=B, S_star=Ss, R_inf=Rinf, S_star_B=SB))
    S_["transitions"] = trans

    # ---------------- prefill ------------------------------------------
    for r in pre:
        r["p"] = pred_prefill(r, P) * 1e3
    perr = []
    for m in models:
        rs = [r for r in pre if r["model"] == m]
        if rs:
            perr.append(dict(model=m, n=len(rs), mape=mape([r["p"] for r in rs], [r["ms"] for r in rs]),
                             tmacs=[float(r["L"] * (W_layer(r) * r["S"] + r["S"] ** 2 * r["D"]) / (r["ms"] * 1e-3) / 1e12) for r in sorted(rs, key=lambda r: r["S"])]))
    S_["prefill_error"] = perr

    # ---------------- eager overhead -----------------------------------
    eg = []
    for r in eag:
        g = next((d for d in dec if d["model"] == r["model"] and d["B"] == r["B"] and d["S"] == r["S"]), None)
        if g:
            eg.append(dict(model=r["model"], S=r["S"], eager=r["ms"], graph=g["ms"], ratio=r["ms"] / g["ms"]))
    S_["eager"] = eg

    # ---------------- figures ------------------------------------------
    # 1 GEMV roofline
    sw = micro.get("gemv_sweep", [])
    if sw:
        fig, ax = plt.subplots(figsize=(5.4, 3.3))
        Bs = np.array([s["B"] for s in sw]); T = np.array([s["TMACs"] for s in sw])
        xx = np.logspace(0, np.log10(max(Bs) * 1.5), 200)
        ax.plot(xx, np.minimum(P, xx / b * Bw) / 1e12, color=INK, lw=1.2, ls="--", label="Roofline (measured $P$, $B$)")
        ax.plot(Bs, T, color=CAT[0], lw=2, marker="o", ms=5, label="Measured GEMV/GEMM")
        ax.axvline(S_["B_c"], color="#888", lw=1, ls=":")
        ax.annotate(f"$B_c\\approx{S_['B_c']:.0f}$", (S_["B_c"], T.max() * 0.08), xytext=(5, 0), textcoords="offset points", color=INK)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("Batch size $B$ (rows per weight pass)"); ax.set_ylabel("Achieved throughput (TMAC/s)")
        ax.legend(loc="lower right"); tidy(ax); fig.savefig(os.path.join(a.figdir, "meas_roofline.pdf")); plt.close(fig)

    # 2 decode B=1, all models
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, m in enumerate(models):
        rs = sorted([r for r in dec if r["model"] == m and r["B"] == 1], key=lambda r: r["S"])
        if not rs:
            continue
        Sx = [r["S"] for r in rs]
        ax.plot(Sx, [r["p_cal"] for r in rs], color=CAT[i], lw=1.6, label=f"{m} (model)")
        ax.plot(Sx, [r["ms"] for r in rs], ls="none", marker=MARK[i], ms=5, mfc="white", mec=CAT[i], mew=1.4, label=f"{m} (measured)")
    ax.set_xscale("log", base=2); ax.set_xlabel("Context length $S$ (tokens)"); ax.set_ylabel("Decode step latency (ms)")
    ax.legend(ncol=2, fontsize=7.5, loc="upper left"); tidy(ax); fig.savefig(os.path.join(a.figdir, "meas_decode_b1.pdf")); plt.close(fig)

    # 3 batches for M1-MHA and M1-GQA4
    for m in [x for x in ("M1-MHA", "M1-GQA4") if x in models]:
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        Bl = sorted({r["B"] for r in dec if r["model"] == m})
        for i, B in enumerate(Bl):
            rs = sorted([r for r in dec if r["model"] == m and r["B"] == B], key=lambda r: r["S"])
            c = RAMP[min(i, len(RAMP) - 1)]
            ax.plot([r["S"] for r in rs], [r["p_cal"] for r in rs], color=c, lw=1.5)
            ax.plot([r["S"] for r in rs], [r["ms"] for r in rs], ls="none", marker=MARK[i % 6], ms=5, mfc="white", mec=c, mew=1.4, label=f"$B={B}$")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("Context length $S$ (tokens)"); ax.set_ylabel("Decode step latency (ms)")
        ax.legend(ncol=3, fontsize=8, loc="upper left", title="markers: measured; lines: model", title_fontsize=7.5)
        tidy(ax); fig.savefig(os.path.join(a.figdir, f"meas_decode_batch_{m}.pdf")); plt.close(fig)

    # 4 parity
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for i, m in enumerate(models):
        rs = [r for r in dec if r["model"] == m]
        ax.plot([r["p_cal"] for r in rs], [r["ms"] for r in rs], ls="none", marker=MARK[i], ms=4.5, mfc="white", mec=CAT[i], mew=1.2, label=m)
    lo = min(min(r["ms"] for r in dec), min(r["p_cal"] for r in dec)) * 0.8
    hi = max(max(r["ms"] for r in dec), max(r["p_cal"] for r in dec)) * 1.25
    xx = np.array([lo, hi])
    ax.plot(xx, xx, color=INK, lw=1); ax.fill_between(xx, xx / 1.2, xx * 1.2, color="#cde2fb", alpha=0.5, lw=0, label="$\\pm$20%")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted latency (ms)"); ax.set_ylabel("Measured latency (ms)"); ax.legend(fontsize=8, loc="upper left")
    tidy(ax); fig.savefig(os.path.join(a.figdir, "meas_parity.pdf")); plt.close(fig)

    # 5 prefill
    if pre:
        fig, ax = plt.subplots(figsize=(5.4, 3.3))
        for i, m in enumerate(models):
            rs = sorted([r for r in pre if r["model"] == m], key=lambda r: r["S"])
            if rs:
                ax.plot([r["S"] for r in rs], [r["p"] for r in rs], color=CAT[i], lw=1.5)
                ax.plot([r["S"] for r in rs], [r["ms"] for r in rs], ls="none", marker=MARK[i], ms=5, mfc="white", mec=CAT[i], mew=1.4, label=m)
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("Prompt length $S$ (tokens)"); ax.set_ylabel("Prefill latency (ms)")
        ax.legend(fontsize=8, title="markers: measured; lines: model", title_fontsize=7.5)
        tidy(ax); fig.savefig(os.path.join(a.figdir, "meas_prefill.pdf")); plt.close(fig)

    # 6 machine balance -> S*   (analytical, M1 geometry, rho = rho_hat)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    bal = np.logspace(-0.5, 2.6, 400)
    for i, (Hkv, st) in enumerate([(16, "-"), (4, "--"), (1, ":")]):
        Wl = 2 * 1024 ** 2 + 2 * 1024 * Hkv * 64 + 8 * 1024 ** 2
        ys = [s_star(1024, Hkv, 64, x, 1.0, rho, Wl)[0] for x in bal]
        ax.plot(bal, ys, color=RAMP[2 + i], lw=2, ls=st, label=f"$H_{{kv}}={Hkv}$")
    pts = [("RTX 3080 Laptop (measured)", P, Bw)] + PLATFORMS
    for j, (nm, p_, b_) in enumerate(pts):
        x = p_ / b_
        y = s_star(1024, 16, 64, p_, b_, rho)[0]
        ax.plot([x], [y], marker=MARK[j], ms=7, color=CAT[j + 1 if j else 1], ls="none", label=nm)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5, 1e6)
    ax.set_xlabel("Machine balance $P_{\\mathrm{eff}}/B_{\\mathrm{eff}}$ (MAC/byte)"); ax.set_ylabel("Transition length $S^{*}$ (tokens)")
    ax.legend(fontsize=7.5, loc="upper right"); tidy(ax)
    from matplotlib.ticker import LogFormatterSciNotation
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    fig.savefig(os.path.join(a.figdir, "balance.pdf")); plt.close(fig)

    # ---------------- tables -------------------------------------------
    def w(name, body):
        open(os.path.join(a.tabdir, name), "w").write(body)

    w("platform.tex",
      "\\begin{tabular}{@{}lr@{}}\n\\toprule\nQuantity & Value \\\\\n\\midrule\n"
      f"Device & {S_['device']} \\\\\n"
      f"Copy bandwidth (read+write) & {S_['copy_GBps']:.0f} GB/s \\\\\n"
      f"GEMV bandwidth, $\\Beff$ & {S_['B_eff_GBps']:.0f} GB/s ({100 * S_['B_eff_frac_of_spec']:.0f}\\% of 448 GB/s peak) \\\\\n"
      f"FP16 GEMM throughput, $\\Peff$ & {S_['P_eff_TMACs']:.1f} TMAC/s \\\\\n"
      f"Machine balance, $\\Peff/\\Beff$ & {S_['balance_MAC_per_byte']:.1f} MAC/byte \\\\\n"
      f"Critical batch, $B_c = b\\Peff/\\Beff$ & {S_['B_c']:.0f} \\\\\n"
      f"Effective KV traffic factor, $\\hat\\rho$ & {rho:.2f} \\\\\n"
      f"Fixed per-step overhead, $t_0$ & {S_['t0_ms']:.3f} ms \\\\\n"
      "\\bottomrule\n\\end{tabular}\n")
    w("decode_error.tex",
      "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
      "Model & Configs & \\multicolumn{3}{c}{MAPE (\\%)} & Worst (\\%) \\\\\n\\cmidrule(lr){3-5}\n"
      " & & Uncalibrated & Calibrated & Sum variant & \\\\\n\\midrule\n" +
      "".join(f"{e['model']} & {e['n']} & {e['raw']:.1f} & {e['cal']:.1f} & {e['sum']:.1f} & {e['worst']:.1f} \\\\\n" for e in err) +
      "\\bottomrule\n\\end{tabular}\n")
    w("slopes.tex",
      "\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
      "Model & \\multicolumn{2}{c}{Slope ($\\mu$s / 1k tokens)} & \\multicolumn{2}{c}{Intercept (ms)} & KV BW & $S_w$ meas./model \\\\\n"
      "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & Meas. & Model & Meas. & Model & (GB/s) & (tokens) \\\\\n\\midrule\n" +
      "".join(f"{s['model']} & {s['slope_us_per_ktok']:.1f} & {s['slope_pred']:.1f} & {s['icpt_ms']:.3f} & {s['icpt_pred']:.3f} & {s['kv_GBps']:.0f} & {s['S_cross_meas']:,.0f} / {s['S_cross_pred']:,.0f} \\\\\n" for s in slopes) +
      "\\bottomrule\n\\end{tabular}\n")
    w("prefill_error.tex",
      "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nModel & Configs & MAPE (\\%) & Achieved TMAC/s (min--max) \\\\\n\\midrule\n" +
      "".join(f"{e['model']} & {e['n']} & {e['mape']:.1f} & {min(e['tmacs']):.1f}--{max(e['tmacs']):.1f} \\\\\n" for e in perr) +
      "\\bottomrule\n\\end{tabular}\n")
    w("eager.tex",
      "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nModel & $S$ & Eager (ms) & CUDA graph (ms) & Ratio \\\\\n\\midrule\n" +
      "".join(f"{e['model']} & {e['S']:,} & {e['eager']:.3f} & {e['graph']:.3f} & {e['ratio']:.2f} \\\\\n" for e in eg) +
      "\\bottomrule\n\\end{tabular}\n")
    tr_rows = [t for t in trans if t["B"] in (1, 16, 64, 128, 256)]
    w("transitions.tex",
      "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nModel & $R_\\infty$ & $S^{*}$ (tokens) & $B$ & $S^{*}_B$ (tokens) \\\\\n\\midrule\n" +
      "".join(f"{t['model']} & {t['R_inf']:.1f} & {t['S_star']:.0f} & {t['B']} & {('%.0f' % t['S_star_B']) if t['S_star_B'] else '---'} \\\\\n" for t in tr_rows) +
      "\\bottomrule\n\\end{tabular}\n")

    json.dump(S_, open(os.path.join(a.runs[0], "summary.json"), "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in S_.items() if k not in ("rho_fits", "transitions")}, indent=1, default=float))


if __name__ == "__main__":
    main()
