# GPU benchmark kit for the phase-aware memory model

Measures real decode and prefill latency on your GPU so that the paper's model can
be validated against hardware. Weights are random FP16 tensors (latency depends on
shapes, not trained values), so **no model download is needed**.

## 1. Install (once)

Requires an NVIDIA driver, Python 3.10–3.12, and a CUDA build of PyTorch.

```bash
# Windows (PowerShell or cmd) or Linux:
python -m venv venv
# Windows:  venv\Scripts\activate        Linux:  source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the name of your GPU.

## 2. Prepare the laptop (important for clean numbers)

- Plug in the charger. Laptop GPUs throttle hard on battery.
- Set the power plan / vendor tool (Armoury Crate, Lenovo Vantage, etc.) to
  **Performance / Turbo**, and in the NVIDIA Control Panel set the power mode to
  *Prefer maximum performance*.
- Close games, browsers with video, and other GPU programs.
- Put the laptop on a hard surface for airflow.

## 3. Run

```bash
cd gpu_bench
python bench.py --smoke      # about 2 minutes: checks that everything works
python bench.py              # full run: about 20–40 minutes
```

Each run writes a folder `results/<date-time>/` containing:

| file | contents |
|---|---|
| `env.json` | GPU, driver, PyTorch/CUDA versions, clocks and power at start/end |
| `micro.json` | measured bandwidth and FP16 throughput, GEMV batch sweep |
| `attn.csv` | decode-attention latency vs. context (KV-cache traffic) |
| `decode.csv` | full decode-step latency over batch × context × model |
| `eager.csv` | decode latency without CUDA graphs (framework overhead) |
| `prefill.csv` | prefill latency vs. prompt length |
| `errors.txt` | only if something failed |

**Send the whole results folder back (zip it).** If the smoke run fails, send the
terminal output and `errors.txt`.

Optional: run the full benchmark **twice** (e.g., on different days). Two runs let the
paper report run-to-run variation.

## 4. Analyse (optional, I can do this for you)

```bash
python analyze.py results/<date-time>
```

This writes figures to `../figures/` and LaTeX tables to `../tables/`, which the
manuscript includes directly.

## Models benchmarked

| name | L | D | H | KV heads | head dim | params (no embeddings) |
|---|---|---|---|---|---|---|
| M1-MHA | 24 | 1024 | 16 | 16 | 64 | 302 M |
| M1-GQA4 | 24 | 1024 | 16 | 4 | 64 | 264 M |
| M1-MQA | 24 | 1024 | 16 | 1 | 64 | 255 M |
| M2-GQA4 | 24 | 2048 | 16 | 4 | 128 | 1.06 B |

Out-of-memory configurations are skipped automatically.
