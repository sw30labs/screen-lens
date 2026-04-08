"""
Batch-size sweep benchmark for the MLX-VLM captioner.

Loads a fixed pool of frames, then for each batch size in the sweep runs
``MLXVLMCaptioner.caption_batch`` repeatedly until the whole pool is captioned,
recording wall-clock time, frames/second, and peak GPU memory.

The model is loaded ONCE per process and reused across batch sizes (the
captioner caches it). MLX peak memory is reset between sweeps via
``mx.reset_peak_memory()`` so each row reflects only that batch size's footprint.

Usage:
    python scripts/bench_caption_batch.py \\
        --frames-dir data/Design/frames \\
        --num-frames 16 \\
        --batch-sizes 1,2,4,8,16 \\
        --mlx-repo mlx-community/Qwen3.5-122B-A10B-bf16 \\
        --output bench_results

Outputs ``bench_results.csv`` and (if matplotlib is installed)
``bench_results.png``. Always prints an ASCII summary table and a recommended
batch size based on the throughput knee + a configurable memory ceiling.
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

# Make the repo importable when run as `python scripts/bench_caption_batch.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CaptioningConfig, CaptionBackend  # noqa: E402
from src.captioner import MLXVLMCaptioner  # noqa: E402


def collect_frames(frames_dir: Path, num_frames: int) -> list[str]:
    """Pick the first ``num_frames`` JPGs in the directory (sorted)."""
    if not frames_dir.is_dir():
        raise SystemExit(f"frames-dir not found: {frames_dir}")
    files = sorted(frames_dir.glob("frame_*.jpg"))
    if not files:
        files = sorted(frames_dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"no .jpg frames in {frames_dir}")
    if len(files) < num_frames:
        print(f"  warning: only {len(files)} frames available, requested {num_frames}")
        num_frames = len(files)
    return [str(p) for p in files[:num_frames]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLX-VLM captioner batch-size sweep")
    p.add_argument("--frames-dir", type=Path, default=Path("data/Design/frames"),
                   help="Directory containing frame_*.jpg files")
    p.add_argument("--num-frames", type=int, default=16,
                   help="How many frames to caption per batch-size run")
    p.add_argument("--batch-sizes", type=str, default="1,2,4,8,16",
                   help="Comma-separated batch sizes to sweep")
    p.add_argument("--mlx-repo", type=str,
                   default="mlx-community/Qwen3.5-122B-A10B-bf16",
                   help="MLX model repo id")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="max_tokens per caption (lower → faster sweep)")
    p.add_argument("--output", type=Path, default=Path("bench_results"),
                   help="Output prefix (writes <prefix>.csv and <prefix>.png)")
    p.add_argument("--memory-ceiling-gb", type=float, default=410.0,
                   help="Recommended max peak memory (default 410GB ~80%% of 512GB)")
    p.add_argument("--warmup", action="store_true",
                   help="Run a small batch-size=1 warmup pass before timing")
    return p.parse_args()


def main():
    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    if not batch_sizes:
        raise SystemExit("--batch-sizes is empty")

    print(f"Batch-size sweep: {batch_sizes}")
    print(f"Frames pool: {args.num_frames} from {args.frames_dir}")
    print(f"Model: {args.mlx_repo}")
    print(f"max_tokens per caption: {args.max_tokens}")
    print()

    image_paths = collect_frames(args.frames_dir, args.num_frames)
    print(f"Loaded {len(image_paths)} frame paths.")

    # Build a captioner with the right model + max_tokens
    cfg = CaptioningConfig(
        backend=CaptionBackend.mlx_vlm,
        mlx_repo_id=args.mlx_repo,
        max_tokens=args.max_tokens,
    )
    captioner = MLXVLMCaptioner(cfg)

    # Force model load up front so its load time isn't counted in the first batch
    print("Loading model (this may take a while on first run)...")
    t_load = time.perf_counter()
    captioner._load_model()
    print(f"Model loaded in {time.perf_counter() - t_load:.1f}s")

    import mlx.core as mx  # imported after captioner load so MLX is initialized

    if args.warmup:
        print("\nWarmup: 1 frame at batch_size=1...")
        captioner.caption_batch(image_paths[:1])
        gc.collect()

    rows: list[dict] = []
    for bs in batch_sizes:
        print(f"\n──────── batch_size = {bs} ────────")
        # Reset MLX peak counter so this run's peak is isolated
        try:
            mx.reset_peak_memory()
        except AttributeError:
            pass  # older mlx — fall back to cumulative peak
        gc.collect()

        n_chunks = -(-len(image_paths) // bs)  # ceil
        print(f"  {n_chunks} chunk(s) of up to {bs} frame(s) each")

        t0 = time.perf_counter()
        all_caps: list[str] = []
        for i in range(0, len(image_paths), bs):
            chunk = image_paths[i : i + bs]
            try:
                caps = captioner.caption_batch(chunk)
            except Exception as e:
                print(f"  ERROR at chunk starting frame {i}: {e}")
                rows.append({
                    "batch_size": bs,
                    "frames": len(image_paths),
                    "wall_seconds": float("nan"),
                    "frames_per_sec": float("nan"),
                    "peak_memory_gb": float("nan"),
                    "error": str(e)[:200],
                })
                all_caps = None
                break
            all_caps.extend(caps)
            print(f"  chunk {i // bs + 1}/{n_chunks}: {len(caps)} captions, "
                  f"avg_len={sum(len(c) for c in caps) / max(len(caps), 1):.0f}")

        if all_caps is None:
            continue

        elapsed = time.perf_counter() - t0
        try:
            peak_gb = mx.get_peak_memory() / 1e9
        except AttributeError:
            peak_gb = float("nan")
        fps = len(image_paths) / elapsed if elapsed > 0 else float("nan")

        print(f"  TOTAL: {elapsed:.1f}s | {fps:.3f} frames/s | peak {peak_gb:.1f} GB")
        rows.append({
            "batch_size": bs,
            "frames": len(image_paths),
            "wall_seconds": round(elapsed, 3),
            "frames_per_sec": round(fps, 4),
            "peak_memory_gb": round(peak_gb, 2),
            "error": "",
        })

    # ── Write CSV ──────────────────────────────────────────────────────────
    csv_path = args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "batch_size", "frames", "wall_seconds",
            "frames_per_sec", "peak_memory_gb", "error",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    # ── ASCII summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  {'batch':>6} {'frames/s':>10} {'speedup':>9} {'peak GB':>10} {'wall s':>9}")
    print("  " + "-" * 56)
    baseline = next((r["frames_per_sec"] for r in rows
                     if r["batch_size"] == 1 and r["frames_per_sec"] == r["frames_per_sec"]),
                    None)
    for r in rows:
        fps = r["frames_per_sec"]
        speedup = (fps / baseline) if (baseline and fps == fps) else float("nan")
        speedup_s = f"{speedup:>8.2f}x" if speedup == speedup else "      n/a"
        fps_s = f"{fps:>10.3f}" if fps == fps else "       n/a"
        peak_s = f"{r['peak_memory_gb']:>10.1f}" if r["peak_memory_gb"] == r["peak_memory_gb"] else "       n/a"
        wall_s = f"{r['wall_seconds']:>9.1f}" if r["wall_seconds"] == r["wall_seconds"] else "      n/a"
        print(f"  {r['batch_size']:>6} {fps_s} {speedup_s} {peak_s} {wall_s}")
    print("=" * 64)

    # ── Recommendation ─────────────────────────────────────────────────────
    valid = [r for r in rows
             if r["frames_per_sec"] == r["frames_per_sec"]
             and r["peak_memory_gb"] == r["peak_memory_gb"]
             and r["peak_memory_gb"] <= args.memory_ceiling_gb]
    if valid:
        # Pick the largest batch size where throughput is still within 5% of the best
        best_fps = max(r["frames_per_sec"] for r in valid)
        within = [r for r in valid if r["frames_per_sec"] >= 0.95 * best_fps]
        recommended = max(within, key=lambda r: r["batch_size"])
        print(f"\nRecommended batch_size: {recommended['batch_size']}  "
              f"(frames/s={recommended['frames_per_sec']:.3f}, "
              f"peak={recommended['peak_memory_gb']:.1f} GB ≤ {args.memory_ceiling_gb:.0f} GB ceiling)")
    else:
        print("\nNo valid runs within memory ceiling — all batch sizes either failed "
              "or exceeded the ceiling. Re-run with --memory-ceiling-gb adjusted or a "
              "smaller --num-frames / --max-tokens.")

    # ── Plot ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bs_x = [r["batch_size"] for r in rows if r["frames_per_sec"] == r["frames_per_sec"]]
        fps_y = [r["frames_per_sec"] for r in rows if r["frames_per_sec"] == r["frames_per_sec"]]
        mem_y = [r["peak_memory_gb"] for r in rows if r["peak_memory_gb"] == r["peak_memory_gb"]]

        if bs_x:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.set_xlabel("batch_size")
            ax1.set_ylabel("frames / second", color="tab:blue")
            ax1.plot(bs_x, fps_y, "o-", color="tab:blue", label="frames/s")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.set_xscale("log", base=2)
            ax1.set_xticks(bs_x)
            ax1.set_xticklabels([str(b) for b in bs_x])

            ax2 = ax1.twinx()
            ax2.set_ylabel("peak memory (GB)", color="tab:red")
            ax2.plot(bs_x, mem_y, "s--", color="tab:red", label="peak GB")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            ax2.axhline(args.memory_ceiling_gb, color="tab:red", linestyle=":",
                        alpha=0.4, label=f"ceiling {args.memory_ceiling_gb:.0f} GB")

            plt.title(f"MLX-VLM batch sweep — {args.mlx_repo.split('/')[-1]}\n"
                      f"{args.num_frames} frames, max_tokens={args.max_tokens}")
            fig.tight_layout()
            png_path = args.output.with_suffix(".png")
            plt.savefig(png_path, dpi=120)
            print(f"Wrote {png_path}")
    except ImportError:
        print("\n(matplotlib not installed — skipping PNG plot. "
              "`pip install matplotlib` to enable it.)")


if __name__ == "__main__":
    main()
