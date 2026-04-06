"""
CLI Interface for ScreenLens.

Usage:
    python -m src.cli ingest VIDEO_PATH                     # Ingest with Qwen3.5 + keyframes
    python -m src.cli ingest VIDEO_PATH --backend ollama    # Use Ollama instead
    python -m src.cli search "your query"                   # Search ingested video
    python -m src.cli run VIDEO_PATH "query"                # Ingest + search in one shot
    python -m src.cli batch FOLDER_PATH                     # Batch-ingest all videos in a folder
    python -m src.cli reconstruct                           # Reconstruct artifacts from captions
    python -m src.cli info                                  # Show vector store stats
"""
import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import ScreenLensConfig, CaptionBackend, ExtractionStrategy
from .pipeline import build_ingest_graph, build_search_graph, build_full_graph, summarize_all_node

app = typer.Typer(
    name="screenlens",
    help="ScreenLens — Local video scene intelligence for Apple Silicon",
    rich_markup_mode="rich",
)
console = Console()


def _load_config(config_path: Optional[str] = None) -> ScreenLensConfig:
    """Load config from file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return ScreenLensConfig(**json.load(f))
    return ScreenLensConfig()


@app.command()
def ingest(
    video_path: str = typer.Argument(..., help="Path to the video file (.mov, .mp4, etc.)"),
    # Extraction strategy
    strategy: str = typer.Option("keyframe", help="Extraction strategy: 'keyframe' (smart) or 'fixed_fps'"),
    fps: float = typer.Option(1.0, help="Frames per second (only for fixed_fps strategy)"),
    max_interval: float = typer.Option(4.0, help="Max seconds between keyframes (keyframe strategy)"),
    # Captioning backend
    backend: str = typer.Option("mlx_vlm", help="Caption backend: 'mlx_vlm' or 'ollama'"),
    mlx_repo: str = typer.Option(
        "mlx-community/Qwen3.5-122B-A10B-bf16",
        help="HuggingFace repo ID for MLX vision model"
    ),
    ollama_model: str = typer.Option("llama3.2-vision", help="Ollama vision model (if backend=ollama)"),
    ollama_url: str = typer.Option("http://127.0.0.1:11434", help="Ollama API URL"),
    # Other
    device: str = typer.Option("mps", help="Device for CLIP: mps, cuda, cpu"),
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Ingest a video: extract keyframes, generate captions, create embeddings."""
    video = Path(video_path)
    if not video.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)

    config = _load_config(config_file)

    # Frame extraction
    config.frame_extraction.strategy = ExtractionStrategy(strategy)
    config.frame_extraction.fps = fps
    config.frame_extraction.max_interval_seconds = max_interval

    # Captioning
    config.captioning.backend = CaptionBackend(backend)
    config.captioning.mlx_repo_id = mlx_repo
    config.captioning.ollama_model = ollama_model
    config.captioning.ollama_base_url = ollama_url
    # Embedding
    config.embedding.device = device

    # Display config
    if backend == "mlx_vlm":
        model_display = mlx_repo.split("/")[-1]
    else:
        model_display = ollama_model

    console.print(Panel.fit(
        f"[bold green]ScreenLens — Video Ingestion[/bold green]\n"
        f"Video: {video.name} ({video.stat().st_size / (1024**2):.0f} MB)\n"
        f"Extraction: {strategy} | Captioning: {backend} ({model_display})\n"
        f"CLIP device: {device}",
        title="Configuration",
    ))

    pipeline = build_ingest_graph()
    initial_state = {
        "video_path": str(video.resolve()),
        "config": config.model_dump(),
    }

    t0 = time.time()
    result = pipeline.invoke(initial_state)
    total_time = time.time() - t0

    # Display results
    console.print(f"\n[bold green]Ingestion complete![/bold green]")

    table = Table(title="Pipeline Summary")
    table.add_column("Stage", style="cyan")
    table.add_column("Time (s)", justify="right")
    table.add_column("Details")

    elapsed = result.get("elapsed_seconds", {})
    table.add_row(
        "Frame Extraction",
        f"{elapsed.get('ingest', 0):.1f}",
        f"{result.get('num_frames', 0)} frames ({strategy})"
    )
    table.add_row(
        "Captioning",
        f"{elapsed.get('caption', 0):.1f}",
        f"{backend} ({model_display})"
    )
    emb_shape = result.get('embeddings_shape', ['?', '?'])
    table.add_row(
        "Embedding + Store",
        f"{elapsed.get('embed', 0):.1f}",
        f"dim={emb_shape[1] if len(emb_shape) > 1 else '?'}"
    )
    table.add_row("Total", f"{total_time:.1f}", "", style="bold")
    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    top_k: int = typer.Option(10, help="Number of results to return"),
    summarize: bool = typer.Option(True, help="Generate LLM summary of results"),
    ollama_url: str = typer.Option("http://127.0.0.1:11434", help="Ollama API URL"),
    device: str = typer.Option("mps", help="Device for CLIP: mps, cuda, cpu"),
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Search the ingested video with a natural language query."""
    config = _load_config(config_file)
    config.search.top_k = top_k
    config.search.base_url = ollama_url
    config.embedding.device = device

    console.print(f"\n[bold cyan]Searching:[/bold cyan] '{query}'\n")

    pipeline = build_search_graph()
    state = {"query": query, "config": config.model_dump()}
    result = pipeline.invoke(state)

    results = result.get("search_results", [])
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Top {len(results)} Results")
    table.add_column("#", justify="right", width=3)
    table.add_column("Time", style="cyan", width=12)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Caption", max_width=80)

    for i, r in enumerate(results):
        table.add_row(
            str(i + 1),
            r.get("timestamp_str", "?"),
            f"{r.get('score', 0):.3f}",
            r.get("caption", "")[:120] + "...",
        )
    console.print(table)

    if summarize and result.get("summary"):
        console.print(Panel(result["summary"], title="[bold]Summary[/bold]", border_style="green"))


@app.command()
def run(
    video_path: str = typer.Argument(..., help="Path to the video file"),
    query: str = typer.Argument(..., help="Natural language search query"),
    strategy: str = typer.Option("keyframe", help="Extraction strategy: 'keyframe' or 'fixed_fps'"),
    backend: str = typer.Option("mlx_vlm", help="Caption backend: 'mlx_vlm' or 'ollama'"),
    mlx_repo: str = typer.Option("mlx-community/Qwen3.5-122B-A10B-bf16", help="MLX model repo"),
    device: str = typer.Option("mps", help="Device for CLIP"),
):
    """Ingest a video AND search it in one shot."""
    video = Path(video_path)
    if not video.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)

    config = ScreenLensConfig()
    config.frame_extraction.strategy = ExtractionStrategy(strategy)
    config.captioning.backend = CaptionBackend(backend)
    config.captioning.mlx_repo_id = mlx_repo
    config.embedding.device = device

    pipeline = build_full_graph()
    state = {
        "video_path": str(video.resolve()),
        "query": query,
        "config": config.model_dump(),
    }

    result = pipeline.invoke(state)

    if result.get("summary"):
        console.print(Panel(result["summary"], title="[bold]Answer[/bold]", border_style="green"))


@app.command()
def summarize(
    mlx_repo: str = typer.Option(
        "mlx-community/Qwen3.5-122B-A10B-bf16",
        help="HuggingFace repo ID for MLX model (same model used for captioning)"
    ),
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Generate a full-video summary from all ingested captions using the MLX model."""
    config = _load_config(config_file)
    config.captioning.mlx_repo_id = mlx_repo

    model_display = mlx_repo.split("/")[-1]
    console.print(Panel.fit(
        f"[bold green]ScreenLens — Video Summarization[/bold green]\n"
        f"Model: {model_display} (via mlx-vlm)\n"
        f"Captions dir: {config.data_dir / 'captions'}",
        title="Configuration",
    ))

    import time
    t0 = time.time()
    state = {"config": config.model_dump()}
    result = summarize_all_node(state)
    total_time = time.time() - t0

    if result.get("summary"):
        console.print(Panel(
            result["summary"],
            title="[bold]Full Video Summary[/bold]",
            border_style="green",
        ))
        console.print(f"\n[dim]Generated in {total_time:.1f}s[/dim]")


VIDEO_EXTENSIONS = {".mov", ".mp4", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv"}


@app.command()
def batch(
    folder_path: str = typer.Argument(..., help="Path to a folder containing video files"),
    # Extraction strategy
    strategy: str = typer.Option("keyframe", help="Extraction strategy: 'keyframe' (smart) or 'fixed_fps'"),
    fps: float = typer.Option(1.0, help="Frames per second (only for fixed_fps strategy)"),
    max_interval: float = typer.Option(4.0, help="Max seconds between keyframes (keyframe strategy)"),
    # Captioning backend
    backend: str = typer.Option("mlx_vlm", help="Caption backend: 'mlx_vlm' or 'ollama'"),
    mlx_repo: str = typer.Option(
        "mlx-community/Qwen3.5-122B-A10B-bf16",
        help="HuggingFace repo ID for MLX vision model"
    ),
    ollama_model: str = typer.Option("llama3.2-vision", help="Ollama vision model (if backend=ollama)"),
    ollama_url: str = typer.Option("http://127.0.0.1:11434", help="Ollama API URL"),
    # Other
    device: str = typer.Option("mps", help="Device for CLIP: mps, cuda, cpu"),
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Batch-ingest all videos in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        console.print(f"[red]Error: Not a directory: {folder_path}[/red]")
        raise typer.Exit(1)

    videos = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        console.print(f"[yellow]No video files found in {folder_path}[/yellow]")
        console.print(f"[dim]Supported extensions: {', '.join(sorted(VIDEO_EXTENSIONS))}[/dim]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold green]ScreenLens — Batch Ingestion[/bold green]\n"
        f"Folder: {folder.resolve()}\n"
        f"Videos found: {len(videos)}\n"
        f"Backend: {backend} | Strategy: {strategy}",
        title="Batch Configuration",
    ))

    for i, video in enumerate(videos, 1):
        console.print(f"\n[bold cyan]({'='*50})[/bold cyan]")
        console.print(f"[bold cyan]  Video {i}/{len(videos)}: {video.name}[/bold cyan]")
        console.print(f"[bold cyan]({'='*50})[/bold cyan]")

        config = _load_config(config_file)
        config.frame_extraction.strategy = ExtractionStrategy(strategy)
        config.frame_extraction.fps = fps
        config.frame_extraction.max_interval_seconds = max_interval
        config.captioning.backend = CaptionBackend(backend)
        config.captioning.mlx_repo_id = mlx_repo
        config.captioning.ollama_model = ollama_model
        config.captioning.ollama_base_url = ollama_url
        config.embedding.device = device

        # Per-video data directory
        video_slug = video.stem.replace(" ", "_")
        config.data_dir = Path(f"./data/{video_slug}")
        config.vector_db.persist_directory = str(config.data_dir / "chromadb")
        config.vector_db.collection_name = f"screenlens_{video_slug}"

        pipeline = build_ingest_graph()
        initial_state = {
            "video_path": str(video.resolve()),
            "config": config.model_dump(),
        }

        t0 = time.time()
        try:
            result = pipeline.invoke(initial_state)
            elapsed = time.time() - t0
            num_frames = result.get("num_frames", 0)
            console.print(f"[green]  ✓ {video.name} — {num_frames} frames in {elapsed:.1f}s[/green]")
        except Exception as e:
            elapsed = time.time() - t0
            console.print(f"[red]  ✗ {video.name} — failed after {elapsed:.1f}s: {e}[/red]")

    console.print(f"\n[bold green]Batch complete — processed {len(videos)} videos.[/bold green]")


@app.command()
def reconstruct(
    data_dir: str = typer.Option("./data", help="Base data directory containing ingested video folders"),
    mlx_repo: str = typer.Option(
        "mlx-community/Qwen3.5-122B-A10B-bf16",
        help="HuggingFace repo ID for MLX model used for reconstruction"
    ),
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Reconstruct artifacts from ingested video captions.

    Scans all folders in the data directory, classifies each recording
    (Python code, Markdown doc, PDF, or GUI demo), and uses LangGraph
    deep agents to reconstruct the original artifacts with QA reflection.
    """
    from .reconstruct import reconstruct_folder

    base = Path(data_dir)
    if not base.is_dir():
        console.print(f"[red]Error: Data directory not found: {data_dir}[/red]")
        raise typer.Exit(1)

    # Find all folders with captions
    folders = sorted(
        d for d in base.iterdir()
        if d.is_dir() and (d / "captions" / "all_captions.json").exists()
    )

    if not folders:
        console.print(f"[yellow]No ingested video data found in {data_dir}[/yellow]")
        console.print("[dim]Run 'screenlens ingest' first to process videos.[/dim]")
        raise typer.Exit(1)

    config = _load_config(config_file)
    config.captioning.mlx_repo_id = mlx_repo

    console.print(Panel.fit(
        f"[bold green]ScreenLens — Artifact Reconstruction[/bold green]\n"
        f"Data dir: {base.resolve()}\n"
        f"Folders: {len(folders)}\n"
        f"Model: {mlx_repo.split('/')[-1]}",
        title="Reconstruction Pipeline",
    ))

    t0_total = time.time()
    results_summary = []

    for i, folder in enumerate(folders, 1):
        console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
        console.print(f"[bold magenta]  Folder {i}/{len(folders)}: {folder.name}[/bold magenta]")
        console.print(f"[bold magenta]{'='*60}[/bold magenta]")

        t0 = time.time()
        try:
            result = reconstruct_folder(str(folder), config)
            elapsed = time.time() - t0

            if result.get("error"):
                console.print(f"[red]  Error: {result['error']}[/red]")
                results_summary.append((folder.name, "error", elapsed))
                continue

            content_type = result.get("content_type", "unknown")
            saved = result.get("saved_paths", [])
            qa_scores = result.get("qa_scores", {})
            overall_qa = qa_scores.get("completeness", 0)

            console.print(
                f"[green]  Reconstructed: {content_type} | "
                f"{len(saved)} files | QA: {json.dumps(qa_scores)} | "
                f"{elapsed:.1f}s[/green]"
            )
            for path in saved:
                console.print(f"[dim]    {path}[/dim]")

            results_summary.append((folder.name, content_type, elapsed))

        except Exception as e:
            elapsed = time.time() - t0
            console.print(f"[red]  Failed after {elapsed:.1f}s: {e}[/red]")
            results_summary.append((folder.name, "failed", elapsed))

    total_time = time.time() - t0_total

    # Summary table
    console.print(f"\n")
    table = Table(title="Reconstruction Summary")
    table.add_column("Folder", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Time (s)", justify="right")
    for name, ctype, elapsed in results_summary:
        style = "red" if ctype in ("error", "failed") else ""
        table.add_row(name, ctype, f"{elapsed:.1f}", style=style)
    table.add_row("Total", "", f"{total_time:.1f}", style="bold")
    console.print(table)


@app.command()
def info(
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Show info about the current vector store."""
    config = _load_config(config_file)
    from .vector_store import ScreenLensVectorStore

    store = ScreenLensVectorStore(config.vector_db)
    count = store.count()

    console.print(Panel.fit(
        f"Collection: {config.vector_db.collection_name}\n"
        f"Frames stored: {count}\n"
        f"Persist dir: {config.vector_db.persist_directory}",
        title="[bold]Vector Store Info[/bold]",
    ))

    if count > 0:
        frames = store.get_all_frames()
        if frames:
            first_ts = frames[0].get("timestamp_str", "?")
            last_ts = frames[-1].get("timestamp_str", "?")
            console.print(f"  Time range: {first_ts} — {last_ts}")


if __name__ == "__main__":
    app()
