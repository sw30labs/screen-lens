"""
Configuration for the ScreenLens pipeline.
All settings are centralized here for easy tuning.
"""
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Frame Extraction ────────────────────────────────────────────────────────

class ExtractionStrategy(str, Enum):
    """How to decide which frames to extract."""
    fixed_fps = "fixed_fps"       # Simple: 1 frame every N seconds
    keyframe = "keyframe"         # Smart: hybrid change detection (SSIM + pHash + HSV)


class FrameExtractionConfig(BaseModel):
    """Settings for video frame extraction."""
    strategy: ExtractionStrategy = Field(
        default=ExtractionStrategy.keyframe,
        description="Extraction strategy: 'keyframe' (smart, recommended) or 'fixed_fps'"
    )
    # Fixed FPS settings
    fps: float = Field(default=1.0, description="Frames per second (only for fixed_fps strategy)")
    # Keyframe detection settings (hybrid change detector)
    ssim_threshold: float = Field(default=0.97, description="SSIM below this = scene change")
    phash_threshold: int = Field(default=8, description="Perceptual hash hamming distance threshold")
    hist_corr_threshold: float = Field(default=0.90, description="HSV histogram correlation threshold")
    min_interval_seconds: float = Field(default=0.5, description="Min seconds between keyframes")
    max_interval_seconds: float = Field(default=4.0, description="Force a keyframe at least this often")
    min_changed_area: float = Field(default=0.02, description="Min fraction of pixels that must change")
    # Shared settings
    max_dimension: int = Field(default=1280, description="Max width or height for extracted frames")
    output_format: str = Field(default="jpg", description="Frame image format (jpg, png)")
    quality: int = Field(default=85, description="JPEG quality (1-100)")


# ── Captioning ──────────────────────────────────────────────────────────────

class CaptionBackend(str, Enum):
    """Which vision model backend to use for captioning."""
    mlx_vlm = "mlx_vlm"    # Qwen3.5-VL via mlx-vlm (Apple Silicon native)
    ollama = "ollama"       # Any Ollama vision model (llama3.2-vision, etc.)


class CaptioningConfig(BaseModel):
    """Settings for frame captioning."""
    backend: CaptionBackend = Field(
        default=CaptionBackend.mlx_vlm,
        description="Vision model backend: 'mlx_vlm' (recommended on Apple Silicon) or 'ollama'"
    )
    # MLX-VLM settings (Qwen3.5)
    mlx_repo_id: str = Field(
        default="mlx-community/Qwen3.5-122B-A10B-bf16",
        description="HuggingFace repo ID for the MLX vision model"
    )
    mlx_model_path: Optional[str] = Field(
        default=None,
        description="Override: local path to MLX model weights (auto-resolved from repo_id if None)"
    )
    # Ollama settings (fallback)
    ollama_model: str = Field(default="llama3.2-vision", description="Ollama vision model name")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", description="Ollama API endpoint")
    # Shared generation settings
    temperature: float = Field(default=0.1, description="LLM temperature for captions")
    max_tokens: int = Field(default=1024, description="Max tokens per caption")
    batch_size: int = Field(
        default=4,
        description=(
            "Frames per mlx-vlm batch_generate call (ignored by Ollama backend). "
            "Empirically tuned on M3 Ultra 512GB with Qwen3.5-122B-A10B-bf16: "
            "batch=4 → 1.54x speedup vs batch=1, batch=8 *regresses* to 1.22x "
            "(likely MoE expert dispersion + compute-bound prefill). Memory is "
            "not the bottleneck (peak ~257GB at batch=4 of 512GB UMA). Smaller "
            "MoE models (e.g. 35B-A3B-4bit) should tolerate larger batch sizes — "
            "re-run scripts/bench_caption_batch.py if you change mlx_repo_id."
        ),
    )
    system_prompt: str = Field(
        default=(
            "You are a meticulous video frame analyst. You respond ONLY with your analysis — "
            "no preamble, no thinking, no meta-commentary. Output raw Markdown directly."
        ),
        description="System prompt for the vision model"
    )
    user_prompt: str = Field(
        default=(
            "Analyze this video frame and describe everything visible. "
            "Respond directly with the analysis — no preamble or reasoning.\n\n"
            "1. **Text**: Reproduce all visible text exactly, preserving hierarchy.\n"
            "2. **UI Elements**: Describe buttons, menus, toolbars, dialogs, and their states.\n"
            "3. **Tables**: Render any tables as Markdown tables.\n"
            "4. **Diagrams**: Describe any diagrams, charts, or visual flows.\n"
            "5. **Actions**: Note what action or interaction appears to be happening.\n\n"
            "IMPORTANT: Ignore browser chrome and OS window decorations. "
            "Focus only on application content."
        ),
        description="User prompt sent with each frame"
    )


# ── Embedding ───────────────────────────────────────────────────────────────

class EmbeddingConfig(BaseModel):
    """Settings for CLIP embedding generation."""
    model_name: str = Field(default="ViT-B-32", description="OpenCLIP model architecture")
    pretrained: str = Field(default="laion2b_s34b_b79k", description="Pretrained weights")
    batch_size: int = Field(default=64, description="Batch size for embedding generation")
    device: str = Field(default="mps", description="Device: mps (Apple Silicon), cuda, or cpu")


# ── Vector DB ───────────────────────────────────────────────────────────────

class VectorDBConfig(BaseModel):
    """Settings for ChromaDB vector storage."""
    collection_name: str = Field(default="screenlens_frames", description="ChromaDB collection name")
    persist_directory: str = Field(default="./data/chromadb", description="ChromaDB storage path")
    distance_metric: str = Field(default="cosine", description="Distance metric for similarity")


# ── Search & Summarization ──────────────────────────────────────────────────

class SearchConfig(BaseModel):
    """Settings for search and summarization."""
    top_k: int = Field(default=10, description="Number of results to return")
    summarization_model: str = Field(default="llama3.2", description="Ollama model for summarization")
    base_url: str = Field(default="http://127.0.0.1:11434", description="Ollama API endpoint")


# ── Top-Level Config ────────────────────────────────────────────────────────

class ScreenLensConfig(BaseModel):
    """Top-level configuration for the entire ScreenLens pipeline."""
    frame_extraction: FrameExtractionConfig = Field(default_factory=FrameExtractionConfig)
    captioning: CaptioningConfig = Field(default_factory=CaptioningConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    data_dir: Path = Field(default=Path("./data"), description="Base data directory")

    def ensure_dirs(self):
        """Create all necessary data directories."""
        (self.data_dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "captions").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "embeddings").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "chromadb").mkdir(parents=True, exist_ok=True)
