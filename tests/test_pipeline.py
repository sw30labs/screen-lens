"""
Integration tests for the ScreenLens pipeline.

Run with: pytest tests/test_pipeline.py -v
"""
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Load test cases
TEST_CASES_PATH = Path(__file__).parent / "test_cases.yaml"


def load_test_cases():
    """Load test case definitions from YAML."""
    if TEST_CASES_PATH.exists():
        with open(TEST_CASES_PATH) as f:
            return yaml.safe_load(f)
    return {"test_cases": []}


class TestConfig:
    """Test the configuration system."""

    def test_default_config(self):
        from src.config import ScreenLensConfig
        config = ScreenLensConfig()
        assert config.frame_extraction.fps == 1.0
        assert config.embedding.device == "mps"
        assert config.vector_db.collection_name == "screenlens_frames"

    def test_config_override(self):
        from src.config import ScreenLensConfig
        config = ScreenLensConfig()
        config.frame_extraction.fps = 0.5
        assert config.frame_extraction.fps == 0.5

    def test_ensure_dirs(self):
        from src.config import ScreenLensConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ScreenLensConfig(data_dir=Path(tmpdir) / "test_data")
            config.ensure_dirs()
            assert (config.data_dir / "frames").exists()
            assert (config.data_dir / "captions").exists()
            assert (config.data_dir / "embeddings").exists()


class TestFrameExtractor:
    """Test frame extraction (requires ffmpeg)."""

    def test_format_timestamp(self):
        from src.frame_extractor import _format_timestamp
        assert _format_timestamp(0) == "00:00:00.000"
        assert _format_timestamp(65.5) == "00:01:05.500"
        assert _format_timestamp(3661.123) == "01:01:01.123"

    def test_resize_frame(self):
        from PIL import Image
        from src.frame_extractor import _resize_frame

        img = Image.new("RGB", (1920, 1080))
        resized = _resize_frame(img, 1280)
        assert max(resized.size) <= 1280

        small = Image.new("RGB", (640, 480))
        same = _resize_frame(small, 1280)
        assert same.size == (640, 480)


class TestEmbedder:
    """Test CLIP embedding generation."""

    @pytest.fixture
    def embedder(self):
        from src.config import EmbeddingConfig
        from src.embedder import CLIPEmbedder
        config = EmbeddingConfig(device="cpu")
        return CLIPEmbedder(config)

    def test_embed_text(self, embedder):
        """Test text embedding generation."""
        embs = embedder.embed_text(["a cat sitting on a mat"])
        assert embs.shape[0] == 1
        assert embs.shape[1] == 512  # ViT-B-32

    def test_embed_images(self, embedder, tmp_path):
        """Test image embedding generation."""
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="red")
        img_path = str(tmp_path / "test.jpg")
        img.save(img_path)

        embs = embedder.embed_images([img_path])
        assert embs.shape == (1, 512)

    def test_embedding_similarity(self, embedder, tmp_path):
        """Test that similar content produces similar embeddings."""
        import numpy as np
        from PIL import Image

        # Create two similar images (red) and one different (blue)
        red1 = Image.new("RGB", (224, 224), color="red")
        red2 = Image.new("RGB", (224, 224), color=(255, 10, 10))
        blue = Image.new("RGB", (224, 224), color="blue")

        for name, img in [("red1.jpg", red1), ("red2.jpg", red2), ("blue.jpg", blue)]:
            img.save(str(tmp_path / name))

        embs = embedder.embed_images([
            str(tmp_path / "red1.jpg"),
            str(tmp_path / "red2.jpg"),
            str(tmp_path / "blue.jpg"),
        ])

        # Red images should be more similar to each other than to blue
        sim_red = np.dot(embs[0], embs[1])
        sim_diff = np.dot(embs[0], embs[2])
        assert sim_red > sim_diff, "Similar images should have higher similarity"


class TestVectorStore:
    """Test ChromaDB vector store operations."""

    @pytest.fixture
    def store(self, tmp_path):
        from src.config import VectorDBConfig
        from src.vector_store import ScreenLensVectorStore
        config = VectorDBConfig(
            persist_directory=str(tmp_path / "chromadb"),
            collection_name="test_collection",
        )
        return ScreenLensVectorStore(config)

    def test_add_and_count(self, store):
        import numpy as np
        frames = [
            {"frame_id": 0, "timestamp": 0.0, "timestamp_str": "00:00:00.000",
             "path": "/tmp/f0.jpg", "caption": "A red screen"},
            {"frame_id": 1, "timestamp": 1.0, "timestamp_str": "00:00:01.000",
             "path": "/tmp/f1.jpg", "caption": "A blue menu bar"},
        ]
        embeddings = np.random.randn(2, 512).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        store.add_frames(frames, embeddings)
        assert store.count() == 2

    def test_search_by_embedding(self, store):
        import numpy as np
        frames = [
            {"frame_id": 0, "timestamp": 0.0, "timestamp_str": "00:00:00.000",
             "path": "/tmp/f0.jpg", "caption": "Red screen"},
        ]
        emb = np.random.randn(1, 512).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        store.add_frames(frames, emb)

        results = store.search_by_embedding(emb[0], top_k=1)
        assert len(results) == 1
        assert results[0]["caption"] == "Red screen"

    def test_reset(self, store):
        import numpy as np
        frames = [{"frame_id": 0, "timestamp": 0.0, "path": "/tmp/f.jpg", "caption": "test"}]
        emb = np.random.randn(1, 512).astype(np.float32)
        store.add_frames(frames, emb)
        assert store.count() == 1
        store.reset()
        assert store.count() == 0


class TestPipeline:
    """Test LangGraph pipeline construction."""

    def test_ingest_graph_builds(self):
        from src.pipeline import build_ingest_graph
        graph = build_ingest_graph()
        assert graph is not None

    def test_search_graph_builds(self):
        from src.pipeline import build_search_graph
        graph = build_search_graph()
        assert graph is not None

    def test_full_graph_builds(self):
        from src.pipeline import build_full_graph
        graph = build_full_graph()
        assert graph is not None
