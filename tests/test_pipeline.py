"""
Integration tests for the ScreenLens pipeline.

Run with: pytest tests/test_pipeline.py -v
"""
import json
import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

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

    def test_dgx_spark_defaults(self, monkeypatch):
        import src.config as config_module
        from src.config import CaptionBackend, InferenceBackend, ScreenLensConfig

        monkeypatch.delenv("SCREENLENS_BACKEND", raising=False)
        monkeypatch.delenv("SCREENLENS_DEVICE", raising=False)
        monkeypatch.delenv("SCREENLENS_BATCH_SIZE", raising=False)
        monkeypatch.setattr(config_module, "_DOTENV_LOADED", True)
        monkeypatch.setattr(config_module.platform, "system", lambda: "Linux")
        monkeypatch.setattr(config_module.platform, "machine", lambda: "aarch64")

        config = ScreenLensConfig()
        assert config.captioning.backend == CaptionBackend.vllm
        assert config.captioning.vllm_base_url == "http://127.0.0.1:8000/v1"
        assert config.captioning.disable_thinking is True
        assert config.captioning.max_tokens == 32768
        assert config.captioning.batch_size == 2
        assert config.ocr.backend == InferenceBackend.vllm
        assert config.frame_extraction.fps == 1.0
        assert config.embedding.device == "cuda"
        assert config.vector_db.collection_name == "screenlens_frames"

    def test_apple_silicon_defaults(self, monkeypatch):
        import src.config as config_module
        from src.config import CaptionBackend, ScreenLensConfig

        monkeypatch.delenv("SCREENLENS_BACKEND", raising=False)
        monkeypatch.delenv("SCREENLENS_DEVICE", raising=False)
        monkeypatch.delenv("SCREENLENS_BATCH_SIZE", raising=False)
        monkeypatch.setattr(config_module, "_DOTENV_LOADED", True)
        monkeypatch.setattr(config_module.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(config_module.platform, "machine", lambda: "arm64")

        config = ScreenLensConfig()
        assert config.captioning.backend == CaptionBackend.omlx
        assert config.captioning.max_tokens == 32768
        assert config.captioning.batch_size == 4
        assert config.embedding.device == "mps"

    def test_platform_defaults_accept_environment_overrides(self, monkeypatch):
        from src.config import CaptionBackend, InferenceBackend, ScreenLensConfig

        monkeypatch.setenv("SCREENLENS_BACKEND", "ollama")
        monkeypatch.setenv("SCREENLENS_DEVICE", "cpu")
        monkeypatch.setenv("SCREENLENS_BATCH_SIZE", "7")

        config = ScreenLensConfig()
        assert config.captioning.backend == CaptionBackend.ollama
        assert config.ocr.backend in (InferenceBackend.vllm, InferenceBackend.omlx)
        assert config.captioning.batch_size == 7
        assert config.embedding.device == "cpu"

    def test_dotenv_applies_platform_default_overrides(self, monkeypatch, tmp_path):
        import src.config as config_module
        from src.config import CaptionBackend, ScreenLensConfig

        (tmp_path / ".env").write_text(
            "SCREENLENS_BACKEND=ollama\n"
            "SCREENLENS_DEVICE=cpu\n"
            "SCREENLENS_BATCH_SIZE=3\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SCREENLENS_BACKEND", raising=False)
        monkeypatch.delenv("SCREENLENS_DEVICE", raising=False)
        monkeypatch.delenv("SCREENLENS_BATCH_SIZE", raising=False)
        monkeypatch.setattr(config_module, "_DOTENV_LOADED", False)

        config = ScreenLensConfig()

        assert config.captioning.backend == CaptionBackend.ollama
        assert config.captioning.batch_size == 3
        assert config.embedding.device == "cpu"

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


class TestOMLXClient:
    """Test the oMLX OpenAI-compatible adapter without network access."""

    def test_normalizes_dashboard_url(self):
        from src.omlx_client import normalize_omlx_base_url

        assert (
            normalize_omlx_base_url("http://127.0.0.1:8000/admin/dashboard?tab=status")
            == "http://127.0.0.1:8000/v1"
        )
        assert normalize_omlx_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1"
        assert normalize_omlx_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"

    def test_dotenv_loads_omlx_values_without_overriding_shell(self, monkeypatch, tmp_path):
        import src.config as config_module
        from src.config import CaptioningConfig
        import src.omlx_client as omlx_client

        (tmp_path / ".env").write_text(
            "\n".join([
                "MLX_API_KEY=your-omlx-api-key-here",
                "OMLX_API_KEY=dotenv-key",
                "MLX_MODEL=dotenv-model",
            ]),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("MLX_API_KEY", raising=False)
        monkeypatch.delenv("OMLX_API_KEY", raising=False)
        monkeypatch.delenv("MLX_MODEL", raising=False)
        monkeypatch.setattr(config_module, "_DOTENV_LOADED", False)

        assert omlx_client.resolve_omlx_api_key(CaptioningConfig()) == "dotenv-key"
        assert omlx_client.resolve_omlx_model(CaptioningConfig()) == "dotenv-model"

    def test_rejects_known_text_only_models_for_image_chat(self):
        from src.config import CaptionBackend, CaptioningConfig
        from src.omlx_client import OMLXClient

        client = OMLXClient(CaptioningConfig(
            backend=CaptionBackend.omlx,
            omlx_model="deepseek-ai-DeepSeek-V4-Flash-8bit",
        ))

        with pytest.raises(ValueError, match="text-only model"):
            client.chat("system", "describe", images=["missing.jpg"])

    def test_tui_hides_known_text_only_omlx_models(self):
        from src.tui import _omlx_model_options

        options = _omlx_model_options(
            [
                "MiniMax-M2.7",
                "deepseek-ai-DeepSeek-V4-Flash-8bit",
                "gpt-oss-120b-MXFP4-Q8",
            ],
            "deepseek-ai-DeepSeek-V4-Flash-8bit",
        )

        assert options == []

    def test_tui_summary_supports_ollama_backend(self):
        from src.config import CaptionBackend, ScreenLensConfig
        from src.tui import _summary_rows

        config = ScreenLensConfig()
        config.captioning.backend = CaptionBackend.ollama
        rows = dict(_summary_rows(config, None))

        assert rows["Inference URL"] == config.captioning.ollama_base_url
        assert rows["Inference key"] == "n/a"

    def test_vllm_defaults_and_legacy_env_isolation(self, monkeypatch):
        from src.config import CaptionBackend, CaptioningConfig, OCRConfig, ReconstructionConfig
        from src.omlx_client import (
            DEFAULT_VLLM_MODEL,
            resolve_inference_api_key,
            resolve_inference_base_url,
            resolve_inference_context,
            resolve_inference_model,
            resolve_llm_model,
            resolve_ocr_model,
            resolve_role_api_key,
            resolve_role_context,
        )

        monkeypatch.setenv("MLX_MODEL", "legacy-mlx-model")
        monkeypatch.setenv("OCR_MODEL", "legacy-ocr-model")
        monkeypatch.setenv("LLM_MODEL", "legacy-text-model")
        monkeypatch.setenv("VLLM_BASE_URL", "http://spark.local:9000/v1/")
        monkeypatch.setenv("VLLM_API_KEY", "spark-secret")
        monkeypatch.delenv("VLLM_MODEL", raising=False)

        captioning = CaptioningConfig(backend=CaptionBackend.vllm)
        assert resolve_inference_base_url(captioning) == "http://spark.local:9000/v1"
        assert resolve_inference_api_key(captioning) == "spark-secret"
        assert resolve_inference_model(captioning) == DEFAULT_VLLM_MODEL
        assert resolve_ocr_model(OCRConfig(backend="vllm")) == DEFAULT_VLLM_MODEL
        assert resolve_llm_model(ReconstructionConfig(backend="vllm")) == DEFAULT_VLLM_MODEL

        monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "16384")
        assert resolve_inference_context(captioning) == 16384
        assert resolve_role_context(ReconstructionConfig(backend="vllm")) == 16384
        assert resolve_inference_context(
            CaptioningConfig(backend=CaptionBackend.vllm, vllm_model_context=24576)
        ) == 24576

        monkeypatch.setenv("VLLM_OCR_API_KEY", "spark-ocr-secret")
        monkeypatch.setenv("OCR_API_KEY", "legacy-ocr-secret")
        assert resolve_role_api_key(
            OCRConfig(backend="vllm"), "VLLM_OCR_API_KEY", "OCR_API_KEY"
        ) == "spark-ocr-secret"
        assert resolve_role_api_key(
            OCRConfig(backend="omlx"), "VLLM_OCR_API_KEY", "OCR_API_KEY"
        ) == "legacy-ocr-secret"

    def test_nvidia_qwen_spark_model_is_known_multimodal(self):
        from src.omlx_client import DEFAULT_VLLM_MODEL, is_known_vision_model

        assert is_known_vision_model(DEFAULT_VLLM_MODEL)

    def test_loopback_requests_bypass_proxy_environment(self, monkeypatch):
        from urllib import request
        import src.omlx_client as inference_client

        captured = {}
        sentinel = object()

        class FakeOpener:
            def open(self, req, timeout):
                captured["url"] = req.full_url
                captured["timeout"] = timeout
                return sentinel

        def fake_build_opener(*handlers):
            captured["handlers"] = handlers
            return FakeOpener()

        monkeypatch.setattr(inference_client.request, "build_opener", fake_build_opener)
        monkeypatch.setattr(
            inference_client.request,
            "urlopen",
            lambda *args, **kwargs: pytest.fail("loopback request inherited proxy handling"),
        )

        result = inference_client._urlopen(
            request.Request("http://127.0.0.1:8000/v1/models"),
            timeout=3,
        )

        assert result is sentinel
        assert captured["timeout"] == 3
        assert captured["handlers"][0].proxies == {}

    def test_chat_posts_openai_vision_payload(self, monkeypatch, tmp_path):
        from PIL import Image
        from src.config import CaptioningConfig
        from src.omlx_client import OMLXClient
        import src.omlx_client as omlx_client

        img_path = tmp_path / "frame.jpg"
        Image.new("RGB", (4, 4), color="red").save(img_path)

        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "<think>hidden</think>visible caption"}}]
                }).encode("utf-8")

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        monkeypatch.setattr(omlx_client, "_urlopen", fake_urlopen)

        from src.config import CaptionBackend
        cfg = CaptioningConfig(
            backend=CaptionBackend.omlx,
            omlx_base_url="http://127.0.0.1:8000/admin/dashboard",
            omlx_model="vision-model",
            omlx_api_key="local-key",
            omlx_timeout_seconds=12,
        )
        result = OMLXClient(cfg).chat("system", "describe", images=[str(img_path)])

        assert result == "visible caption"
        assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
        assert captured["headers"]["Authorization"] == "Bearer local-key"
        assert captured["timeout"] == 12
        assert captured["payload"]["model"] == "vision-model"
        user_content = captured["payload"]["messages"][1]["content"]
        assert user_content[0] == {"type": "text", "text": "describe"}
        assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    @pytest.mark.parametrize(
        ("backend", "context_size", "default_max_tokens", "expected_max_tokens"),
        [
            ("vllm", 32768, 32768, None),
            ("vllm", 65536, 32768, 32768),
            ("vllm", 32768, 4096, 4096),
            ("omlx", 32768, 32768, 32768),
        ],
    )
    def test_chat_uses_remaining_vllm_context_at_full_ceiling(
        self,
        backend,
        context_size,
        default_max_tokens,
        expected_max_tokens,
        monkeypatch,
    ):
        from src.omlx_client import InferenceClient
        import src.omlx_client as inference_client

        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{
                        "message": {"content": "complete"},
                        "finish_reason": "stop",
                    }],
                }).encode("utf-8")

        def fake_urlopen(req, timeout):
            captured.update(json.loads(req.data.decode("utf-8")))
            return FakeResponse()

        monkeypatch.setattr(inference_client, "_urlopen", fake_urlopen)
        client = InferenceClient.from_endpoint(
            base_url="http://127.0.0.1:8000/v1",
            model="vision-model",
            api_key="local",
            backend=backend,
            context_size=context_size,
            default_max_tokens=default_max_tokens,
        )

        assert client.chat("system", "user") == "complete"
        if expected_max_tokens is None:
            assert "max_tokens" not in captured
        else:
            assert captured["max_tokens"] == expected_max_tokens

    def test_vllm_context_overflow_retries_with_exact_remaining_tokens(
        self,
        monkeypatch,
    ):
        from src.omlx_client import InferenceClient
        import src.omlx_client as inference_client

        chat_payloads = []
        tokenize_payloads = []

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(req, timeout):
            payload = json.loads(req.data.decode("utf-8"))
            if req.full_url.endswith("/tokenize"):
                tokenize_payloads.append(payload)
                return FakeResponse({"count": 30721, "max_model_len": 32768, "tokens": []})

            chat_payloads.append(payload)
            if len(chat_payloads) == 1:
                detail = json.dumps({
                    "error": {
                        "message": (
                            "This model's maximum context length is 32768 tokens. "
                            "However, you requested 2048 output tokens and your prompt "
                            "contains at least 30721 input tokens."
                        ),
                        "type": "BadRequestError",
                        "param": "input_tokens",
                        "code": 400,
                    },
                }).encode("utf-8")
                raise HTTPError(req.full_url, 400, "Bad Request", {}, BytesIO(detail))
            return FakeResponse({
                "choices": [{
                    "message": {"content": "recovered"},
                    "finish_reason": "stop",
                }],
            })

        monkeypatch.setattr(inference_client, "_urlopen", fake_urlopen)
        client = InferenceClient.from_endpoint(
            base_url="http://127.0.0.1:8000/v1",
            model="vision-model",
            api_key="local",
            backend="vllm",
            context_size=32768,
            default_max_tokens=4096,
        )

        result = client.chat(
            "system",
            "large prompt",
            max_tokens=2048,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
        )

        assert result == "recovered"
        assert [payload["max_tokens"] for payload in chat_payloads] == [2048, 2047]
        assert tokenize_payloads == [{
            "model": "vision-model",
            "messages": chat_payloads[0]["messages"],
            "chat_template_kwargs": {"enable_thinking": False},
        }]

    def test_vllm_context_retry_rejects_prompt_larger_than_context(self, monkeypatch):
        from src.omlx_client import InferenceClient

        client = InferenceClient.from_endpoint(
            base_url="http://127.0.0.1:8000/v1",
            model="vision-model",
            api_key="local",
            backend="vllm",
            context_size=32768,
        )
        monkeypatch.setattr(client, "_tokenize_chat", lambda payload: (40012, 32768))
        detail = json.dumps({
            "error": {
                "message": "This model's maximum context length is 32768 tokens.",
                "param": "input_tokens",
            },
        })

        with pytest.raises(RuntimeError, match="prompt uses 40,012 tokens"):
            client._context_retry_payload({"max_tokens": 2048}, 400, detail)


class TestCaptioner:
    """Test caption generation controls without contacting oMLX."""

    @pytest.mark.parametrize("disable_thinking", [True, False])
    def test_omlx_captioner_controls_model_thinking(
        self,
        disable_thinking,
        monkeypatch,
        tmp_path,
    ):
        from PIL import Image
        from src.captioner import OMLXCaptioner
        from src.config import CaptionBackend, CaptioningConfig

        img_path = tmp_path / "frame.jpg"
        Image.new("RGB", (4, 4), color="blue").save(img_path)

        config = CaptioningConfig(
            backend=CaptionBackend.omlx,
            omlx_model="vision-model",
            disable_thinking=disable_thinking,
        )
        captioner = OMLXCaptioner(config)
        captured = {}

        def fake_post(payload):
            captured.update(payload)
            return "visible caption"

        monkeypatch.setattr(captioner._client, "_post_chat", fake_post)

        assert captioner.caption(str(img_path)) == "visible caption"
        assert captured["repetition_penalty"] == 1.05
        assert captured["no_repeat_ngram_size"] == 12
        if disable_thinking:
            assert captured["chat_template_kwargs"] == {"enable_thinking": False}
        else:
            assert "chat_template_kwargs" not in captured


class TestEmbedder:
    """Test CLIP embedding generation."""

    @pytest.fixture
    def embedder(self, monkeypatch):
        """Use a deterministic local OpenCLIP stand-in; live CUDA is helper-smoked."""
        import sys
        from types import SimpleNamespace
        import numpy as np
        import torch

        class FakeModel:
            visual = SimpleNamespace(output_dim=512)

            def eval(self):
                return self

            def encode_image(self, images):
                rgb = images.mean(dim=(-2, -1))
                repeats = (512 + rgb.shape[1] - 1) // rgb.shape[1]
                return rgb.repeat(1, repeats)[:, :512]

            def encode_text(self, tokens):
                rows = torch.arange(1, tokens.shape[0] + 1, dtype=torch.float32)
                return rows[:, None].repeat(1, 512)

        def preprocess(image):
            array = np.asarray(image, dtype=np.float32) / 255.0
            return torch.from_numpy(array).permute(2, 0, 1)

        fake_open_clip = SimpleNamespace(
            create_model_and_transforms=lambda *args, **kwargs: (
                FakeModel(), None, preprocess
            ),
            get_tokenizer=lambda model_name: (
                lambda queries: torch.ones((len(queries), 4), dtype=torch.long)
            ),
        )
        monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

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

    def test_search_summary_uses_selected_vllm_client(self, monkeypatch):
        import src.pipeline as pipeline
        from src.config import CaptionBackend, ScreenLensConfig

        captured = {}

        class FakeClient:
            def __init__(self, config):
                captured["backend"] = config.backend

            def chat(self, system, user, **kwargs):
                captured["system"] = system
                captured["user"] = user
                captured["kwargs"] = kwargs
                return "DGX summary"

        monkeypatch.setattr(pipeline, "InferenceClient", FakeClient)
        config = ScreenLensConfig()
        config.captioning.backend = CaptionBackend.vllm

        result = pipeline.summarize_node({
            "query": "What application is shown?",
            "search_results": [{
                "timestamp_str": "00:00:01.000",
                "caption": "A terminal shows ScreenLens.",
                "score": 0.9,
            }],
            "config": config.model_dump(),
        })

        assert result["summary"] == "DGX summary"
        assert captured["backend"] == CaptionBackend.vllm
        assert captured["kwargs"]["extra"] == {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def test_caption_chunks_budget_each_skewed_caption_in_order(self):
        from src.pipeline import (
            _chunk_captions_by_budget,
            _compute_chunk_strategy,
            _estimated_caption_tokens,
        )

        captions = [
            {
                "frame_id": i,
                "timestamp_str": f"00:00:{i:02d}.000",
                "caption": "x" * 3000,
            }
            for i in range(54)
        ]
        captions[20]["caption"] = "runaway `...`, " * 5000  # ~75K chars

        strategy = _compute_chunk_strategy(captions, 32768)
        chunks = _chunk_captions_by_budget(
            captions,
            strategy["safe_context_tokens"],
        )

        assert strategy["strategy"] == "hierarchical"
        assert len(chunks) > 2
        flattened = [item for chunk in chunks for item in chunk]
        frame_ids = [item["frame_id"] for item in flattened]
        assert frame_ids == sorted(frame_ids)
        rebuilt = {i: "" for i in range(54)}
        for item in flattened:
            rebuilt[item["frame_id"]] += item["caption"]
        assert [rebuilt[i] for i in range(54)] == [item["caption"] for item in captions]
        assert all(
            sum(_estimated_caption_tokens(item) for item in chunk)
            <= strategy["safe_context_tokens"]
            for chunk in chunks
        )

    def test_caption_chunks_split_one_caption_larger_than_budget(self):
        from src.pipeline import _chunk_captions_by_budget, _estimated_caption_tokens

        original = "0123456789" * 1500
        chunks = _chunk_captions_by_budget(
            [{"frame_id": 7, "timestamp_str": "00:00:07.000", "caption": original}],
            1000,
        )
        pieces = [item for chunk in chunks for item in chunk]

        assert len(pieces) > 1
        assert "".join(item["caption"] for item in pieces) == original
        assert all(_estimated_caption_tokens(item) <= 1000 for item in pieces)

    def test_reconstruction_synthesis_keeps_spark_context_headroom(self, monkeypatch):
        import src.reconstruct as reconstruct

        captured = {}

        class LegacyClient:
            _default_max_tokens = 32768

        def fake_generate(client, system, user, *, max_tokens, temperature):
            captured["max_tokens"] = max_tokens
            return "artifact"

        monkeypatch.setattr(reconstruct, "generate_text", fake_generate)

        result = reconstruct._hierarchical_synthesize(
            ["a short extraction note"],
            "Rebuild the artifact.",
            "Return only the artifact.",
            LegacyClient(),
            model_context=32768,
        )

        assert result == "artifact"
        assert captured["max_tokens"] == 8192
        assert captured["max_tokens"] < 32768

    def test_reconstruction_synthesis_splits_one_oversized_note(self, monkeypatch):
        import src.reconstruct as reconstruct

        calls = []

        class LegacyClient:
            _default_max_tokens = 32768

        def fake_generate(client, system, user, *, max_tokens, temperature):
            calls.append({"user": user, "max_tokens": max_tokens})
            return f"condensed-{len(calls)}"

        monkeypatch.setattr(reconstruct, "generate_text", fake_generate)

        result = reconstruct._hierarchical_synthesize(
            ["`...`, " * 18000],
            "Rebuild the artifact.",
            "Return only the artifact.",
            LegacyClient(),
            model_context=32768,
        )

        assert result.startswith("condensed-")
        assert len(calls) >= 4
        assert all(len(call["user"]) < 40000 for call in calls)
        assert calls[-1]["max_tokens"] == 8192

    def test_ollama_caption_config_uses_direct_reconstruction_backend(self):
        import src.reconstruct as reconstruct
        from src.config import CaptionBackend, InferenceBackend, ScreenLensConfig

        config = ScreenLensConfig()
        config.captioning.backend = CaptionBackend.ollama
        config.reconstruction.backend = InferenceBackend.vllm
        config.reconstruction.model = "org/reconstruction-model"
        config.reconstruction.api_key = "direct-key"

        direct = reconstruct._reconstruction_captioning_config(config)

        assert direct.backend == CaptionBackend.vllm
        assert direct.vllm_model == "org/reconstruction-model"
        assert direct.vllm_api_key == "direct-key"
        assert direct.max_tokens == config.reconstruction.max_tokens
