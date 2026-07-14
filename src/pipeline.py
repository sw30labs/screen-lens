"""
LangGraph ScreenLens Pipeline.

Orchestrates the full ScreenLens workflow:
  1. Ingest — extract keyframes from video (hybrid change detection)
  2. Caption — generate dense captions (vLLM, oMLX, or Ollama)
  3. Embed — generate CLIP embeddings for semantic search
  4. Store — persist embeddings + metadata in ChromaDB
  5. Search — semantic query (text → CLIP → vector search)
  6. Summarize — LLM-powered answer generation from search results

Uses LangGraph's StateGraph for explicit state management and checkpointing.
"""
import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from .config import ScreenLensConfig
from .frame_extractor import extract_frames, get_video_metadata
from .captioner import caption_frames
from .embedder import CLIPEmbedder
from .omlx_client import (
    InferenceClient,
    resolve_inference_context,
    resolve_inference_model,
)
from .vector_store import ScreenLensVectorStore


# ── Pipeline State ──────────────────────────────────────────────────────────

class ScreenLensState(TypedDict, total=False):
    """Shared state flowing through the LangGraph pipeline."""
    # Input
    video_path: str
    query: str
    config: dict  # Serialized ScreenLensConfig

    # Frame extraction
    video_metadata: dict
    frames_meta: list[dict]
    num_frames: int

    # Captioning
    captioned_frames: list[dict]

    # Embedding
    embeddings_shape: list[int]  # [N, dim]

    # Search results
    search_results: list[dict]

    # Summary
    summary: str

    # Pipeline status
    stage: str
    error: str
    elapsed_seconds: dict  # timing per stage


# ── Pipeline Nodes ──────────────────────────────────────────────────────────

def ingest_node(state: ScreenLensState) -> dict:
    """Extract frames from the input video using configured strategy."""
    t0 = time.time()
    config = ScreenLensConfig(**state["config"])
    config.ensure_dirs()

    video_path = state["video_path"]
    output_dir = str(config.data_dir / "frames")

    print(f"\n{'='*60}")
    print(f"[1/4] INGESTING VIDEO: {Path(video_path).name}")
    print(f"      Strategy: {config.frame_extraction.strategy.value}")
    print(f"{'='*60}")

    metadata = get_video_metadata(video_path)
    frames = extract_frames(video_path, output_dir, config.frame_extraction)

    elapsed = time.time() - t0
    print(f"Extracted {len(frames)} frames in {elapsed:.1f}s")

    return {
        "video_metadata": metadata,
        "frames_meta": frames,
        "num_frames": len(frames),
        "stage": "ingested",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "ingest": round(elapsed, 2)},
    }


def caption_node(state: ScreenLensState) -> dict:
    """Generate captions for all extracted frames."""
    t0 = time.time()
    config = ScreenLensConfig(**state["config"])
    output_dir = str(config.data_dir / "captions")

    backend = config.captioning.backend.value
    if backend in ("vllm", "omlx"):
        model_name = resolve_inference_model(config.captioning).split("/")[-1]
    else:
        model_name = config.captioning.ollama_model

    print(f"\n{'='*60}")
    print(f"[2/4] CAPTIONING {len(state['frames_meta'])} FRAMES")
    print(f"      Backend: {backend} ({model_name})")
    print(f"{'='*60}")

    captioned = caption_frames(
        state["frames_meta"],
        config.captioning,
        output_dir=output_dir,
    )

    elapsed = time.time() - t0
    print(f"Captioned {len(captioned)} frames in {elapsed:.1f}s")

    return {
        "captioned_frames": captioned,
        "stage": "captioned",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "caption": round(elapsed, 2)},
    }


def embed_node(state: ScreenLensState) -> dict:
    """Generate CLIP embeddings and store in vector DB."""
    t0 = time.time()
    config = ScreenLensConfig(**state["config"])

    print(f"\n{'='*60}")
    print(f"[3/4] EMBEDDING {len(state['captioned_frames'])} FRAMES")
    print(f"      Model: {config.embedding.model_name}")
    print(f"{'='*60}")

    embedder = CLIPEmbedder(config.embedding)

    image_paths = [f["path"] for f in state["captioned_frames"]]
    embeddings = embedder.embed_images(image_paths)

    print(f"\nStoring in ChromaDB ({config.vector_db.collection_name})...")
    store = ScreenLensVectorStore(config.vector_db)
    store.add_frames(state["captioned_frames"], embeddings)

    elapsed = time.time() - t0
    print(f"Embedded and stored {len(image_paths)} frames in {elapsed:.1f}s")

    return {
        "embeddings_shape": list(embeddings.shape),
        "stage": "embedded",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "embed": round(elapsed, 2)},
    }


def search_node(state: ScreenLensState) -> dict:
    """Search for frames matching the query."""
    t0 = time.time()
    config = ScreenLensConfig(**state["config"])
    query = state.get("query", "")

    if not query:
        return {"search_results": [], "stage": "search_skipped"}

    print(f"\n{'='*60}")
    print(f"[SEARCH] Query: '{query}'")
    print(f"{'='*60}")

    embedder = CLIPEmbedder(config.embedding)
    store = ScreenLensVectorStore(config.vector_db)

    query_emb = embedder.embed_text([query])[0]
    results = store.search_by_embedding(query_emb, top_k=config.search.top_k)

    elapsed = time.time() - t0
    print(f"Found {len(results)} results in {elapsed:.1f}s")

    for i, r in enumerate(results[:5]):
        print(f"  [{i+1}] t={r.get('timestamp_str', '?')} score={r.get('score', 0):.3f}")
        caption_preview = r.get("caption", "")[:100]
        print(f"      {caption_preview}...")

    return {
        "search_results": results,
        "stage": "searched",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "search": round(elapsed, 2)},
    }


def summarize_node(state: ScreenLensState) -> dict:
    """Generate a natural language summary from search results."""
    t0 = time.time()
    config = ScreenLensConfig(**state["config"])
    results = state.get("search_results", [])
    query = state.get("query", "")

    if not results:
        return {"summary": "No results to summarize.", "stage": "summarized"}

    print(f"\n{'='*60}")
    print(f"[SUMMARIZE] Generating answer for: '{query}'")
    print(f"{'='*60}")

    context_parts = []
    for r in results[:config.search.top_k]:
        ts = r.get("timestamp_str", "unknown")
        caption = r.get("caption", "No caption")
        score = r.get("score", 0)
        context_parts.append(f"[{ts}] (relevance: {score:.2f})\n{caption}")

    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are a video analysis assistant. Synthesize a direct answer to the user's "
        "question by drawing across MULTIPLE frame descriptions — do not echo or "
        "reformat a single frame verbatim. Identify what is consistent across frames, "
        "what changes over time, and which timestamps are most relevant. Reference "
        "specific timestamps for any concrete claim. Be concise and focused on the "
        "question asked: skip details that are not relevant. Output the answer directly "
        "with no preamble, planning notes, sign-off, or meta-commentary."
    )
    user = f"Question: {query}\n\nVideo frame descriptions:\n\n{context}"

    if config.captioning.backend.value == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=config.search.summarization_model,
            base_url=config.search.base_url,
            temperature=0.3,
        )
        summary = llm.invoke([("system", system), ("human", user)]).content
    else:
        summary = _inference_text_generate(
            InferenceClient(config.captioning),
            system,
            user,
            max_tokens=2048,
            temperature=0.3,
        )

    elapsed = time.time() - t0
    print(f"\nSummary generated in {elapsed:.1f}s")
    print(f"\n{summary}")

    return {
        "summary": summary,
        "stage": "summarized",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "summarize": round(elapsed, 2)},
    }


def _inference_text_generate(
    client: InferenceClient,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> str:
    """Generate text through the configured OpenAI-compatible server."""
    return client.chat(
        system_prompt,
        user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    )


def _compute_chunk_strategy(captioned: list[dict], model_context_tokens: int) -> dict:
    """Compute optimal chunking strategy based on caption stats and model context.

    Returns a dict with:
      chunk_size: number of captions per chunk
      num_chunks: total chunks
      estimated_tokens_per_chunk: estimated input tokens per Pass 1 call
      strategy: 'single_pass' if everything fits in one call, else 'hierarchical'
    """
    # Estimate tokens per caption (chars / 4 is a reasonable approximation)
    caption_lengths = [len(c.get("caption", "")) for c in captioned]
    avg_caption_chars = sum(caption_lengths) / max(len(caption_lengths), 1)
    avg_caption_tokens = avg_caption_chars / 4
    total_tokens = sum(l / 4 for l in caption_lengths)

    # Reserve tokens for: system prompt (~300), formatting overhead (~200), output (~2048)
    OVERHEAD_TOKENS = 2548
    usable_context = model_context_tokens - OVERHEAD_TOKENS

    # Safety margin: use only 80% of usable context to avoid edge-case truncation
    safe_context = int(usable_context * 0.8)

    # Can everything fit in a single pass?
    if total_tokens <= safe_context:
        return {
            "chunk_size": len(captioned),
            "num_chunks": 1,
            "estimated_tokens_per_chunk": int(total_tokens),
            "total_estimated_tokens": int(total_tokens),
            "strategy": "single_pass",
        }

    # Hierarchical: how many captions fit per chunk?
    # Each caption also has ~50 tokens of timestamp/formatting overhead
    tokens_per_caption = avg_caption_tokens + 50
    chunk_size = max(1, int(safe_context / tokens_per_caption))

    # Don't make tiny chunks — minimum 3 captions per chunk
    chunk_size = max(3, chunk_size)

    num_chunks = -(-len(captioned) // chunk_size)  # ceil division

    return {
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "estimated_tokens_per_chunk": int(chunk_size * tokens_per_caption),
        "total_estimated_tokens": int(total_tokens),
        "avg_caption_tokens": int(avg_caption_tokens),
        "model_context_tokens": model_context_tokens,
        "safe_context_tokens": safe_context,
        "strategy": "hierarchical",
    }


def summarize_all_node(state: ScreenLensState) -> dict:
    """Generate a full-video summary from ALL captions (not search-based).

    Uses the same configured model backend for text summarization.
    Dynamically computes chunk size based on model context window and caption stats.
      - If all captions fit in one call → single-pass summary
      - Otherwise → hierarchical: chunk summaries → final synthesis
    """
    import json as _json

    t0 = time.time()
    config = ScreenLensConfig(**state["config"])

    # Load captions: prefer from state, fall back to file on disk
    captioned = state.get("captioned_frames")
    if not captioned:
        captions_file = config.data_dir / "captions" / "all_captions.json"
        if captions_file.exists():
            with open(captions_file) as f:
                captioned = _json.load(f)
        else:
            return {"summary": "No captions found. Run ingestion first.", "stage": "summarized"}

    print(f"\n{'='*60}")
    print(f"[SUMMARIZE] Full-video summary from {len(captioned)} frames")
    print(f"{'='*60}")

    # Full summarization uses the selected direct provider.
    model = InferenceClient(config.captioning)
    model_context = resolve_inference_context(config.captioning)
    print(f"Using {model.backend.value} model: {model.model} at {model.base_url}")

    # ── Compute chunking strategy ────────────────────────────────────────
    strategy = _compute_chunk_strategy(captioned, model_context)

    print(f"\n  Model context: {model_context:,} tokens")
    print(f"  Total caption tokens: ~{strategy['total_estimated_tokens']:,}")
    print(f"  Strategy: {strategy['strategy']}")
    if strategy["strategy"] == "hierarchical":
        print(f"  Chunk size: {strategy['chunk_size']} captions/chunk "
              f"(~{strategy['estimated_tokens_per_chunk']:,} tokens/chunk)")
        print(f"  Total chunks: {strategy['num_chunks']}")
    print()

    CHUNK_SIZE = strategy["chunk_size"]

    # ── Single-pass: everything fits in one call ─────────────────────────
    if strategy["strategy"] == "single_pass":
        print("  Single-pass summarization (all captions fit in context)...")

        all_text = []
        for frame in captioned:
            ts = frame.get("timestamp_str", "?")
            caption = frame.get("caption", "")
            all_text.append(f"[{ts}]\n{caption}")
        captions_block = "\n\n---\n\n".join(all_text)

        video_meta = state.get("video_metadata", {})
        duration = video_meta.get("duration_seconds", "unknown")

        system = (
            "You are a video analysis assistant. You are given frame-by-frame descriptions "
            "of a screen recording. Produce a cohesive, well-structured summary of the entire "
            "recording. Include:\n"
            "1. **Overview** — What the recording is about in 1-2 sentences\n"
            "2. **Workflow** — The step-by-step process shown in the recording\n"
            "3. **Key Details** — Important specifics (tools used, settings, configurations)\n"
            "4. **Outcome** — What was accomplished by the end\n\n"
            "Write in clear, professional prose. Reference approximate timestamps where helpful."
        )
        user = (
            f"Video: {duration}s duration, {len(captioned)} keyframes analyzed.\n\n"
            f"Frame descriptions:\n\n{captions_block}"
        )

        summary = _inference_text_generate(model, system, user, max_tokens=4096)

        elapsed = time.time() - t0
        print(f"\nFull-video summary generated in {elapsed:.1f}s")
        print(f"\n{summary}")

        return {
            "summary": summary,
            "stage": "summarized",
            "elapsed_seconds": {**state.get("elapsed_seconds", {}), "summarize": round(elapsed, 2)},
        }

    # ── Hierarchical: chunk → summarize → synthesize ─────────────────────
    # Pass 1: Chunk summaries
    chunks = []
    for i in range(0, len(captioned), CHUNK_SIZE):
        chunk = captioned[i : i + CHUNK_SIZE]
        chunk_text = []
        for frame in chunk:
            ts = frame.get("timestamp_str", "?")
            caption = frame.get("caption", "")
            chunk_text.append(f"[{ts}]\n{caption}")
        chunks.append("\n\n---\n\n".join(chunk_text))

    print(f"  Pass 1: Summarizing {len(chunks)} chunks of ~{CHUNK_SIZE} frames each...")

    chunk_summaries = []
    for i, chunk_text in enumerate(chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE - 1, len(captioned) - 1)
        time_range = (
            f"{captioned[start_idx].get('timestamp_str', '?')} — "
            f"{captioned[end_idx].get('timestamp_str', '?')}"
        )

        system = (
            "You are a video analyst. Summarize what is happening in this segment of a "
            "screen recording. Focus on: what application is being used, what the user is "
            "doing, key content visible on screen, and the workflow being demonstrated. "
            "Be specific and factual. Keep your summary to 3-5 sentences."
        )
        user = f"Segment {i+1} ({time_range}):\n\n{chunk_text}"

        response = _inference_text_generate(model, system, user)
        chunk_summaries.append(f"**Segment {i+1} ({time_range}):** {response}")
        print(f"  Chunk {i+1}/{len(chunks)} summarized.")

    # Pass 2: Synthesize final summary
    print(f"\n  Pass 2: Synthesizing final summary...")

    all_chunk_summaries = "\n\n".join(chunk_summaries)
    video_meta = state.get("video_metadata", {})
    duration = video_meta.get("duration_seconds", "unknown")
    num_frames = len(captioned)

    system = (
        "You are a video analysis assistant. You are given segment-by-segment summaries "
        "of a screen recording. Produce a cohesive, well-structured summary of the entire "
        "recording. Include:\n"
        "1. **Overview** — What the recording is about in 1-2 sentences\n"
        "2. **Workflow** — The step-by-step process shown in the recording\n"
        "3. **Key Details** — Important specifics (tools used, settings, configurations)\n"
        "4. **Outcome** — What was accomplished by the end\n\n"
        "Write in clear, professional prose. Reference approximate timestamps where helpful."
    )
    user = (
        f"Video: {duration}s duration, {num_frames} keyframes analyzed.\n\n"
        f"Segment summaries:\n\n{all_chunk_summaries}"
    )

    summary = _inference_text_generate(model, system, user, max_tokens=4096)

    elapsed = time.time() - t0
    print(f"\nFull-video summary generated in {elapsed:.1f}s")
    print(f"\n{summary}")

    return {
        "summary": summary,
        "stage": "summarized",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "summarize": round(elapsed, 2)},
    }


# ── Graph Construction ──────────────────────────────────────────────────────

def build_ingest_graph():
    """Build the ingestion pipeline: extract → caption → embed."""
    graph = StateGraph(ScreenLensState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("caption", caption_node)
    graph.add_node("embed", embed_node)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "caption")
    graph.add_edge("caption", "embed")
    graph.add_edge("embed", END)
    return graph.compile()


def build_search_graph():
    """Build the search pipeline: search → summarize."""
    graph = StateGraph(ScreenLensState)
    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_edge(START, "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


def build_full_graph():
    """Build the complete pipeline: ingest → caption → embed → search → summarize."""
    graph = StateGraph(ScreenLensState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("caption", caption_node)
    graph.add_node("embed", embed_node)
    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "caption")
    graph.add_edge("caption", "embed")
    graph.add_edge("embed", "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()
