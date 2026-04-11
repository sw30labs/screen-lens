"""
Corpus Assembly Pipeline — LangGraph.

After ``reconstruct`` produces per-folder artifacts under ``data/*/output/``,
this pipeline detects whether the corpus represents a coherent coding project,
infers the original source-tree path of every artifact via batched LLM
sub-agents, validates the assembled tree (import resolution, orphans,
structure), and writes a single unified project tree to
``OUTPUT/<timestamp>/``.

Architecture (mirrors reconstruct.py):
  1. Discover         — walk data/*/output/, load meta + content snippets (no LLM)
  2. Gate             — is this a coding project?               (1 LLM call)
  3. Classify Corpus  — what are the project root directories?  (1 LLM call)
  4. Plan Paths       — partition into batches for sub-agents   (no LLM)
  5. Infer Workers    — sequential Send fan-out, batched inference (N LLM calls)
  6. Cluster          — group by inferred root, detect collisions (no LLM)
  7. QA Reflect       — mechanical (AST imports, orphans) + LLM judgment
  8. Materialize      — write OUTPUT/<timestamp>/ + MANIFEST + REPORT (no LLM)

The gate-fail and qa-fail paths exit gracefully without writing OUTPUT.
The pipeline reuses ``reconstruct.get_mlx_model``, ``reconstruct.mlx_generate``,
and ``reconstruct.parse_json_response``, so a single Python process running
``reconstruct --assemble`` loads the 122B model only once.
"""
from __future__ import annotations

import json
import logging
import operator
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from .config import ScreenLensConfig
from .reconstruct import get_mlx_model, mlx_generate, parse_json_response

logger = logging.getLogger("screenlens.assemble")


# ── Constants ────────────────────────────────────────────────────────────────

MAX_QA_ITERATIONS = 3
INFERENCE_BATCH_SIZE = 10
SNIPPET_CHARS = 400


# ── System Prompts ───────────────────────────────────────────────────────────

GATE_SYSTEM = (
    "You are evaluating whether a corpus of reconstructed screen-recording artifacts "
    "represents a coherent coding project that should be assembled into a single "
    "source tree.\n\n"
    "INPUT: a list of folder slugs + their reconstructed content types + brief "
    "descriptions.\n\n"
    "A 'coding project' means: the artifacts come from one or more related codebases — "
    "shared package roots, cross-references, project files (pyproject.toml, "
    "requirements.txt, .env), or naming conventions suggesting a directory tree. "
    "Even a corpus that mixes Python files with READMEs, YAML configs, and shell "
    "scripts qualifies if those files plausibly belong to the same project(s).\n\n"
    "It is NOT a coding project if the artifacts are: a single non-code document, a "
    "GUI walkthrough, unrelated PDFs, or a random grab-bag of files with no shared "
    "structure.\n\n"
    "Respond with ONLY valid JSON (no markdown fences):\n"
    '{"is_code_project": true/false, "reasoning": "brief explanation", '
    '"estimated_root_count": N}'
)

CLASSIFY_CORPUS_SYSTEM = (
    "You are detecting project root directories in a corpus of reconstructed "
    "artifacts. The downstream path-inference step will use your roots as the "
    "valid choices for where each file belongs.\n\n"
    "INPUT: folder slugs + brief descriptions.\n\n"
    "A 'root' is the top directory of a self-contained project (e.g. "
    "'my-package', 'my-package-dashboard' if separate). It is COMMON for a corpus "
    "to contain multiple roots — a backend package, a separate dashboard, a tests "
    "directory, etc. Use the empty string '' to denote 'project root' (files at the "
    "top level, like .env, pyproject.toml).\n\n"
    "Heuristics:\n"
    "- Slugs prefixed with the same package-like name often share a root\n"
    '- "standalone_*" slugs typically belong inside a "*-api/src/standalone_*/" tree\n'
    '- "src_*", "tests_*", "seed_*", "scripts_*" are intra-project paths, NOT roots\n'
    "- Different naming styles (e.g. snake_case vs kebab-case top dirs) often signal "
    "separate projects\n\n"
    "Respond with ONLY valid JSON (no markdown fences):\n"
    '{"roots": ["root1", "root2", ...], "confidence": 0.0-1.0, '
    '"reasoning": "brief explanation"}'
)


# ── State ────────────────────────────────────────────────────────────────────

class AssembleState(TypedDict, total=False):
    # Input
    data_dir: str
    output_dir: str
    config: dict
    mapping_override: Optional[str]
    dry_run: bool

    # Discovery
    artifacts: list[dict]   # {folder, src_rel, content_type, description, snippet, size, qa_scores}

    # Gate
    is_code_project: bool
    gate_reasoning: str
    estimated_root_count: int

    # Corpus classification
    project_roots: list[str]
    roots_confidence: float
    roots_reasoning: str

    # Path inference (filled by sub-agents — uses reducer)
    inference_batches: list[list[dict]]
    path_mappings: Annotated[list[dict], operator.add]

    # Cluster
    clusters: dict
    collisions: list[str]

    # QA
    qa_findings: dict
    qa_passed: bool
    qa_feedback: str
    qa_iteration: int

    # Output
    timestamp: str
    materialized_files: list[str]

    # Bookkeeping
    stage: str
    elapsed_seconds: dict


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_snippet(path: Path, max_chars: int = SNIPPET_CHARS) -> str:
    try:
        return path.read_text(errors="replace")[:max_chars]
    except Exception as e:
        return f"<read error: {e}>"


def _format_artifact_summary(artifacts: list[dict], max_per_line: int = 100) -> str:
    """Compact one-line-per-artifact summary for use in LLM prompts."""
    lines = []
    for a in artifacts:
        desc = (a.get("description") or "").strip().replace("\n", " ")[:max_per_line]
        ct = a.get("content_type", "?")
        lines.append(f"- {a['folder']}  [{ct}]  {desc}")
    return "\n".join(lines)


# ── Pipeline Nodes ───────────────────────────────────────────────────────────

def discover_node(state: AssembleState) -> dict:
    """Walk data/*/output/ and load all reconstructed artifacts + their meta."""
    t0 = time.time()
    data_dir = Path(state["data_dir"])

    print(f"\n{'='*60}")
    print(f"[1/8] DISCOVERING ARTIFACTS")
    print(f"      Scanning {data_dir}/")
    print(f"{'='*60}")

    if not data_dir.is_dir():
        print(f"  ERROR: {data_dir}/ not found")
        return {"artifacts": [], "stage": "error",
                "elapsed_seconds": {"discover": 0.0}}

    artifacts: list[dict] = []
    folders_seen = 0
    folders_with_meta = 0

    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        out = sub / "output"
        if not out.is_dir():
            continue
        folders_seen += 1

        meta_path = out / "reconstruction_meta.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                folders_with_meta += 1
            except Exception as e:
                logger.warning(f"Failed to parse {meta_path}: {e}")

        # Per-artifact descriptions live in meta["artifacts"], keyed by filename
        meta_artifacts = {a["filename"]: a for a in meta.get("artifacts", [])}

        for f in sorted(out.rglob("*")):
            if not f.is_file() or f.name == "reconstruction_meta.json":
                continue
            rel = str(f.relative_to(out))
            meta_entry = meta_artifacts.get(rel, {})
            artifacts.append({
                "folder": sub.name,
                "src_rel": rel,
                "src_abs": str(f),
                "content_type": meta.get("content_type", "unknown"),
                "classification_confidence": meta.get("classification_confidence", 0.0),
                "description": meta_entry.get("description", ""),
                "snippet": _read_snippet(f),
                "size": f.stat().st_size,
                "qa_scores": meta.get("qa_scores", {}),
            })

    elapsed = time.time() - t0
    print(f"\n  Folders scanned: {folders_seen}")
    print(f"  With meta.json:  {folders_with_meta}")
    print(f"  Total artifacts: {len(artifacts)}")
    print(f"  Discovered in {elapsed:.1f}s")

    return {
        "artifacts": artifacts,
        "stage": "discovered",
        "elapsed_seconds": {"discover": round(elapsed, 2)},
    }


def gate_node(state: AssembleState) -> dict:
    """Decide whether the corpus represents a coherent coding project."""
    t0 = time.time()
    artifacts = state["artifacts"]

    print(f"\n{'='*60}")
    print(f"[2/8] GATE — IS THIS A CODING PROJECT?")
    print(f"      Evaluating {len(artifacts)} artifact(s)")
    print(f"{'='*60}")

    if not artifacts:
        print("  No artifacts found — gate skipped")
        return {
            "is_code_project": False,
            "gate_reasoning": "No artifacts discovered",
            "estimated_root_count": 0,
            "stage": "gated",
            "elapsed_seconds": {"gate": 0.0},
        }

    config = ScreenLensConfig(**state["config"])
    model, tokenizer = get_mlx_model(config)

    # Aggregate content type distribution
    type_counts: dict[str, int] = {}
    for a in artifacts:
        type_counts[a["content_type"]] = type_counts.get(a["content_type"], 0) + 1
    type_summary = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))

    user_prompt = (
        f"Corpus of {len(artifacts)} reconstructed artifacts.\n"
        f"Content type distribution: {type_summary}\n\n"
        f"Artifacts (folder slug + content_type + description):\n\n"
        f"{_format_artifact_summary(artifacts)}\n\n"
        "Is this a coherent coding project that should be assembled into a unified "
        "source tree?"
    )

    response = mlx_generate(model, tokenizer, GATE_SYSTEM, user_prompt,
                            max_tokens=512, temperature=0.1)
    result = parse_json_response(response)

    is_code = bool(result.get("is_code_project", False))
    reasoning = result.get("reasoning", "No reasoning provided")
    root_hint = int(result.get("estimated_root_count", 1))

    elapsed = time.time() - t0
    print(f"\n  Decision:        {'YES — coding project' if is_code else 'NO'}")
    print(f"  Estimated roots: {root_hint}")
    print(f"  Reasoning:       {reasoning}")
    print(f"  Gated in {elapsed:.1f}s")

    return {
        "is_code_project": is_code,
        "gate_reasoning": reasoning,
        "estimated_root_count": root_hint,
        "stage": "gated",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "gate": round(elapsed, 2)},
    }


def classify_corpus_node(state: AssembleState) -> dict:
    """Detect project root directories from the corpus."""
    t0 = time.time()
    artifacts = state["artifacts"]
    root_hint = state.get("estimated_root_count", 1)

    print(f"\n{'='*60}")
    print(f"[3/8] CLASSIFY CORPUS — DETECT PROJECT ROOTS")
    print(f"      Hint from gate: ~{root_hint} root(s)")
    print(f"{'='*60}")

    config = ScreenLensConfig(**state["config"])
    model, tokenizer = get_mlx_model(config)

    user_prompt = (
        f"Corpus of {len(artifacts)} artifacts. The previous gate step estimated "
        f"~{root_hint} project root(s).\n\n"
        f"Artifacts (folder slug + content_type + description):\n\n"
        f"{_format_artifact_summary(artifacts)}\n\n"
        "Identify the top-level project root directories. Return them as a list — "
        "use '' for 'top of project'."
    )

    response = mlx_generate(model, tokenizer, CLASSIFY_CORPUS_SYSTEM, user_prompt,
                            max_tokens=512, temperature=0.1)
    result = parse_json_response(response)

    roots = result.get("roots", [])
    if not isinstance(roots, list):
        roots = []
    confidence = float(result.get("confidence", 0.0))
    reasoning = result.get("reasoning", "No reasoning provided")

    elapsed = time.time() - t0
    print(f"\n  Detected roots ({len(roots)}):")
    for r in roots:
        display = "(project top)" if r == "" else r
        print(f"    - {display}")
    print(f"  Confidence:  {confidence:.0%}")
    print(f"  Reasoning:   {reasoning}")
    print(f"  Classified in {elapsed:.1f}s")

    return {
        "project_roots": roots,
        "roots_confidence": confidence,
        "roots_reasoning": reasoning,
        "stage": "classified",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "classify": round(elapsed, 2)},
    }


# ── Stub nodes (filled in step 3+) ───────────────────────────────────────────

def plan_paths_node(state: AssembleState) -> dict:
    """STUB — implemented in step 3."""
    print(f"\n  [stub] plan_paths_node — not implemented yet")
    return {"inference_batches": [], "stage": "planned"}


def infer_paths_worker(state: dict) -> dict:
    """STUB — implemented in step 3."""
    return {"path_mappings": []}


def cluster_node(state: AssembleState) -> dict:
    """STUB — implemented in step 3."""
    return {"clusters": {}, "collisions": [], "stage": "clustered"}


def qa_reflect_node(state: AssembleState) -> dict:
    """STUB — implemented in step 4."""
    return {"qa_passed": True, "qa_findings": {}, "stage": "qa_passed"}


def materialize_node(state: AssembleState) -> dict:
    """STUB — implemented in step 5. In dry-run, never reached."""
    print(f"\n  [stub] materialize_node — not implemented yet")
    return {"materialized_files": [], "stage": "materialized"}


def end_with_explanation_node(state: AssembleState) -> dict:
    """Terminal node when the gate decides this isn't a coding project."""
    print(f"\n{'='*60}")
    print(f"GATE FAILED — pipeline halted")
    print(f"{'='*60}")
    print(f"  Reasoning: {state.get('gate_reasoning', '?')}")
    return {"stage": "gate_failed"}


# ── Routing ──────────────────────────────────────────────────────────────────

def route_after_gate(state: AssembleState) -> str:
    return "classify_corpus" if state.get("is_code_project") else "end_with_explanation"


def route_after_classify(state: AssembleState) -> str:
    """In dry-run, stop after classification. Otherwise continue to planning."""
    return "end_dry_run" if state.get("dry_run") else "plan_paths"


def end_dry_run_node(state: AssembleState) -> dict:
    """Terminal node for --dry-run mode after the corpus-classification phase."""
    print(f"\n{'='*60}")
    print(f"DRY RUN — stopping after corpus classification")
    print(f"{'='*60}")
    print(f"  Project: {'YES' if state.get('is_code_project') else 'NO'}")
    print(f"  Roots:   {state.get('project_roots', [])}")
    return {"stage": "dry_run_complete"}


# ── Graph Construction ───────────────────────────────────────────────────────

def build_assemble_graph():
    """Build the corpus assembly pipeline.

    Topology (step 2 — only the first three nodes do real work, rest are stubs):
        START → discover → gate → (classify_corpus | end_with_explanation)
                                       ↓
                           (end_dry_run | plan_paths → ... → materialize → END)
    """
    graph = StateGraph(AssembleState)

    graph.add_node("discover", discover_node)
    graph.add_node("gate", gate_node)
    graph.add_node("classify_corpus", classify_corpus_node)
    graph.add_node("end_with_explanation", end_with_explanation_node)
    graph.add_node("end_dry_run", end_dry_run_node)
    graph.add_node("plan_paths", plan_paths_node)
    graph.add_node("infer_paths_worker", infer_paths_worker)
    graph.add_node("cluster", cluster_node)
    graph.add_node("qa_reflect", qa_reflect_node)
    graph.add_node("materialize", materialize_node)

    graph.add_edge(START, "discover")
    graph.add_edge("discover", "gate")
    graph.add_conditional_edges(
        "gate", route_after_gate,
        {"classify_corpus": "classify_corpus", "end_with_explanation": "end_with_explanation"},
    )
    graph.add_conditional_edges(
        "classify_corpus", route_after_classify,
        {"end_dry_run": "end_dry_run", "plan_paths": "plan_paths"},
    )
    graph.add_edge("end_with_explanation", END)
    graph.add_edge("end_dry_run", END)

    # Stub edges — wired so the graph is structurally complete; nodes are no-ops
    graph.add_edge("plan_paths", "infer_paths_worker")
    graph.add_edge("infer_paths_worker", "cluster")
    graph.add_edge("cluster", "qa_reflect")
    graph.add_edge("qa_reflect", "materialize")
    graph.add_edge("materialize", END)

    return graph.compile()


# ── Public API ───────────────────────────────────────────────────────────────

def assemble_corpus(
    data_dir: str,
    output_dir: str,
    config: ScreenLensConfig,
    mapping_override: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Run the assembly pipeline against an existing data/ directory."""
    pipeline = build_assemble_graph()
    initial_state: AssembleState = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "config": config.model_dump(),
        "mapping_override": mapping_override,
        "dry_run": dry_run,
        "path_mappings": [],
        "qa_iteration": 0,
        "elapsed_seconds": {},
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    return pipeline.invoke(initial_state)
