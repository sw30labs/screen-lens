"""
Artifact Reconstruction Pipeline — LangGraph Deep Agents.

Analyzes ingested video captions and reconstructs the original artifacts
(Python code, Markdown docs, PDFs, or GUI demo documentation) shown in recordings.

Architecture:
  1. Classify  — Determine content type from captions (code/doc/pdf/demo)
  2. Plan      — Generate tailored prompts + decompose into reconstruction tasks
  3. Execute   — Fan-out to parallel sub-agents via LangGraph Send (when safe)
                 OR sequential execution when ordering/coherence matters
  4. Reflect   — QA review with reflection agents (max 3 iterations)
  5. Save      — Write reconstructed artifacts to output folder

Uses LangGraph's Send API for conditional parallel dispatch and
Annotated reducers for collecting sub-agent outputs.
"""
import json
import logging
import operator
import re
import time
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from .config import ScreenLensConfig

logger = logging.getLogger("screenlens.reconstruct")


# ── Constants ────────────────────────────────────────────────────────────────

CONTENT_TYPES = {
    "python_code": "Python source code being written or edited in an IDE/editor",
    "markdown_document": "A Markdown or text document being authored or edited",
    "pdf_document": "A PDF document being viewed, reviewed, or presented",
    "gui_demo": "A GUI application walkthrough or demonstration",
}

MAX_QA_ITERATIONS = 3


# ── System Prompts ───────────────────────────────────────────────────────────

CLASSIFY_SYSTEM = (
    "You are a content classifier for screen recordings. Based on frame-by-frame "
    "captions from a screen recording, determine what type of content is primarily "
    "being shown.\n\n"
    "Categories:\n"
    "- python_code: Python code being written, edited, debugged, or reviewed in an IDE or editor\n"
    "- markdown_document: A Markdown, RST, or text document being authored or edited\n"
    "- pdf_document: A PDF or formatted document being viewed, reviewed, or discussed\n"
    "- gui_demo: A GUI application being demonstrated — navigating menus, clicking buttons, "
    "configuring settings\n\n"
    "Respond with ONLY a valid JSON object (no markdown fences):\n"
    '{"type": "<category>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}'
)

PLAN_PYTHON_SYSTEM = (
    "You are a reconstruction planner for screen recordings of Python coding sessions. "
    "Based on frame captions, identify ALL distinct Python files visible in the recording.\n\n"
    "For each file, provide:\n"
    "- filename: the file name as visible in the editor tab/title\n"
    "- description: what the file contains/does\n"
    "- key_content: notable imports, classes, functions visible\n\n"
    "Respond with ONLY valid JSON (no markdown fences):\n"
    '{\n'
    '  "files": [{"filename": "...", "description": "...", "key_content": "..."}],\n'
    '  "parallel_safe": true/false,\n'
    '  "reasoning": "why parallel is or isn\'t safe"\n'
    '}\n\n'
    "Set parallel_safe to true ONLY if the files are independent (no cross-imports "
    "between them). If you can only identify one file, just list that one."
)

RECONSTRUCT_PYTHON_SYSTEM = (
    "You are an expert Python developer reconstructing source code from a screen recording. "
    "You are given frame-by-frame descriptions showing Python code in an editor.\n\n"
    "CRITICAL RULES:\n"
    "1. Reconstruct the COMPLETE, FINAL version of the code as it appears at the end\n"
    "2. Include ALL imports, function/class definitions, constants, and logic\n"
    "3. If the recording shows iterative edits, produce the FINAL state only\n"
    "4. Reproduce the code EXACTLY — do not add, remove, or 'improve' anything\n"
    "5. Use proper Python formatting, indentation, and style as shown\n"
    "6. Output ONLY the raw Python code — no markdown fences, no explanations"
)

RECONSTRUCT_MARKDOWN_SYSTEM = (
    "You are a document reconstruction specialist. You are given frame-by-frame "
    "descriptions of a screen recording showing a Markdown document being written or edited.\n\n"
    "CRITICAL RULES:\n"
    "1. Reconstruct the COMPLETE, FINAL version of the document\n"
    "2. Preserve ALL headings, lists, code blocks, tables, links, and formatting\n"
    "3. Reproduce text EXACTLY as shown — do not paraphrase or summarize\n"
    "4. If the recording shows edits, produce the FINAL state only\n"
    "5. Output ONLY the raw Markdown content — no wrapping or explanations"
)

RECONSTRUCT_PDF_SYSTEM = (
    "You are a document reconstruction specialist. Based on frame descriptions of a PDF "
    "document being viewed, reconstruct the document's full content in Markdown format.\n\n"
    "CRITICAL RULES:\n"
    "1. Preserve the document's structure — sections, subsections, numbered items\n"
    "2. Reproduce ALL text content as accurately as possible\n"
    "3. Render tables as Markdown tables\n"
    "4. Describe figures, charts, or diagrams in [Figure: ...] blocks\n"
    "5. Include page/slide numbers if visible\n"
    "6. Output ONLY the reconstructed Markdown content"
)

RECONSTRUCT_DEMO_WALKTHROUGH_SYSTEM = (
    "You are a technical writer producing a step-by-step walkthrough from a screen "
    "recording of a GUI application demonstration.\n\n"
    "Produce a structured Markdown document with:\n"
    "1. **Application Overview** — What application, version, and platform\n"
    "2. **Prerequisites** — What's needed before starting\n"
    "3. **Step-by-Step Walkthrough** — Numbered steps with:\n"
    "   - What to click/interact with\n"
    "   - What appears on screen after each action\n"
    "   - Any values entered or options selected\n"
    "4. **Key Observations** — Important settings, configurations, or behaviors noted\n\n"
    "Be specific about UI elements (button names, menu paths, field labels). "
    "Reference approximate timestamps where helpful."
)

RECONSTRUCT_DEMO_REFERENCE_SYSTEM = (
    "You are a technical writer producing a reference guide from a screen recording "
    "of a GUI application demonstration.\n\n"
    "Produce a structured Markdown document covering:\n"
    "1. **Application Architecture** — Components, panels, and navigation structure\n"
    "2. **Features Demonstrated** — Each feature with description and location in UI\n"
    "3. **Configuration Options** — Settings, preferences, and their effects\n"
    "4. **Keyboard Shortcuts / Controls** — Any shortcuts or special controls shown\n\n"
    "Focus on factual, reference-style documentation. No narrative."
)

QA_REFLECT_SYSTEM = (
    "You are a quality assurance specialist reviewing reconstructed artifacts against "
    "the original screen recording frame captions.\n\n"
    "Evaluate the artifact on:\n"
    "1. **Completeness** (0-10): Does it capture ALL content shown in the recording?\n"
    "2. **Accuracy** (0-10): Is the content faithfully reproduced (not paraphrased/invented)?\n"
    "3. **Structure** (0-10): Is formatting, hierarchy, and organization correct?\n"
    "4. **Fidelity** (0-10): Would someone comparing this to the original find it faithful?\n\n"
    "Respond with ONLY valid JSON (no markdown fences):\n"
    "{\n"
    '  "passed": true/false,\n'
    '  "scores": {"completeness": N, "accuracy": N, "structure": N, "fidelity": N},\n'
    '  "overall": <average 0-10>,\n'
    '  "feedback": "specific issues to address if not passed",\n'
    '  "missing_elements": ["list of specific things missing from the artifact"]\n'
    "}\n\n"
    "Pass threshold: overall >= 7.0. Be rigorous but fair."
)


# ── State Definitions ────────────────────────────────────────────────────────

class ReconstructState(TypedDict, total=False):
    """Main graph state for the reconstruction pipeline."""
    # Input
    folder_path: str
    folder_name: str
    captions: list[dict]
    config: dict

    # Classification
    content_type: str
    classification_confidence: float
    classification_reasoning: str

    # Reconstruction plan
    system_prompt: str
    reconstruction_tasks: list[dict]
    parallel_safe: bool

    # Sub-agent output — uses add reducer for parallel fan-out collection
    artifacts: Annotated[list[dict], operator.add]

    # QA reflection
    qa_feedback: str
    qa_passed: bool
    qa_iteration: int
    qa_scores: dict

    # Output
    saved_paths: list[str]
    stage: str
    error: str
    elapsed_seconds: dict


# ── Model Cache ──────────────────────────────────────────────────────────────

_MODEL_CACHE: dict = {}


def _get_mlx_model(config: ScreenLensConfig):
    """Load MLX model once and cache for reuse across all nodes."""
    key = config.captioning.mlx_repo_id
    if key not in _MODEL_CACHE:
        from mlx_vlm import load

        model_id = config.captioning.mlx_model_path or config.captioning.mlx_repo_id
        print(f"Loading MLX model: {model_id}")
        loaded = load(model_id, lazy=True)
        _MODEL_CACHE[key] = loaded[:2]
        print("Model loaded.")
    return _MODEL_CACHE[key]


def _mlx_generate(model, tokenizer, system: str, user: str,
                   max_tokens: int = 4096, temperature: float = 0.2) -> str:
    """Generate text using the MLX model (text-only, no image)."""
    from mlx_vlm import generate

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    result = generate(model, tokenizer, prompt, max_tokens=max_tokens, temperature=temperature)

    if isinstance(result, str):
        raw = result
    elif hasattr(result, "text"):
        raw = result.text
    else:
        raw = str(result)

    return re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and extra text."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}...")
    return {}


def _build_caption_block(captions: list[dict], max_chars: int = 80000) -> str:
    """Build a formatted caption block for LLM consumption."""
    parts = []
    total_chars = 0
    for c in captions:
        ts = c.get("timestamp_str", "?")
        text = c.get("caption", "")
        entry = f"[{ts}]\n{text}"
        if total_chars + len(entry) > max_chars:
            parts.append("[... additional frames truncated for context limit ...]")
            break
        parts.append(entry)
        total_chars += len(entry)
    return "\n\n---\n\n".join(parts)


# ── Pipeline Nodes ───────────────────────────────────────────────────────────

def classify_node(state: ReconstructState) -> dict:
    """Classify the content type from captions."""
    t0 = time.time()
    config = ScreenLensConfig(**state["config"])
    model, tokenizer = _get_mlx_model(config)
    captions = state["captions"]

    print(f"\n{'='*60}")
    print(f"[1/5] CLASSIFYING CONTENT — {state['folder_name']}")
    print(f"      Analyzing {len(captions)} frame captions")
    print(f"{'='*60}")

    # Build a summary of captions for classification (cap at 50 frames)
    caption_texts = []
    for c in captions[:50]:
        ts = c.get("timestamp_str", "?")
        text = c.get("caption", "")[:500]
        caption_texts.append(f"[{ts}] {text}")

    user_prompt = (
        f"Screen recording: {len(captions)} frames total.\n\n"
        f"Frame captions (first {min(len(captions), 50)}):\n\n"
        + "\n\n---\n\n".join(caption_texts)
    )

    response = _mlx_generate(model, tokenizer, CLASSIFY_SYSTEM, user_prompt,
                              max_tokens=512, temperature=0.1)
    result = _parse_json_response(response)

    content_type = result.get("type", "gui_demo")
    if content_type not in CONTENT_TYPES:
        content_type = "gui_demo"

    confidence = result.get("confidence", 0.5)
    reasoning = result.get("reasoning", "No reasoning provided")

    elapsed = time.time() - t0
    print(f"\n  Content type: {content_type} (confidence: {confidence:.0%})")
    print(f"  Reasoning: {reasoning}")
    print(f"  Classified in {elapsed:.1f}s")

    return {
        "content_type": content_type,
        "classification_confidence": confidence,
        "classification_reasoning": reasoning,
        "stage": "classified",
        "elapsed_seconds": {"classify": round(elapsed, 2)},
    }


def plan_node(state: ReconstructState) -> dict:
    """Generate reconstruction plan: system prompt, task list, parallelism decision."""
    t0 = time.time()
    config = ScreenLensConfig(**state["config"])
    model, tokenizer = _get_mlx_model(config)
    content_type = state["content_type"]
    captions = state["captions"]
    qa_feedback = state.get("qa_feedback", "")
    qa_iteration = state.get("qa_iteration", 0)

    print(f"\n{'='*60}")
    print(f"[2/5] PLANNING RECONSTRUCTION — {content_type}")
    if qa_iteration > 0:
        print(f"      Retry #{qa_iteration} — incorporating QA feedback")
    print(f"{'='*60}")

    caption_block = _build_caption_block(captions)

    tasks = []
    parallel_safe = False
    system_prompt = ""

    if content_type == "python_code":
        system_prompt = RECONSTRUCT_PYTHON_SYSTEM

        # Ask LLM to identify distinct files from captions
        file_id_prompt = f"Frame captions from a Python coding session:\n\n{caption_block}"
        response = _mlx_generate(model, tokenizer, PLAN_PYTHON_SYSTEM,
                                  file_id_prompt, max_tokens=1024, temperature=0.1)
        plan = _parse_json_response(response)

        files = plan.get("files", [{"filename": "reconstructed.py",
                                     "description": "Main script"}])
        parallel_safe = plan.get("parallel_safe", False) and len(files) > 1

        for f in files:
            task_prompt = (
                f"Reconstruct the file '{f['filename']}' ({f.get('description', '')}).\n\n"
            )
            if qa_feedback and qa_iteration > 0:
                task_prompt += (
                    f"PREVIOUS QA FEEDBACK — address these issues:\n{qa_feedback}\n\n"
                )
            task_prompt += f"Frame captions:\n\n{caption_block}"

            tasks.append({
                "filename": f["filename"],
                "description": f.get("description", ""),
                "prompt": task_prompt,
                "output_type": "python",
            })

        print(f"  Identified {len(files)} Python file(s):")
        for f in files:
            print(f"    - {f['filename']}: {f.get('description', '')}")

    elif content_type == "markdown_document":
        system_prompt = RECONSTRUCT_MARKDOWN_SYSTEM
        parallel_safe = False

        task_prompt = "Reconstruct the complete Markdown document.\n\n"
        if qa_feedback and qa_iteration > 0:
            task_prompt += f"PREVIOUS QA FEEDBACK — address these issues:\n{qa_feedback}\n\n"
        task_prompt += f"Frame captions:\n\n{caption_block}"

        tasks.append({
            "filename": "document.md",
            "description": "Reconstructed Markdown document",
            "prompt": task_prompt,
            "output_type": "markdown",
        })

    elif content_type == "pdf_document":
        system_prompt = RECONSTRUCT_PDF_SYSTEM
        parallel_safe = False

        task_prompt = "Reconstruct the PDF document content in Markdown format.\n\n"
        if qa_feedback and qa_iteration > 0:
            task_prompt += f"PREVIOUS QA FEEDBACK — address these issues:\n{qa_feedback}\n\n"
        task_prompt += f"Frame captions:\n\n{caption_block}"

        tasks.append({
            "filename": "document.md",
            "description": "Reconstructed PDF content",
            "prompt": task_prompt,
            "output_type": "markdown",
        })

    elif content_type == "gui_demo":
        # GUI demos produce independent documents → parallel-safe
        parallel_safe = True

        base_context = ""
        if qa_feedback and qa_iteration > 0:
            base_context = f"PREVIOUS QA FEEDBACK — address these issues:\n{qa_feedback}\n\n"
        base_context += f"Frame captions:\n\n{caption_block}"

        tasks.append({
            "filename": "walkthrough.md",
            "description": "Step-by-step walkthrough",
            "prompt": f"Generate a detailed step-by-step walkthrough.\n\n{base_context}",
            "output_type": "markdown",
            "system_override": RECONSTRUCT_DEMO_WALKTHROUGH_SYSTEM,
        })
        tasks.append({
            "filename": "reference.md",
            "description": "Technical reference guide",
            "prompt": f"Generate a technical reference guide.\n\n{base_context}",
            "output_type": "markdown",
            "system_override": RECONSTRUCT_DEMO_REFERENCE_SYSTEM,
        })

    print(f"  Tasks: {len(tasks)} | Parallel dispatch: {parallel_safe}")
    elapsed = time.time() - t0
    print(f"  Planned in {elapsed:.1f}s")

    return {
        "system_prompt": system_prompt,
        "reconstruction_tasks": tasks,
        "parallel_safe": parallel_safe,
        "stage": "planned",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "plan": round(elapsed, 2)},
    }


def reconstruct_worker(state: dict) -> dict:
    """Execute a single reconstruction task. Invoked via LangGraph Send for parallel fan-out."""
    task = state["task"]
    config = ScreenLensConfig(**state["config"])
    model, tokenizer = _get_mlx_model(config)

    system = task.get("system_override", state.get("system_prompt", ""))
    user = task["prompt"]

    print(f"    [sub-agent] Reconstructing: {task['filename']}")
    t0 = time.time()

    content = _mlx_generate(model, tokenizer, system, user,
                             max_tokens=8192, temperature=0.1)

    elapsed = time.time() - t0
    print(f"    [sub-agent] {task['filename']} done ({len(content)} chars, {elapsed:.1f}s)")

    return {
        "artifacts": [{
            "filename": task["filename"],
            "content": content,
            "type": task["output_type"],
            "description": task.get("description", ""),
            "iteration": state.get("qa_iteration", 0),
        }],
    }


def reconstruct_sequential(state: ReconstructState) -> dict:
    """Process all reconstruction tasks sequentially when parallel dispatch isn't safe."""
    config = ScreenLensConfig(**state["config"])
    model, tokenizer = _get_mlx_model(config)
    tasks = state["reconstruction_tasks"]
    system_prompt = state.get("system_prompt", "")
    qa_iteration = state.get("qa_iteration", 0)

    print(f"\n  Processing {len(tasks)} task(s) sequentially...")

    new_artifacts = []
    for i, task in enumerate(tasks, 1):
        system = task.get("system_override", system_prompt)
        user = task["prompt"]

        print(f"    [{i}/{len(tasks)}] Reconstructing: {task['filename']}")
        t0 = time.time()

        content = _mlx_generate(model, tokenizer, system, user,
                                 max_tokens=8192, temperature=0.1)

        elapsed = time.time() - t0
        print(f"    [{i}/{len(tasks)}] {task['filename']} done "
              f"({len(content)} chars, {elapsed:.1f}s)")

        new_artifacts.append({
            "filename": task["filename"],
            "content": content,
            "type": task["output_type"],
            "description": task.get("description", ""),
            "iteration": qa_iteration,
        })

    return {"artifacts": new_artifacts}


def qa_reflect_node(state: ReconstructState) -> dict:
    """Quality-check artifacts using a reflection agent. Routes to retry or save."""
    t0 = time.time()
    config = ScreenLensConfig(**state["config"])
    model, tokenizer = _get_mlx_model(config)
    qa_iteration = state.get("qa_iteration", 0)
    captions = state["captions"]

    # Get only artifacts from current iteration
    all_artifacts = state.get("artifacts", [])
    current_artifacts = [a for a in all_artifacts if a.get("iteration", 0) == qa_iteration]
    if not current_artifacts:
        # Fallback: take the most recent N artifacts
        n_tasks = len(state.get("reconstruction_tasks", []))
        current_artifacts = all_artifacts[-n_tasks:] if n_tasks else all_artifacts

    print(f"\n{'='*60}")
    print(f"[4/5] QA REFLECTION — iteration {qa_iteration + 1}/{MAX_QA_ITERATIONS}")
    print(f"      Reviewing {len(current_artifacts)} artifact(s)")
    print(f"{'='*60}")

    # Build QA context — abbreviated captions + full artifacts
    caption_summary = _build_caption_block(captions[:30], max_chars=30000)

    artifacts_text = ""
    for a in current_artifacts:
        artifacts_text += f"\n\n--- {a['filename']} ({a['type']}) ---\n"
        artifacts_text += a["content"][:4000]
        if len(a["content"]) > 4000:
            artifacts_text += f"\n[... truncated, {len(a['content'])} total chars ...]"
        artifacts_text += "\n"

    user_prompt = (
        f"Content type: {state.get('content_type', 'unknown')}\n\n"
        f"ORIGINAL FRAME CAPTIONS (reference):\n{caption_summary}\n\n"
        f"RECONSTRUCTED ARTIFACTS:\n{artifacts_text}"
    )

    response = _mlx_generate(model, tokenizer, QA_REFLECT_SYSTEM, user_prompt,
                              max_tokens=1024, temperature=0.1)
    result = _parse_json_response(response)

    passed = result.get("passed", True)
    overall = result.get("overall", 7.0)
    feedback = result.get("feedback", "")
    scores = result.get("scores", {})
    missing = result.get("missing_elements", [])

    # Force pass after max iterations
    if qa_iteration >= MAX_QA_ITERATIONS - 1 and not passed:
        passed = True
        feedback += " [Max iterations reached — accepting current output]"
        print(f"  Max QA iterations reached — accepting output.")

    elapsed = time.time() - t0
    print(f"\n  QA Score: {overall}/10")
    if scores:
        print(f"  Breakdown: {json.dumps(scores)}")
    print(f"  Passed: {'YES' if passed else 'NO'}")
    if not passed:
        print(f"  Feedback: {feedback}")
        if missing:
            print(f"  Missing: {', '.join(missing[:5])}")
    print(f"  Reflection completed in {elapsed:.1f}s")

    return {
        "qa_passed": passed,
        "qa_feedback": feedback,
        "qa_scores": scores,
        "qa_iteration": qa_iteration + 1 if not passed else qa_iteration,
        "stage": "qa_passed" if passed else "qa_retry",
        "elapsed_seconds": {
            **state.get("elapsed_seconds", {}),
            f"qa_{qa_iteration}": round(elapsed, 2),
        },
    }


def save_node(state: ReconstructState) -> dict:
    """Save reconstructed artifacts to the output folder."""
    t0 = time.time()
    folder_path = Path(state["folder_path"])
    output_dir = folder_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_iteration = state.get("qa_iteration", 0)

    # Get latest-iteration artifacts
    all_artifacts = state.get("artifacts", [])
    latest = [a for a in all_artifacts if a.get("iteration", 0) == qa_iteration]
    if not latest:
        n_tasks = len(state.get("reconstruction_tasks", []))
        latest = all_artifacts[-n_tasks:] if n_tasks else all_artifacts

    print(f"\n{'='*60}")
    print(f"[5/5] SAVING ARTIFACTS — {len(latest)} file(s)")
    print(f"      Output: {output_dir}")
    print(f"{'='*60}")

    saved = []
    for artifact in latest:
        filepath = output_dir / artifact["filename"]
        filepath.write_text(artifact["content"])
        saved.append(str(filepath))
        print(f"  saved {artifact['filename']} ({len(artifact['content']):,} chars)")

    # Save reconstruction metadata
    meta = {
        "content_type": state.get("content_type"),
        "classification_confidence": state.get("classification_confidence"),
        "classification_reasoning": state.get("classification_reasoning"),
        "qa_scores": state.get("qa_scores", {}),
        "qa_iterations_used": state.get("qa_iteration", 0) + 1,
        "max_qa_iterations": MAX_QA_ITERATIONS,
        "artifacts": [
            {
                "filename": a["filename"],
                "type": a["type"],
                "description": a.get("description", ""),
                "size_chars": len(a["content"]),
            }
            for a in latest
        ],
    }
    meta_path = output_dir / "reconstruction_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    saved.append(str(meta_path))

    elapsed = time.time() - t0
    print(f"\n  Saved {len(saved)} files in {elapsed:.1f}s")

    return {
        "saved_paths": saved,
        "stage": "saved",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "save": round(elapsed, 2)},
    }


# ── Routing Functions ────────────────────────────────────────────────────────

def route_to_workers(state: ReconstructState):
    """Dispatch reconstruction tasks — parallel via Send when safe, else sequential."""
    tasks = state.get("reconstruction_tasks", [])
    parallel_safe = state.get("parallel_safe", False)

    if parallel_safe and len(tasks) > 1:
        print(f"\n  Dispatching {len(tasks)} parallel sub-agents")
        return [
            Send("reconstruct_worker", {
                "task": task,
                "config": state["config"],
                "system_prompt": state.get("system_prompt", ""),
                "qa_iteration": state.get("qa_iteration", 0),
            })
            for task in tasks
        ]
    else:
        print(f"\n  Sequential execution ({len(tasks)} task(s))")
        return "reconstruct_sequential"


def should_retry_or_save(state: ReconstructState) -> str:
    """After QA: retry reconstruction or proceed to save."""
    if state.get("qa_passed", False):
        return "save"
    return "plan"


# ── Graph Construction ───────────────────────────────────────────────────────

def build_reconstruct_graph():
    """Build the reconstruction pipeline with parallel sub-agents and reflection loop.

    Graph topology:
        START → classify → plan →[dispatch]→ worker(s)    → qa_reflect → save → END
                                  ↘ sequential ↗            ↓      ↑
                                                            plan ←─╯ (retry)
    """
    graph = StateGraph(ReconstructState)

    # Nodes
    graph.add_node("classify", classify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("reconstruct_worker", reconstruct_worker)
    graph.add_node("reconstruct_sequential", reconstruct_sequential)
    graph.add_node("qa_reflect", qa_reflect_node)
    graph.add_node("save", save_node)

    # Edges
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "plan")
    graph.add_conditional_edges(
        "plan", route_to_workers,
        ["reconstruct_worker", "reconstruct_sequential"],
    )
    graph.add_edge("reconstruct_worker", "qa_reflect")
    graph.add_edge("reconstruct_sequential", "qa_reflect")
    graph.add_conditional_edges(
        "qa_reflect", should_retry_or_save,
        {"plan": "plan", "save": "save"},
    )
    graph.add_edge("save", END)

    return graph.compile()


# ── Public API ───────────────────────────────────────────────────────────────

def reconstruct_folder(folder_path: str, config: ScreenLensConfig) -> dict:
    """Run the full reconstruction pipeline on a single data folder.

    Args:
        folder_path: Path to a data/<video_name> folder containing captions/
        config: ScreenLensConfig instance

    Returns:
        Pipeline result dict with saved_paths, qa_scores, content_type, etc.
    """
    folder = Path(folder_path)
    captions_file = folder / "captions" / "all_captions.json"

    if not captions_file.exists():
        return {"error": f"No captions found at {captions_file}", "stage": "error"}

    with open(captions_file) as f:
        captions = json.load(f)

    if not captions:
        return {"error": "Captions file is empty", "stage": "error"}

    pipeline = build_reconstruct_graph()
    initial_state = {
        "folder_path": str(folder),
        "folder_name": folder.name,
        "captions": captions,
        "config": config.model_dump(),
        "artifacts": [],
        "qa_iteration": 0,
    }

    return pipeline.invoke(initial_state)
