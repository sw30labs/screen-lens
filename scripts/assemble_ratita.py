"""
Merge per-folder reconstruct outputs into a single unified file tree.

For every level-1 subfolder of ./data/ that has an ``output/`` directory,
copy the reconstructed file(s) into ./ratita/ at the destination path
specified in ``scripts/ratita_mapping.json``. The mapping was produced
by LLM subagents that infer the most likely original source-tree path
from the folder slug + the reconstructed file content (handling cases
where the slug encodes the path with underscores, has typos, uses
``dot.X`` for hidden files, or is a generic ``document.md``/``app.py``
that needs content-based disambiguation).

For any source folder NOT covered by the mapping, fall back to copying
the file at its in-output relative path (mechanical behaviour). Files
called ``reconstruction_meta.json`` are always skipped.

The destination is wiped on each run so it always reflects the current
state of ./data/ + the current mapping.

Usage:
    python scripts/assemble_ratita.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

DATA = Path("data")
DEST = Path("ratita")
MAPPING_FILE = Path("scripts/ratita_mapping.json")
SKIP_NAMES = {"reconstruction_meta.json"}


def load_mapping() -> dict[tuple[str, str], dict]:
    """Return {(folder, src_rel): mapping_entry} keyed by source identity."""
    if not MAPPING_FILE.exists():
        return {}
    entries = json.loads(MAPPING_FILE.read_text())
    return {(e["folder"], e["src_rel"]): e for e in entries}


def main() -> int:
    if not DATA.is_dir():
        print(f"error: {DATA}/ not found (run from project root)", file=sys.stderr)
        return 1

    mapping = load_mapping()
    if mapping:
        print(f"Loaded {len(mapping)} path mappings from {MAPPING_FILE}")
    else:
        print(f"No mapping at {MAPPING_FILE} — falling back to mechanical copy")

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    copied = 0
    sources_used = 0
    unmapped: list[str] = []
    collisions: list[str] = []
    used_dst: set[str] = set()

    for sub in sorted(DATA.iterdir()):
        if not sub.is_dir():
            continue
        out = sub / "output"
        if not out.is_dir():
            continue
        sources_used += 1

        for src in sorted(out.rglob("*")):
            if not src.is_file() or src.name in SKIP_NAMES:
                continue
            rel = str(src.relative_to(out))
            entry = mapping.get((sub.name, rel))
            if entry:
                dst_rel = entry["dst_rel"]
            else:
                dst_rel = rel
                unmapped.append(f"{sub.name}/{rel}")

            if dst_rel in used_dst:
                collisions.append(dst_rel)
            used_dst.add(dst_rel)

            dst = DEST / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    print(f"\nMerged {copied} file(s) from {sources_used} source folder(s) into {DEST}/")
    if unmapped:
        print(f"\n{len(unmapped)} file(s) not in mapping (used in-output rel path):")
        for u in unmapped[:10]:
            print(f"  {u}")
        if len(unmapped) > 10:
            print(f"  ... and {len(unmapped) - 10} more")
    if collisions:
        print(f"\n{len(collisions)} destination collision(s) (last writer won):")
        for c in collisions[:10]:
            print(f"  {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
