# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.

import json
from pathlib import Path
from typing import Dict, Tuple, Any, List


def meta_to_jsonable(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Python meta (with tuple edge keys) to a JSON-friendly dict."""
    nodes = meta["nodes"]
    edges_list: List[Dict[str, Any]] = []
    for (src, rel, dst), attrs in meta["edges"].items():
        entry = {"src": src, "rel": rel, "dst": dst}
        entry.update(attrs)
        edges_list.append(entry)
    return {"nodes": nodes, "edges": edges_list}


def jsonable_to_meta(js: Dict[str, Any]) -> Dict[str, Any]:
    """Convert back from JSON-friendly dict to Python meta with tuple edge keys."""
    nodes = js["nodes"]
    edges_dict: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for e in js["edges"]:
        src, rel, dst = e["src"], e["rel"], e["dst"]
        attrs = {k: v for k, v in e.items() if k not in ("src", "rel", "dst")}
        edges_dict[(src, rel, dst)] = attrs
    return {"nodes": nodes, "edges": edges_dict}


def save_meta(path: str | Path, meta: Dict[str, Any]) -> None:
    js = meta_to_jsonable(meta)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False, sort_keys=True)


def load_meta(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    return jsonable_to_meta(js)
