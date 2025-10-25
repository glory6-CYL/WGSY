#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_multimodal_jsonl.py

Read medgemma annotations (train_ws_features.json / test_ws_features.json) and
produce multimodal JSONL files for end-to-end LoRA training.

Output files:
  <out_dir>/multimodal_train.jsonl
  <out_dir>/multimodal_test.jsonl
  <out_dir>/multimodal_all.jsonl  (optional combined)

"""

import os
import re
import json
import argparse
from pathlib import Path

# -------------------- utility: robust extract JSON from SDK-like text --------------------
def extract_json_from_raw_response(raw_text: str):
    """Robustly try to extract embedded JSON object from SDK/text wrapper.
       Returns parsed dict or None.
    """
    if not raw_text:
        return None
    import json, re
    s = str(raw_text)
    # try direct json
    try:
        return json.loads(s)
    except Exception:
        pass
    # find likely JSON start
    m = re.search(r'\{\s*"(?:image_id|features|feature_explanations|notes|ws)"', s)
    if not m:
        m = re.search(r'\{\s*"', s)
    if not m:
        pos0 = s.find('{')
        if pos0 == -1:
            return None
        start_idx = pos0
    else:
        start_idx = m.start()

    # scan to matching closing brace
    i = start_idx
    n = len(s)
    stack = []
    in_str = False
    esc = False
    end_idx = None
    while i < n:
        ch = s[i]
        if ch == '"' and not esc:
            in_str = not in_str
        if not in_str:
            if ch == '{':
                stack.append('{')
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        end_idx = i + 1
                        break
        if ch == '\\' and not esc:
            esc = True
        else:
            esc = False
        i += 1

    if end_idx is None:
        return None

    candidate = s[start_idx:end_idx]
    # try loads directly
    try:
        return json.loads(candidate)
    except Exception:
        pass
    # try unescape
    try:
        unescaped = bytes(candidate, "utf-8").decode("unicode_escape")
        return json.loads(unescaped)
    except Exception:
        pass
    # try simple replacements
    try:
        tmp = candidate.replace('\\"', '"').replace('\\\\n', '\\n').replace('\\\\t', '\\t')
        tmp = tmp.strip()
        if tmp.startswith("'") and tmp.endswith("'"):
            tmp = tmp[1:-1]
        return json.loads(tmp)
    except Exception:
        return None

# -------------------- resolve image path relative to root --------------------
def resolve_image_path(root: str, rel_path: str):
    """Return absolute path if exists, else None. Try a few heuristics."""
    if not rel_path:
        return None
    # if already absolute
    if os.path.isabs(rel_path):
        if os.path.exists(rel_path):
            return rel_path
        # try strip leading slash and join root
        stripped = rel_path.lstrip(os.sep)
        alt = os.path.join(root, stripped)
        if os.path.exists(alt):
            return alt
        return None
    # relative: join root
    p = os.path.join(root, rel_path)
    if os.path.exists(p):
        return os.path.abspath(p)
    # if rel_path starts with 'dataset/...' and root already contains 'dataset', try removing duplicate
    root_base = os.path.basename(os.path.normpath(root))
    if rel_path.startswith(root_base + os.sep):
        p2 = os.path.join(root, rel_path[len(root_base)+1:])
        if os.path.exists(p2):
            return os.path.abspath(p2)
    # try basename only
    p3 = os.path.join(root, os.path.basename(rel_path))
    if os.path.exists(p3):
        return os.path.abspath(p3)
    # progressive strip
    parts = rel_path.split(os.sep)
    for i in range(1, len(parts)):
        cand = os.path.join(root, *parts[i:])
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None

# -------------------- build instruction/response --------------------
def build_feature_summary(parsed):
    # Build concise human-readable summary string from parsed['features']
    feats = parsed.get("features", {}) if parsed else {}
    ordered = ["forehead","periorbital_fullness","epicanthic_fold","intercanthal_distance",
               "eye_shape","nose_shape","philtrum_length","lip_fullness","mouth_width",
               "chin_size","dental_anomalies","iris_pattern"]
    parts = []
    for k in ordered:
        v = feats.get(k, "uncertain")
        parts.append(f"{k.replace('_',' ')}: {v}")
    return "; ".join(parts)

def build_instruction(include_features: bool, feature_summary: str):
    if include_features and feature_summary:
        inst = (
            "You will be given an aligned face image and a short feature summary extracted from it. "
            "Based only on the image and/or the feature summary, return EXACTLY one JSON object (no extra text) with fields: "
            "\"ws\" (\"yes\" or \"no\"), \"explanation\" (one-sentence English justification referencing features), "
            "and \"features\" (the provided mapping). The features field should be a JSON mapping of feature keys to labels."
            "\n\nFeature summary:\n" + feature_summary
        )
    else:
        inst = (
            "You will be given an aligned face image. Return EXACTLY one JSON object (no extra text) with fields: "
            "\"ws\" (\"yes\" or \"no\"), \"explanation\" (one-sentence English justification referencing observable features), "
            "and \"features\" (a JSON mapping of the observed feature labels)."
        )
    return inst

def build_response_object(parsed, label):
    # parsed may be None or incomplete
    feats = parsed.get("features", {}) if parsed else {}
    expls = parsed.get("feature_explanations", {}) if parsed else {}
    # explanation choose a representative field if present
    explanation = ""
    for k in ("nose_shape","philtrum_length","lip_fullness","forehead","chin_size"):
        if expls.get(k):
            explanation = expls[k]
            break
    if not explanation:
        explanation = "Assessment based on the provided facial features."
    resp = {
        "ws": "yes" if int(label) == 1 else "no",
        "explanation": explanation,
        "features": feats,
        "notes": parsed.get("notes","") if parsed else ""
    }
    return resp

# -------------------- main conversion --------------------
def process_file(root, ann_path, out_jsonl_path, include_features=True):
    ann_path = Path(ann_path)
    if not ann_path.exists():
        print(f"[WARN] annotation file not found: {ann_path}")
        return 0, 0, 0
    data = json.load(open(ann_path, encoding='utf-8'))
    os.makedirs(Path(out_jsonl_path).parent, exist_ok=True)
    written = 0
    missing_images = 0
    needs_parsed = 0
    with open(out_jsonl_path, 'w', encoding='utf-8') as fout:
        for rec in data:
            rel = rec.get("image")
            label = rec.get("label", None)
            # resolve image
            img_path = resolve_image_path(root, rel)
            if not img_path:
                missing_images += 1
                # still write entry but mark image as original rel (optional: skip)
                img_path = os.path.join(root, rel)  # keep best-effort
            # get parsed_annotation
            parsed = rec.get("parsed_annotation")
            if not parsed:
                # try extract from model_raw_text or model_raw_response
                raw = rec.get("model_raw_text") or rec.get("model_raw_response") or ""
                parsed = extract_json_from_raw_response(raw)
                if parsed:
                    # if extracted but 'features' nested under a string, ensure proper type
                    pass
            if not parsed:
                needs_parsed += 1
                parsed = {"features": {}, "feature_explanations": {}, "notes": "needs_review"}
            # build summary/instruction/response
            feat_summary = build_feature_summary(parsed) if include_features else ""
            instruction = build_instruction(include_features, feat_summary)
            response_obj = build_response_object(parsed, label)
            # write jsonl: image (abs path), instruction (text), response (JSON string), label
            out_rec = {
                "image": os.path.abspath(img_path),
                "instruction": instruction,
                "response": json.dumps(response_obj, ensure_ascii=False),
                "label": label
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            written += 1
    return written, missing_images, needs_parsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="dataset root folder (parent of dataset/...)")
    parser.add_argument("--ann-dir", default="dataset/annotations", help="annotations folder relative to root (or absolute)")
    parser.add_argument("--train-file", default="train_ws_features.json", help="train annotation file name")
    parser.add_argument("--test-file", default="test_ws_features.json", help="test annotation file name")
    parser.add_argument("--out-dir", default="multimodal_data", help="output folder for jsonl files")
    parser.add_argument("--which", choices=["train","test","both"], default="both", help="which splits to convert")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--include-features", dest="include_features", action="store_true", help="include feature summary in instruction (default)")
    group.add_argument("--no-features", dest="include_features", action="store_false", help="do NOT include feature summary in instruction")
    parser.set_defaults(include_features=True)
    args = parser.parse_args()

    root = args.root
    ann_dir = args.ann_dir if os.path.isabs(args.ann_dir) else os.path.join(root, args.ann_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    if args.which in ("train","both"):
        train_ann = os.path.join(ann_dir, args.train_file)
        train_out = os.path.join(args.out_dir, "multimodal_train.jsonl")
        w, miss, needp = process_file(root, train_ann, train_out, include_features=args.include_features)
        results["train"] = (w, miss, needp)
    if args.which in ("test","both"):
        test_ann = os.path.join(ann_dir, args.test_file)
        test_out = os.path.join(args.out_dir, "multimodal_test.jsonl")
        w, miss, needp = process_file(root, test_ann, test_out, include_features=args.include_features)
        results["test"] = (w, miss, needp)

    # optionally create combined all jsonl
    combined_out = os.path.join(args.out_dir, "multimodal_all.jsonl")
    with open(combined_out, "w", encoding='utf-8') as fout_comb:
        for split in ("train","test"):
            if split in results:
                path = os.path.join(args.out_dir, f"multimodal_{split}.jsonl")
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        for ln in f:
                            fout_comb.write(ln)
    print("Conversion complete. Summary (written, missing_images, missing_parsed_annotation):")
    for k,v in results.items():
        print(f"  {k}: written={v[0]}, missing_images={v[1]}, parsed_missing={v[2]}")
    print("Files written to:", args.out_dir)

if __name__ == "__main__":
    main()
