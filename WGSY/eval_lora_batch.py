#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluation for MedGemma + LoRA fine-tuned checkpoints (multi-disease).
基于 eval_lora.py 改进：支持 batchsize=128 批量推理，提升GPU利用率
"""

import os
import json
import re
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- 固定配置（新增BATCH_SIZE） -----------------
BASE_MODEL_PATH = "your_model/medgemma-base"  # 替换为实际基础模型路径
LORA_ROOTS = [
    "your_lora/medgemma_lora_diseases"  # 替换为实际LoRA模型根目录
]
TEST_JSONS = {
    "your_test": "your_test.jsonl"  # 替换为实际测试集JSON路径
}
OUTPUT_BASE = "outputs/eval_lora_batch"
BATCH_SIZE = 128  # 核心修改：设置批量大小为128
MAX_NEW_TOKENS = 256
EVAL_MAX_SAMPLES = None  # 限制样本数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

os.makedirs(OUTPUT_BASE, exist_ok=True)

# ----------------- 标签标准化（无修改） -----------------
DISEASE_CLASSES = ["WS", "DOWN", "Goldenhar", "Healthy"]
label_map = {
    "williams syndrome": "WS", "williams": "WS", "ws": "WS",
    "down syndrome": "DOWN", "trisomy 21": "DOWN", "down": "DOWN",
    "goldenhar syndrome": "Goldenhar", "oculo-auriculo-vertebral spectrum": "Goldenhar", "goldenhar": "Goldenhar",
    "healthy": "Healthy", "normal": "Healthy", "control": "Healthy",
}

def normalize_label(label):
    if not label: return "Unknown"
    label = str(label).strip().lower()
    if label in label_map: return label_map[label]
    for k in label_map:
        if k in label: return label_map[k]
    return label.capitalize()

# ----------------- JSON 提取（无修改） -----------------
def extract_json_safe(text):
    if not text: return None
    m = re.search(r'(\{[\s\S]*\})', text)
    if not m: return None
    cand = m.group(1)
    try:
        return json.loads(cand)
    except Exception:
        try:
            cand2 = cand.replace("'", '"').replace(",}", "}")
            return json.loads(cand2)
        except Exception:
            return None

# ----------------- message 构造（无修改） -----------------
def build_messages_from_record(rec):
    if "messages" in rec: return rec["messages"]
    inst = rec.get("instruction") or rec.get("prompt") or ""
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": inst}]}]

# ----------------- 核心评估函数（批量推理修改） -----------------
def evaluate_lora(lora_path, test_json_path, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n{'='*70}\n▶ 开始评估 LoRA: {lora_path}\n▶ 测试集: {test_json_path}\n▶ 输出路径: {result_dir}\n▶ 批量大小: {BATCH_SIZE}\n{'='*70}")

    # 加载模型（无修改）
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True, device_map="auto", dtype=DTYPE)
    model = PeftModel.from_pretrained(model, lora_path, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    # 加载测试数据（无修改）
    samples = [json.loads(line) for line in open(test_json_path, "r", encoding="utf-8") if line.strip()]
    if EVAL_MAX_SAMPLES:
        samples = samples[:EVAL_MAX_SAMPLES]
    total_samples = len(samples)
    print(f"✅ 总样本数量: {total_samples}")

    y_true, y_pred, preds_out, errors = [], [], [], []
    device = next(model.parameters()).device

    # 核心修改1：按批次遍历样本（替代单样本循环）
    with tqdm(total=total_samples, desc=f"Evaluating {Path(lora_path).name}") as pbar:
        for batch_start in range(0, total_samples, BATCH_SIZE):
            # 1. 切分当前批次的样本
            batch_end = min(batch_start + BATCH_SIZE, total_samples)
            batch_samples = samples[batch_start:batch_end]
            
            # 2. 批量加载图片+构造prompt（过滤无效样本）
            batch_texts = []  # 批量存储prompt文本
            batch_imgs = []   # 批量存储图片对象
            batch_meta = []   # 批量存储样本元信息（idx、true_label）
            for idx_in_batch, rec in enumerate(batch_samples):
                global_idx = batch_start + idx_in_batch  # 原始样本的全局索引
                try:
                    # 加载图片
                    img_path = rec.get("image") or rec.get("image_path")
                    if not img_path or not os.path.exists(img_path):
                        raise FileNotFoundError(f"Missing image: {img_path}")
                    img = Image.open(img_path).convert("RGB")
                    
                    # 构造prompt文本
                    messages = build_messages_from_record(rec)
                    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
                    
                    # 收集有效样本
                    batch_texts.append(prompt_text)
                    batch_imgs.append(img)
                    batch_meta.append({"global_idx": global_idx, "true_label": normalize_label(rec.get("label"))})
                except Exception as e:
                    # 单个样本错误不影响批次，直接记录
                    errors.append({"index": global_idx, "error": str(e)})
                    continue

            # 3. 跳过空批次（所有样本均无效时）
            if len(batch_texts) == 0:
                pbar.update(batch_end - batch_start)
                continue

            # 4. 核心修改2：批量处理文本和图片（生成batch张量）
            batch = processor(
                text=batch_texts,
                images=[[img] for img in batch_imgs],  # 正确：每个图片用列表包裹，匹配文本批量结构
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            batch = {k: v.to(device) for k, v in batch.items()}  # 转移到GPU

            # 5. 批量推理（无修改，模型自动处理batch维度）
            with torch.no_grad():
                out_ids = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            
            # 6. 批量解码（对应每个样本的输出文本）
            out_texts = processor.batch_decode(out_ids, skip_special_tokens=True)

            # 7. 批量解析标签（对应每个有效样本）
            for text_idx, (out_text, meta) in enumerate(zip(out_texts, batch_meta)):
                global_idx = meta["global_idx"]
                true_label = meta["true_label"]
                
                # 解析预测标签（逻辑与原代码一致）
                parsed = extract_json_safe(out_text)
                pred_label = None
                if parsed and isinstance(parsed, dict):
                    for key in ("diagnosis", "disease", "label"):
                        if key in parsed:
                            pred_label = parsed[key]
                            break
                if not pred_label:
                    low = out_text.lower()
                    for c in DISEASE_CLASSES:
                        if c.lower() in low:
                            pred_label = c
                            break
                if not pred_label:
                    pred_label = "Unknown"
                pred_label = normalize_label(pred_label)

                # 记录结果
                preds_out.append({
                    "index": global_idx,
                    "true": true_label,
                    "pred": pred_label,
                    "raw_output": out_text
                })
                if true_label in DISEASE_CLASSES:
                    y_true.append(true_label)
                    y_pred.append(pred_label)

            # 8. 更新进度条
            pbar.update(batch_end - batch_start)

    # 后续指标计算、结果保存（无修改）
    if not y_true:
        print("⚠️ 没有有效预测，跳过指标计算。")
        return

    report = classification_report(y_true, y_pred, labels=DISEASE_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=DISEASE_CLASSES)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES)
    plt.title(f"Confusion Matrix ({Path(lora_path).name})")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    with open(os.path.join(result_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(result_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for p in preds_out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(result_dir, "error_log.jsonl"), "w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"✅ {Path(lora_path).name} 评估完成并保存结果。")

# ----------------- 主批量流程（无修改） -----------------
if __name__ == "__main__":
    print("🚀 Batch Evaluation Started")

    for lora_root in LORA_ROOTS:
        ckpts = sorted([os.path.join(lora_root, d) for d in os.listdir(lora_root) if d.startswith("checkpoint")])
        all_paths = [lora_root] + ckpts
        for lora_path in all_paths:
            for prompt_name, json_path in TEST_JSONS.items():
                result_dir = os.path.join(
                    OUTPUT_BASE, f"eval_{Path(lora_root).name}_{Path(lora_path).name}_{prompt_name}"
                )
                evaluate_lora(lora_path, json_path, result_dir)

    print("🏁 所有模型评估完成！")