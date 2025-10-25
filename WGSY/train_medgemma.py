import os
import json
import re
import argparse
from pathlib import Path
from PIL import Image
from datasets import load_from_disk
import torch

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

# ----------------- CLI -----------------
parser = argparse.ArgumentParser(description="Multi-disease LoRA fine-tuning for MedGemma")
parser.add_argument("--model-path", default="")
# 修改训练和测试数据集路径
parser.add_argument("--train-ds", default="")
parser.add_argument("--test-ds", default="")
# 父输出目录（不再指定具体子目录，子目录由参数动态生成）
parser.add_argument("--parent-output-dir", default="", help="父目录，子目录将自动包含r和dropout参数")
parser.add_argument("--load-in-4bit", action="store_true", help="Load in 4-bit quantization")
parser.add_argument("--use-fp16", action="store_true", help="Use FP16 training")
parser.add_argument("--merge", action="store_true", help="Optionally merge LoRA into base model (not recommended)")
parser.add_argument("--offload-folder", default=None, help="Folder for CPU/NVMe offload")
parser.add_argument("--num-epochs", type=int, default=3)
parser.add_argument("--logging-steps", type=int, default=5)
# 修改评估子集默认值为None（表示使用全部测试集）
parser.add_argument("--eval-subset", type=int, default=None, help="Number of test samples to evaluate (use all if None)")
parser.add_argument("--max-new-tokens", type=int, default=256)
args = parser.parse_args()

# ----------------- 定义要遍历的参数（增加r=16） -----------------
# 待遍历的r值（4、8、16）和lora_dropout值（0.1、0.2、0.3）
r_list = [4, 8, 16]  # 增加r=16的情况
lora_dropout_list = [0.1, 0.2, 0.3]
# 确保父输出目录存在
os.makedirs(args.parent_output_dir, exist_ok=True)

# ----------------- Environment Setup -----------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
train_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ----------------- 加载数据集（只加载一次，避免重复耗时） -----------------
print("📂 Loading datasets...")
train_ds = load_from_disk(args.train_ds)
test_ds = load_from_disk(args.test_ds)
# 确定评估数据集大小（如果未指定则使用全部测试集）
EVAL_SUBSET = len(test_ds) if args.eval_subset is None else min(args.eval_subset, len(test_ds))

# ----------------- 嵌套循环遍历参数，执行实验 -----------------
for r in r_list:
    for lora_dropout in lora_dropout_list:
        # 1. 动态生成当前实验的输出目录（包含r和dropout参数）
        current_output_dir = os.path.join(
            args.parent_output_dir, 
            f"medgemma-r{r}-dropout{lora_dropout}"  # 目录名格式：medgemma-r16-dropout0.1（新增格式）
        )
        os.makedirs(current_output_dir, exist_ok=True)
        print(f"\n{'='*50}")
        print(f"🚀 开始实验：r={r}, lora_dropout={lora_dropout}")
        print(f"📁 实验结果保存至：{current_output_dir}")
        print(f"{'='*50}")

        # 2. 重新加载基础模型（每次实验用全新模型，避免参数污染）
        print(f"🔧 加载基础模型（r={r}, lora_dropout={lora_dropout}）...")
        load_kwargs = dict(
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=train_dtype,
        )
        if args.offload_folder:
            offload_path = os.path.join(args.offload_folder, f"r{r}_dropout{lora_dropout}")
            os.makedirs(offload_path, exist_ok=True)
            load_kwargs["offload_folder"] = offload_path

        if args.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                load_kwargs["quantization_config"] = bnb_config
            except ImportError:
                print("⚠️ bitsandbytes not available, continuing without 4bit.")

        # 每次实验重新加载模型
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, **load_kwargs)
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        processor.tokenizer.padding_side = "right"

        # 3. 动态构建LoRA配置（lora_alpha与r保持一致）
        peft_config = LoraConfig(
            lora_alpha=r,  # 关键：lora_alpha值跟随当前r值
            lora_dropout=lora_dropout,  # 跟随当前遍历的dropout值
            r=r,  # 跟随当前遍历的r值（包括新增的16）
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head", "embed_tokens"],
        )

        # 4. 动态构建Trainer配置（输出目录为当前实验目录）
        sft_args = SFTConfig(
            output_dir=current_output_dir,  # 输出目录指向当前实验目录
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=args.logging_steps,
            save_strategy="epoch",
            eval_strategy="epoch",
            learning_rate=2e-4,
            fp16=args.use_fp16,
            bf16=not args.use_fp16,
            warmup_ratio=0.03,
            lr_scheduler_type="linear",
            report_to="tensorboard",
            remove_unused_columns=False,
            label_names=["labels"],
        )

        # 5. 数据整理函数（复用原逻辑）
        def collate_fn(examples):
            texts, images = [], []
            for ex in examples:
                img_path = ex.get("image") or ex.get("image_path")
                if not img_path or not os.path.exists(img_path):
                    raise FileNotFoundError(img_path)
                text = processor.apply_chat_template(ex["messages"], add_generation_prompt=False, tokenize=False)
                texts.append(text)
                images.append([Image.open(img_path).convert("RGB")])
            batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
            labels = batch["input_ids"].clone()
            pad = processor.tokenizer.pad_token_id
            labels[labels == pad] = -100
            labels[labels == 262144] = -100  # for safety
            batch["labels"] = labels
            return batch

        # 6. 执行训练
        # 评估数据集处理：如果是全量则不做select，否则取前EVAL_SUBSET个
        eval_dataset = test_ds.shuffle() if EVAL_SUBSET == len(test_ds) else test_ds.shuffle().select(range(EVAL_SUBSET))
        
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_ds,
            eval_dataset=eval_dataset if EVAL_SUBSET > 0 else None,
            peft_config=peft_config,
            processing_class=processor,
            data_collator=collate_fn,
        )

        print(f"🧠 开始微调（r={r}, lora_dropout={lora_dropout}）...")
        trainer.train()
        trainer.save_model(current_output_dir)
        print(f"✅ LoRA适配器已保存至：{current_output_dir}")

        # 7. 推理评估（使用完整测试集）
        print(f"🔍 开始评估（r={r}, lora_dropout={lora_dropout}）...")
        from tqdm import tqdm

        # 加载当前实验的LoRA适配器
        model_inf = PeftModel.from_pretrained(model, current_output_dir, device_map="auto")
        processor_inf = processor

        def extract_json_safe(text):
            if not text:
                return None
            m = re.search(r'(\{[\s\S]*\})', text)
            if not m:
                return None
            try:
                return json.loads(m.group(1))
            except Exception:
                try:
                    s = m.group(1).replace("'", '"')
                    s = re.sub(r",\s*}", "}", s)
                    return json.loads(s)
                except Exception:
                    return None

        # 统一标签映射表（简写 -> 全称）
        label_mapping = {
            "ws": "Williams Syndrome",
            "williams": "Williams Syndrome",
            "williams syndrome": "Williams Syndrome",
            "down": "Down Syndrome",
            "down syndrome": "Down Syndrome",
            "goldenhar": "Goldenhar Syndrome",
            "goldenhar syndrome": "Goldenhar Syndrome",
            "healthy": "Healthy",
            "normal": "Healthy",
        }

        def normalize_label(label):
            if not label:
                return "Unknown"
            label = str(label).strip().lower()
            for k, v in label_mapping.items():
                if k in label:
                    return v
            return "Unknown"

        # 评估类别
        disease_classes = ["Williams Syndrome", "Down Syndrome", "Goldenhar Syndrome", "Healthy"]
        correct_per_class = {c: 0 for c in disease_classes}
        total_per_class = {c: 0 for c in disease_classes}
        preds = []
        error_log = []

        # 测试时使用完整测试集（移除select限制）
        eval_subset = test_ds
        for i, ex in enumerate(tqdm(eval_subset, desc=f"Evaluating (r={r}, dropout={lora_dropout})")):
            try:
                img_path = ex.get("image") or ex.get("image_path")
                if not img_path or not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image not found: {img_path}")
                img = Image.open(img_path).convert("RGB")

                prompt = processor_inf.apply_chat_template(ex["messages"], add_generation_prompt=False, tokenize=False)
                batch = processor_inf(text=[prompt], images=[[img]], return_tensors="pt", padding=True)
                batch = {k: v.to(next(model_inf.parameters()).device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                with torch.no_grad():
                    out_ids = model_inf.generate(** batch, max_new_tokens=args.max_new_tokens, do_sample=False)
                out_text = processor_inf.tokenizer.decode(out_ids[0], skip_special_tokens=True)
                parsed = extract_json_safe(out_text)

                pred_diag_raw = parsed.get("diagnosis") if parsed else "Unknown"
                pred_diag = normalize_label(pred_diag_raw)
                true_label_raw = ex.get("label")
                true_label = normalize_label(true_label_raw)

                preds.append({
                    "index": i,
                    "image": img_path,
                    "true_raw": true_label_raw,
                    "pred_raw": pred_diag_raw,
                    "true": true_label,
                    "pred": pred_diag,
                    "text": out_text,
                })

                # 统计正确率
                if true_label in disease_classes:
                    total_per_class[true_label] += 1
                    if pred_diag == true_label:
                        correct_per_class[true_label] += 1
            except Exception as e:
                error_log.append({"index": i, "error": str(e), "sample": ex.get("image")})
                continue

        # 汇总当前实验结果
        print(f"\n📊 实验结果（r={r}, lora_dropout={lora_dropout}）:")
        for c in disease_classes:
            total = total_per_class[c]
            correct = correct_per_class[c]
            acc = correct / total if total > 0 else 0.0
            print(f"{c:20s}: {acc:.3f} ({correct}/{total})")

        overall_acc = sum(correct_per_class.values()) / max(1, sum(total_per_class.values()))
        print(f"✅ 总体准确率: {overall_acc:.3f}")

        # 保存当前实验的预测结果和错误日志
        preds_path = os.path.join(current_output_dir, "eval_preds_multidisease.jsonl")
        with open(preds_path, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        error_path = os.path.join(current_output_dir, "eval_error_log.jsonl")
        with open(error_path, "w", encoding="utf-8") as f:
            for e in error_log:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        print(f"✅ 预测结果保存至: {preds_path}")
        print(f"⚠️ 错误日志保存至: {error_path}")
        print(f"🎉 实验（r={r}, lora_dropout={lora_dropout}）完成！\n")

        # 清理GPU内存，避免后续实验内存溢出
        del model, model_inf, trainer
        torch.cuda.empty_cache()




