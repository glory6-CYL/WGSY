import os
import json
import base64
from tqdm import tqdm
from openai import OpenAI

# ========== 基础配置 ==========
API_KEY = ""  # 建议放在环境变量里 export OPENAI_API_KEY=xxxx
API_BASE = ""
MODEL_NAME = ""

# 输入数据集路径（UTKFace）
UTKFACE_ROOT = ""
# 输出目录和文件名
OUTPUT_DIR = ""
OUTPUT_FILENAME = ""

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# ========== Prompt模板 ==========
PROMPT_TEMPLATE = r"""
You are a clinical geneticist specializing in craniofacial dysmorphology.
You are given a facial image and the confirmed diagnosis of the individual.
Your task is **not to infer**, but to **describe the facial morphology objectively** according to the given label.

====================
DIAGNOSIS: {label}
====================
Describe the observed facial morphology typically consistent with {label}.
Focus on medically relevant features such as forehead, eyes, eyelids, nasal bridge, nose tip, philtrum, mouth, lips, chin, facial symmetry, and ears.

====================
TASK INSTRUCTIONS
====================
1. DO NOT predict or infer any disease. Treat the given diagnosis as factual.
2. Provide structured, factual, English-language morphological descriptors.
3. If any feature cannot be determined clearly, use "uncertain".
4. Keep consistent structure and categories for all samples.

====================
OUTPUT FORMAT (STRICT JSON)
====================
{
  "facial_features": {
    "forehead_shape": "<broad|normal|narrow|asymmetric|uncertain>",
    "eye_spacing": "<wide|normal|narrow|uncertain>",
    "palpebral_fissure_slant": "<upslanted|downslanted|horizontal|uncertain>",
    "eyelid_ptosis": "<present|absent|uncertain>",
    "epicanthal_folds": "<present|absent|uncertain>",
    "periorbital_fullness": "<present|absent|uncertain>",
    "eyebrow_shape": "<arched|straight|irregular|uncertain>",
    "nasal_bridge": "<flat|normal|prominent|uncertain>",
    "nasal_tip": "<upturned|broad|normal|uncertain>",
    "philtrum_length": "<long|normal|short|uncertain>",
    "mouth_shape": "<wide|downturned|normal|asymmetric|uncertain>",
    "lip_thickness": "<full|thin|normal|uncertain>",
    "chin_size": "<small|prominent|normal|uncertain>",
    "facial_asymmetry": "<present|absent|uncertain>",
    "ear_position": "<low_set|normal|high|asymmetric|uncertain>",
    "ear_shape": "<normal|microtia|malformed|uncertain>",
    "hairline": "<low_anterior|normal|high|uncertain>"
  },
  "summary": "A brief, factual English sentence summarizing the facial morphology of a patient with {label}."
}

====================
DISEASE REFERENCE
====================
**Williams Syndrome (WS):**
  Broad forehead, periorbital fullness, upturned nose, long philtrum, wide mouth, full lips, small chin.
**Down Syndrome:**
  Flat nasal bridge, upslanting palpebral fissures, epicanthal folds, small chin, open mouth.
**Goldenhar Syndrome:**
  Facial asymmetry, microtia or malformed ear, mandibular hypoplasia, macrostomia.
**Healthy:**
  Symmetrical facial features, proportional eyes and nose, normal lips and chin, no deformities.
"""


# ========== 辅助函数 ==========
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"[WARN] Image not found: {image_path}")
        return None


def generate_utk_data(start_index=0, batch_size=5000):
    """
    分批次处理UTKFace图像
    :param start_index: 起始索引（从0开始）
    :param batch_size: 本次处理的图像数量
    """
    # 检查UTKFace目录是否存在
    if not os.path.exists(UTKFACE_ROOT):
        print(f"[ERROR] UTKFace directory not found at {UTKFACE_ROOT}")
        return

    # 获取目录下所有图像文件（排序确保处理顺序固定）
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = sorted([  # 排序保证每次处理顺序一致
        f for f in os.listdir(UTKFACE_ROOT)
        if f.lower().endswith(image_extensions)
    ])
    total_images = len(image_files)
    print(f"[INFO] Found {total_images} images in total")

    if total_images == 0:
        print(f"[ERROR] No image files found in {UTKFACE_ROOT}")
        return

    # 准备输出路径和已有结果
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载已处理的结果（避免重复标注）
    existing_results = []
    processed_images = set()  # 记录已处理的图像路径
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
            processed_images = {entry["image"] for entry in existing_results}
        print(f"[INFO] Loaded {len(existing_results)} existing annotations")

    # 计算本次需要处理的图像范围（排除已处理的）
    remaining_images = [
        img for img in image_files
        if os.path.join(UTKFACE_ROOT, img) not in processed_images
    ]
    print(f"[INFO] Remaining images to process: {len(remaining_images)}")

    # 根据start_index和batch_size截取本次处理的子集
    end_index = start_index + batch_size
    batch_images = remaining_images[start_index:end_index]
    if not batch_images:
        print(f"[INFO] No images to process in this batch (start={start_index}, batch_size={batch_size})")
        return
    print(f"[INFO] Processing batch: {start_index} to {min(end_index, len(remaining_images))} ({len(batch_images)} images)")

    # 处理当前批次
    new_results = []
    label = "healthy"
    for img_filename in tqdm(batch_images, desc=f"Processing batch {start_index}-{end_index}"):
        img_path = os.path.join(UTKFACE_ROOT, img_filename)
        
        # 再次检查是否已处理（双重保险）
        if img_path in processed_images:
            continue

        # 编码图像为base64
        base64_image = encode_image_to_base64(img_path)
        if not base64_image:
            continue

        # 构造prompt
        prompt = PROMPT_TEMPLATE.replace("{label}", label)

        try:
            # 调用API
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.2,
                max_tokens=1024,
            )

            generated_text = response.choices[0].message.content

            # 去除可能的```json包裹
            if generated_text.strip().startswith("```json"):
                generated_text = generated_text.strip()[7:-3]

            try:
                parsed_response = json.loads(generated_text)
                entry = {
                    "image": img_path,
                    "label": label,
                    "annotation": parsed_response,
                }
                new_results.append(entry)
                processed_images.add(img_path)  # 标记为已处理
            except json.JSONDecodeError:
                print(f"[WARN] JSON parse failed for {img_path}")
                continue

        except Exception as e:
            print(f"[ERROR] API error for {img_path}: {e}")
            continue

    # 合并已有结果和新结果并保存
    all_results = existing_results + new_results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] ✅ Saved {len(all_results)} total entries (added {len(new_results)} in this batch) to {output_path}")


if __name__ == "__main__":
    # 第一次运行：标注前5000张（start_index=0, batch_size=5000）
    # 后续运行：标注剩余图像（例如start_index=5000, batch_size=5000，依此类推）
    generate_utk_data(start_index=0, batch_size=5000)