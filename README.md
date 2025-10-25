# WGSY
罕见病分析及检测项目代码
本仓库包含四个核心 Python 脚本，用于医学面部图像的多模态数据处理、模型训练、评估及注释生成，主要针对面部特征与遗传综合征关联分析场景。
目录
文件说明
依赖项
使用方法
注意事项


文件说明
1. converttext.py
功能：将医学图像注释文件转换为多模态训练数据（JSONL 格式），用于后续模型微调。
输入：MedGemma 注释文件（如train_ws_features.json、test_ws_features.json）、图像数据集根目录
输出：
多模态训练集（multimodal_train.jsonl）
多模态测试集（multimodal_test.jsonl）
合并数据集（multimodal_all.jsonl，可选）
核心逻辑：
从原始注释中提取 JSON 结构数据
解析图像路径并映射到绝对路径
构建标准化的指令（instruction）和响应（response），包含面部特征描述、诊断标签等信息
2. train_medgemma.py
功能：基于 LoRA（Low-Rank Adaptation）对 MedGemma 模型进行微调，支持多组参数实验，并自动评估模型性能。
输入：converttext.py生成的 JSONL 数据集、基础模型路径
输出：
微调后的 LoRA 适配器（按参数分组保存）
训练日志、评估指标（准确率、每类疾病正确率）
预测结果与错误日志（JSONL 格式）
核心逻辑：
遍历不同 LoRA 参数（r值：4/8/16；lora_dropout：0.1/0.2/0.3）
配置训练参数（批量大小、学习率、epochs 等）
训练后自动评估模型在测试集上的表现，计算分类准确率
3. eval_lora_batch.py
功能：批量评估微调后的 LoRA 模型，支持高批量推理以提升 GPU 利用率，生成详细评估报告。
输入：LoRA 模型路径、测试集 JSONL 文件
输出：
混淆矩阵（可视化热力图）
分类指标报告（精确率、召回率、F1 值）
预测结果与错误日志（JSONL 格式）
核心逻辑：
批量加载图像与文本提示，高效执行模型推理
标准化标签（如将 "ws" 映射为 "Williams Syndrome"）
计算并可视化评估指标，支持多模型、多测试集对比
4. three_text_gen.py
功能：使用 OpenAI API 生成面部特征结构化注释，用于扩充训练数据（基于 UTKFace 等数据集）。
输入：UTKFace 图像目录、诊断标签（如 "healthy"、"Williams Syndrome"）
输出：包含图像路径、标签及结构化面部特征的 JSON 文件
核心逻辑：
将图像编码为 Base64 格式，结合诊断标签构造提示（Prompt）
调用 OpenAI API 生成标准化面部特征描述（如额头形状、眼距、鼻型等）
支持分批次处理，避免重复标注已处理图像


依赖项
安装所需依赖：
bash
pip install torch transformers peft trl datasets pillow openai scikit-learn matplotlib seaborn tqdm


使用方法
1. 数据转换（converttext.py）
bash
python converttext.py \
  --root /path/to/dataset_root \
  --ann-dir /path/to/annotations \
  --out-dir /path/to/multimodal_data \
  --which both  # 转换训练集和测试集
2. 模型微调（train_medgemma.py）
bash
python train_medgemma.py \
  --model-path /path/to/medgemma-base \
  --train-ds /path/to/multimodal_train \
  --test-ds /path/to/multimodal_test \
  --parent-output-dir /path/to/train_results \
  --num-epochs 3 \
  --load-in-4bit  # 4-bit量化节省显存
3. 批量评估（eval_lora_batch.py）
需先修改脚本中BASE_MODEL_PATH、LORA_ROOTS、TEST_JSONS等配置：
bash
python eval_lora_batch.py
4. 生成注释（three_text_gen.py）
需配置 OpenAI API 密钥和模型信息，然后运行：
bash
python three_text_gen.py  # 处理前5000张图像，可通过参数调整批次


注意事项
数据格式：输入注释文件需包含image（图像路径）、label（诊断标签）等字段，具体参考converttext.py的解析逻辑。
硬件要求：模型训练建议使用 GPU（如 NVIDIA A100/A800），支持 4-bit 量化以降低显存占用。
API 密钥：three_text_gen.py需配置有效的 OpenAI API 密钥，建议通过环境变量设置（export OPENAI_API_KEY=xxx）。
参数调整：train_medgemma.py中的r和lora_dropout可根据需求在脚本中修改，eval_lora_batch.py的BATCH_SIZE需根据 GPU 显存调整。
