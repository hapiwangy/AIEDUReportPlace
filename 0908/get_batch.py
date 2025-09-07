# -*- coding: utf-8 -*-
# Accelerated evaluation with Hugging Face Datasets + batched generation
# 每筆結果都會包含：
#   source_file（來源檔名）、source_index（在來源檔的索引）、
#   conversation_history（從來源檔直通/正規化）、
#   original_response（從來源檔直通/正規化）、
#   eval_response（此腳本用評估模型產生）
# get orgnial result from the LLM
import os
import json
import torch
from math import ceil
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from packages import prompts

# -----------------------------
# 路徑設定
# -----------------------------
current_dir = os.path.dirname(__file__)
txt_dir = os.path.join(current_dir)

llama_json_file = os.path.join(txt_dir, "result/llama_output.json")
mistral_json_file = os.path.join(txt_dir, "result/mistral_output.json")
prompt_path = os.path.join(txt_dir, "MRBench_V1/testing_evalutaion_prompt.txt")

# -----------------------------
# 載入評估模板文字
# -----------------------------
with open(prompt_path, "r", encoding="utf-8") as fp:
    evaluation_prompt = fp.read()

# -----------------------------
# 小工具：將來源欄位正規化
#   - conversation_history：從常見鍵擇一帶出，若是 list/obj 轉成漂亮 JSON 字串保存
#   - original_response：優先使用 tutor_response；否則退回 generated_text
# -----------------------------
def _stringify_if_needed(v):
    if isinstance(v, (dict, list, tuple)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return v

def normalize_fields(example):
    # conversation history: try common keys
    conv_keys_try = [
        "conversation_history", "history", "dialogue", "messages", "conversation",
        "chat_history", "context"
    ]
    conv_val = None
    for k in conv_keys_try:
        if k in example and example[k] not in (None, ""):
            conv_val = example[k]
            break
    example["conversation_history"] = _stringify_if_needed(conv_val)

    # original response: prefer tutor_response; else fallback to generated_text in source
    if "tutor_response" in example and example["tutor_response"] not in (None, ""):
        example["original_response"] = _stringify_if_needed(example["tutor_response"])
    elif "generated_text" in example and example["generated_text"] not in (None, ""):
        # 若來源本身就叫 generated_text，避免與評估輸出衝突，搬到 original_response
        example["original_response"] = _stringify_if_needed(example["generated_text"])
    else:
        example["original_response"] = None

    return example

# -----------------------------
# 載入資料為 Dataset，並加上來源檔名與索引
# -----------------------------
def load_with_source(path: str, source_name: str):
    ds = load_dataset("json", data_files=path)["train"]  # 單檔 → 'train' split
    ds = ds.map(normalize_fields, batched=False)
    # 加上來源檔名與在來源內的索引
    ds = ds.add_column("source_file", [source_name] * len(ds))
    ds = ds.add_column("source_index", list(range(len(ds))))
    # 若來源原本有 generated_text，為避免與評估輸出衝突，將其移到 original_response 後移除舊欄位
    if "generated_text" in ds.column_names:
        ds = ds.remove_columns(["generated_text"])
    return ds

ds_llama   = load_with_source(llama_json_file,   "llama_output.json")
ds_mistral = load_with_source(mistral_json_file, "mistral_output.json")

# 你可以選擇合併處理或分開處理。以下示範合併（保持每筆都標示來源）
dataset = concatenate_datasets([ds_llama, ds_mistral])

# -----------------------------
# 定義 rubric 與 definitions（如需用得到就保留，不影響生成）
# -----------------------------
definitions = {
    "mistake_identification": "Has the tutor identified a mistake in a student’s response?",
    "mistake_location": "Does the tutor’s response accurately point to a genuine mistake and its location?",
    "revealing_answer": "Does the tutor reveal the final answer (whether correct or not)?",
    "providing_guidance": "Does the tutor offer correct and relevant guidance, such as an explanation, elaboration, hint,examples, and so on?",
    "coherent": "Is the tutor’s response logically consistent with the student’s previous response?",
    "actionability": "Is it clear from the tutor’s feedback what the student should do next?",
    "tutor_tone": "Is the tutor’s response encouraging, neutral, or offensive?",
    "humanness": "Does the tutor’s response sound natural, rather than robotic or artificial?",
}
definitions = tuple(definitions.keys())

point2rate = {
    "mistake_identification_rubric": {1: "Yes", 2: "To some extent", 3: "No"},
    "mistake_location_rubric": {1: "Yes", 2: "To some extent", 3: "No"},
    "revealing_answer_rubric": {
        1: "Yes (and the revealed answer is correct",
        2: "Yes (but the revealed answer is incorrect)",
        3: "No",
    },
    "providing_guidance_rubric": {
        1: "Yes (guidance is correct and relevant to the mistake)",
        2: "To some extent (guidance is provided but it is fully or partially incorrect or incomplete)",
        3: "No",
    },
    "coherent_rubric": {1: "Yes", 2: "To some extent", 3: "No"},
    "actionability_rubric": {1: "Yes", 2: "To some extent", 3: "No"},
    "tutor_tone_rubric": {1: "Encouraging", 2: "Neutral", 3: "Offensive"},
    "humanness_rubric": {1: "Yes", 2: "To some extent", 3: "No"},
}

# -----------------------------
# 建立 prompt 欄位（與你原本邏輯一致：evaluation_prompt + 每筆樣本）
# -----------------------------
def build_prompt(example):
    # 將整個 example 丟給你的 evaluation_prompt 組裝器
    # （保持與你原本：prompts.evaluation_prompt(evaluation_prompt, item) 相同邏輯）
    example["__prompt_text"] = prompts.evaluation_prompt(evaluation_prompt, example)
    return example

dataset = dataset.map(build_prompt, batched=False)

# -----------------------------
# 模型與 pipeline（維持你原本參數/行為）
# -----------------------------
model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto"
).eval()

# 關閉梯度、允許 TF32（若硬體支援），屬推論層級最佳化，不改輸出語義
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

gen_cfg = GenerationConfig(
    do_sample=False,
    max_new_tokens=200,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# -----------------------------
# 批次推論（保持產生結果一致；僅提升吞吐）
# -----------------------------
BATCH_SIZE = 8  # 可依 GPU 調大（例如 8/16/32）

def generate_batch(batch):
    prompts_batch = batch["__prompt_text"]
    sequences = generator(
        prompts_batch,
        generation_config=gen_cfg,
        return_full_text=False,
        truncation=True,
        batch_size=BATCH_SIZE,
    )
    outs = []
    for entry in sequences:
        if isinstance(entry, list):
            outs.append(entry[0]["generated_text"])
        else:
            outs.append(entry["generated_text"])
    # 存為 eval_response，避免覆蓋來源欄位
    return {"eval_response": outs}

with torch.inference_mode():
    dataset = dataset.map(generate_batch, batched=True, batch_size=BATCH_SIZE)

# -----------------------------
# 產出只保留需要的欄位：來源與輸出
# -----------------------------
keep_cols = [
    "source_file",
    "source_index",
    "conversation_history",
    "result",
    "eval_response",
]
dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_cols])

# -----------------------------
# 儲存輸出（合併檔 + 依來源各一檔）
# -----------------------------
out_dir = os.path.join(txt_dir, "result")
os.makedirs(out_dir, exist_ok=True)

# 合併輸出（可選擇需要時啟用）
# dataset.to_json(os.path.join(out_dir, "eval_outputs_tagged_all.json"), force_ascii=False)

# 依來源各自輸出（保留 conversation_history / original_response / eval_response）
dataset.filter(lambda x: x["source_file"] == "llama_output.json")  \
       .to_json(os.path.join(out_dir, "eval_outputs_tagged_llama.json"),   force_ascii=False)
dataset.filter(lambda x: x["source_file"] == "mistral_output.json")\
       .to_json(os.path.join(out_dir, "eval_outputs_tagged_mistral.json"), force_ascii=False)
