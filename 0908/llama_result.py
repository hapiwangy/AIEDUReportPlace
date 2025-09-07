# # import necessary
# ## here we only deal with dataset == MathDial
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
# from packages import prompts
# import os
# import json

# current_dir = os.path.dirname(__file__)
# txt_dir = os.path.join(current_dir)

# # read the prompt
# MathDialPrompt = ""
# BridgePrompt = ""
# with open(os.path.join(txt_dir, "MRBench_V1/llama_prompt_Bridge.txt"), "r", encoding="utf-8") as f:
#     BridgePrompt = f.read()

# with open(os.path.join(txt_dir, "MRBench_V1/llama_prompt_MathDial.txt"), "r", encoding="utf-8") as f:
#     MathDialPrompt = f.read()

# current_json_file = os.path.join(txt_dir, "MRBench_V1/extract_data.json")
# with open(current_json_file, "r", encoding="utf-8") as fp:
#     json_data = json.load(fp)

# # set the model tokenizer pipeline
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
  
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     dtype=torch.float16,
#     device_map="auto"
# )
# generator = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer
# )
# gen_cfg = GenerationConfig(
#     do_sample=False,                
#     max_new_tokens=200,             
#     eos_token_id=tokenizer.eos_token_id,
#     pad_token_id=tokenizer.pad_token_id,
# )
# def safe_cut_at_first_heading(text: str) -> str:

#     if "###" in text:
#         return text.split("###", 1)[0].strip()
#     return text.strip()
# final_result = []
# # used to decided the read in data is txt or json
# org = True
# for x in range(len(json_data)):
#     print(f"doning {x}")
#     cur_data = json_data[x]
#     temp = {}
#     if cur_data['Data'] == "MathDial":
#         prompt = prompts.MathDial_Prompt(MathDialPrompt, cur_data, org) if cur_data['Data'] == "MathDial" else prompts.Bridge_Prompt(BridgePrompt, cur_data, org)
#         sequences = generator(
#             prompt,
#             generation_config=gen_cfg,
#             return_full_text=False,        
#             truncation=True,
#         )
#         result = safe_cut_at_first_heading(sequences[0]['generated_text'])
#         temp["result"] = result
#         temp["Data"] = cur_data["Data"]
#         temp["conversation_history"] = cur_data["conversation_history"]
#         temp["Topic"] = cur_data["Topic"]
#         temp["Ground_Truth_Solution"] = cur_data["Ground_Truth_Solution"]
#         final_result.append(temp)
#     else:
#         continue
# with open("./AI+EDU project/result/output.json", "w", encoding="utf-8") as f:
#     json.dump(final_result, f, ensure_ascii=False, indent=2)

# import necessary
## here we only deal with dataset == MathDial
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from packages import prompts

current_dir = os.path.dirname(__file__)
txt_dir = os.path.join(current_dir)

# read the prompt
MathDialPrompt = ""
BridgePrompt = ""
with open(os.path.join(txt_dir, "MRBench_V1/llama_prompt_Bridge.txt"), "r", encoding="utf-8") as f:
    BridgePrompt = f.read()

with open(os.path.join(txt_dir, "MRBench_V1/llama_prompt_MathDial.txt"), "r", encoding="utf-8") as f:
    MathDialPrompt = f.read()

current_json_file = os.path.join(txt_dir, "MRBench_V1/extract_data.json")
with open(current_json_file, "r", encoding="utf-8") as fp:
    json_data = json.load(fp)

# set the model tokenizer pipeline
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# ✅ 讓 batched pipeline 能 padding（很重要）
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"  # causal LM 批次較穩

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto"
).eval()

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

gen_cfg = GenerationConfig(
    do_sample=False,
    max_new_tokens=200,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

def safe_cut_at_first_heading(text: str) -> str:
    """保留到第一個 '###' 出現前；若沒有 '###' 就回傳全文"""
    return text.split("###", 1)[0].strip()

final_result = []
org = True

# 收集要處理的索引與 prompt（僅處理 MathDial，其餘略過：與原本一致）
idx_list, prompt_list = [], []
for x in range(len(json_data)):
    cur_data = json_data[x]
    prompt = prompt = prompts.MathDial_Prompt(MathDialPrompt, cur_data, org) if cur_data['Data'] == "MathDial" else prompts.Bridge_Prompt(BridgePrompt, cur_data, org)
    idx_list.append(x)
    prompt_list.append(prompt)

# 逐批推論 + 進度列
batch_size = 8  # 依 GPU 記憶體調整
for start in tqdm(range(0, len(prompt_list), batch_size), desc="Generating", unit="batch"):
    end = min(start + batch_size, len(prompt_list))
    batch_prompts = prompt_list[start:end]
    batch_idxs = idx_list[start:end]

    generations = generator(
        batch_prompts,
        batch_size=len(batch_prompts),   # 避免最後一批不足 batch_size
        generation_config=gen_cfg,
        return_full_text=False,
        truncation=True,
    )

    # pipeline 對每筆通常回傳 [{'generated_text': ...}]
    def _take_text(g):
        if isinstance(g, list):
            return g[0]['generated_text']
        return g['generated_text']

    for x, g in zip(batch_idxs, generations):
        print(f"doning {x}")  # 保留你原來的進度印出
        cur_data = json_data[x]
        temp = {}
        result = safe_cut_at_first_heading(_take_text(g))
        temp["result"] = result
        temp["Data"] = cur_data["Data"]
        temp["conversation_history"] = cur_data["conversation_history"]
        temp["Topic"] = cur_data["Topic"]
        temp["Ground_Truth_Solution"] = cur_data["Ground_Truth_Solution"]
        final_result.append(temp)

# 輸出結果（與原本一致）
with open("./AI+EDU project/result/output.json", "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)

print(f"Saved {len(final_result)} items to ./AI+EDU project/result/output.json")
