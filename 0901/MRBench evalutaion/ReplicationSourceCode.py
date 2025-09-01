# import necessary
## here we only deal with dataset == MathDial
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from packages import prompts
import os
import json

current_dir = os.path.dirname(__file__)
txt_dir = os.path.join(current_dir)

# read the prompt
MathDialPrompt = ""
with open(os.path.join(txt_dir, "MRBench_V1/llama_prompt_MathDial.json"), "r", encoding="utf-8") as f:
    MathDialPrompt = json.load(f)

# get the whole dataset
current_json_file = os.path.join(txt_dir, "MRBench_V1/test.json")
with open(current_json_file, "r", encoding="utf-8") as fp:
    json_data = json.load(fp)


# set the model tokenizer pipeline
model_id = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = (
    "<|begin_of_text|>"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|start_header_id|>system<|end_header_id|>{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'user' %}"
    "<|start_header_id|>user<|end_header_id|>{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|start_header_id|>assistant<|end_header_id|>{{ message['content'] }}<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto"
)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)
gen_cfg = GenerationConfig(
    do_sample=False,  # for consistent              
    max_new_tokens=50,  # to avoid to much unrelated words           
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

final_result = []



for x in range(len(json_data)):
    print(f"doning {x}")
    cur_data = json_data[x]
    temp = {}
    if cur_data['Data'] == "MathDial":
        messages = prompts.MathDial_Prompt(MathDialPrompt, cur_data)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sequences = generator(
            prompt,
            generation_config=gen_cfg,
            return_full_text=False,        
            truncation=True,
        )
        result = sequences[0]['generated_text']
        temp["result"] = result
        temp["Data"] = cur_data["Data"]
        temp["conversation_history"] = cur_data["conversation_history"]
        final_result.append(temp)
    else:
        continue
with open("./AI+EDU project/result/output.json", "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)