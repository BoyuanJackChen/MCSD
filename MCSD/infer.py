import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from inference.generate import Generator

target_model_name = "bigscience/bloomz-7b1"
draft_model_name = "bigscience/bloom-560m"
draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name,
    torch_dtype=torch.float16,
    device_map=0,
)
target_model = AutoModelForCausalLM.from_pretrained(
    target_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(target_model_name)

generator = Generator(
    draft_model,
    target_model,
    eos_token_id=tokenizer.eos_token_id,
    k_config=(4, 2, 2),
    max_new_tokens=128,
    draft_model_temp=0.8,
    target_model_temp=0.8,
    replacement=False,
    speculative_sampling=False
    # tree_attn=False,
)

prompt_text = "What is the weather like in New York during winters? \nOutput: "
inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
input_ids = inputs.input_ids
output = generator.generate(input_ids)
output_text = tokenizer.batch_decode(
    output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("Output:\n{}".format(output_text))
