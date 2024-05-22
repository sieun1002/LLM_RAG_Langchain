import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("quantumaikr/llama-2-70b-fb16-korean")
model = AutoModelForCausalLM.from_pretrained("quantumaikr/llama-2-70b-fb16-korean", torch_dtype=torch.float16, device_map="auto").to(device=f"cuda", non_blocking=True)
model.eval()

system_prompt = "### System:\n귀하는 지시를 매우 잘 따르는 AI인 QuantumLM입니다. 최대한 많이 도와주세요. 안전에 유의하고 불법적인 행동은 하지 마세요.\n\n"

message = "인공지능이란 무엇인가요?"
prompt = f"{system_prompt}### User: {message}\n\n### Assistant:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, do_sample=True, temperature=0.9, top_p=0.75, max_new_tokens=4096)

print(tokenizer.decode(output[0], skip_special_tokens=True))
