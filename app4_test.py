from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "beomi/KoAlpaca-Polyglot-12.8B"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "딥러닝이 뭐야?"
# prompt_template=f'''[INST] <<SYS>>
# 당신은 도움이 되고, 존경심이 강하고, 정직한 조수입니다. 안전한 상태에서 항상 가능한 한 도움이 되는 답변을 하세요. 당신의 답변에는 해로운, 비윤리적인, 인종 차별적인, 성차별적인, 유독한, 위험한, 불법적인 내용이 포함되어서는 안 됩니다. 당신의 답변이 사회적으로 편견이 없고, 본질적으로 긍정적인지 확인하세요. 질문이 전혀 이해가 되지 않거나, 사실 일관성이 없다면, 정답이 아닌 것에 답하는 대신 왜 그런지 설명하세요. 질문에 대한 답을 모르면, 잘못된 정보를 공유하지 마세요.
# <</SYS>>
# {prompt}[/INST]

# # '''



# print("\n\n*** Generate:")

# input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.9,  max_new_tokens=256)
# print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    # tokenizer=tokenizer,
    tokenizer=model_name_or_path,
    # max_new_tokens=265,
    # do_sample=True,
    # temperature=0.7,
    # top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

def ask(x, context='', is_input_full=False):
    ans = pipe(
        f"### 질문: {x}\n\n### 맥락: {context}\n\n### 답변:" if context else f"### 질문: {x}\n\n### 답변:", 
        # "text-generation",
        do_sample=True, 
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2,
    )
    print(ans[0]['generated_text'])

# print(pipe(prompt_template)[0]['generated_text'])


ask(prompt)