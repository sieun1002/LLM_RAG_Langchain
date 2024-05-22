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