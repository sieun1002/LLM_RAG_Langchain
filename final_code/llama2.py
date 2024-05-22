from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained(
    "beomi/llama-2-ko-70b", 
    device_map="auto",
)
tk = AutoTokenizer.from_pretrained('beomi/llama-2-ko-70b')
pipe = pipeline('text-generation', model=model, tokenizer=tk)

def gen(x):
    gended = pipe(f"### Title: {x}\n\n### Contents:",  # Since it this model is NOT finetuned with Instruction dataset, it is NOT optimal prompt.
        max_new_tokens=300,
        top_p=0.95,
        do_sample=True,
    )[0]['generated_text']
    print(len(gended))
    print(gended)