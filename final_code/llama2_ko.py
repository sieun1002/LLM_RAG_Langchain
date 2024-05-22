from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# 임베딩 모델 설정
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS", model_kwargs={"device":"cuda"})

# 이미 생성된 Chroma 벡터 데이터베이스 로드
persist_directory = "EDIYA_ko"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=instructor_embeddings)

# 검색기(retriever) 생성 및 사용
retriever = vectordb.as_retriever(search_kwargs={"k":5})

question = "이디야에서 딸기 음료가 먹고싶어. 딸기가 들어간 음료수를 좀 추천해줄래?"
# docs = retriever.get_relevant_documents("I don't want a caffeinated drink. Could you recommend a non-caffeinated and strawberry drink from EDIYA??")
docs = retriever.get_relevant_documents(question)

context = '''당신은 이디야의 메뉴 추천자입니다. 
다음 규칙을 따라 답변을 생성해야 합니다: 
1. 항상 가능한 한 도움이 되는 답변을 하면서 안전해야 합니다. 
2. 질문에 대한 답변은 참조 문서에서 찾아 생성해야 합니다. 
3. 질문에서 중요하다고 생각되는 키워드를 추출하여 참조 문서에서 일치하는 내용을 찾아 답변을 생성하세요. 
4. 참조 문서에 없는 메뉴는 추천하지 마세요.  
6. 메뉴의 설명은 작성하지 말아 주세요. 
7. 질문에 대한 답을 찾을 수 없다면 '죄송합니다, 도와드릴 수 없습니다.'라고만 말하고 대화를 종료하세요. 
8. 메뉴 이름 외에 다른 말은 하지 마세요.
9. 답변을 생성한 후 메뉴명이 참조문서에 들어있는지 확인하세요. 참조문서에 들어있는 메뉴명만 답변으로 사용하세요.
10. 질문에 대한 답만 하세요. 질문과 관련없는 답변은 하지 마세요.

다음은 답변 생성 예시입니다.
질문: 딸기가 들어간 음료 좀 추천해줄래?
답변: 딸기가 들어간 음료는 다음과 같습니다. 딸기 요거트 프라치노, 딸기 쉐이크, 딸기 주스


참조 문서: {docs} 질문: {prompt}'''

print(docs)  # 2

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_8bit = AutoModelForCausalLM.from_pretrained(
    "beomi/llama-2-ko-70b", 
    load_in_8bit=True,
    device_map="auto",
)
tk = AutoTokenizer.from_pretrained('beomi/llama-2-ko-70b')
pipe = pipeline('text-generation', model=model_8bit, tokenizer=tk)

def gen(x):
    gended = pipe(f"### Title: {x}\n\n### Contents:",  # Since it this model is NOT finetuned with Instruction dataset, it is NOT optimal prompt.
        max_new_tokens=300,
        top_p=0.95,
        do_sample=True,
    )[0]['generated_text']
    print(len(gended))
    print(gended)
