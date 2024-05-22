from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# 임베딩 모델 설정
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device":"cuda"})

# 이미 생성된 Chroma 벡터 데이터베이스 로드
persist_directory = "pdf_db"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=instructor_embeddings)

# 검색기(retriever) 생성 및 사용
retriever = vectordb.as_retriever(search_kwargs={"k":5})

question = "국가용 침입차단시스템 보호프로파일에서 정의하는 TOE는 어떤 형태로 제공될까?"
# docs = retriever.get_relevant_documents("I don't want a caffeinated drink. Could you recommend a non-caffeinated and strawberry drink from EDIYA??")
docs = retriever.get_relevant_documents(question)

print(docs)  # 2

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/Llama-2-70B-chat-GPTQ"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


prompt = question
prompt_template = f'''[INST] <<SYS>>
You are a helpful, respectful, and honest assistant. 
Always answer as helpfully as possible, while being safe. 
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
 Please ensure that your responses are socially unbiased and positive in nature. 
 Answers to questions must be found and created in docs.
 If you don't know the answer to a question, say "I don't know.
 " 
Referenced Documents: {docs}
Question: {prompt}
<</SYS>>[/INST]

'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))