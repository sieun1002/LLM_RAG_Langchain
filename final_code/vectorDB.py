from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

# 임베딩 모델 설정
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS", model_kwargs={"device":"cuda"})

# 이미 생성된 Chroma 벡터 데이터베이스 로드
persist_directory = "firewall"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=instructor_embeddings)

# 검색기(retriever) 생성 및 사용
retriever = vectordb.as_retriever()
# docs = retriever.get_relevant_documents("I don't want a caffeinated drink. Could you recommend a non-caffeinated and strawberry drink from EDIYA??")
docs = retriever.get_relevant_documents("국가용 침입차단시스템 보호프로파일에서 정의하는 TOE는 어떤 형태로 제공될까?")

print(docs)  # 2
