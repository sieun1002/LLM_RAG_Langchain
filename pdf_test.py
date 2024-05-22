import json
from langchain.vectorstores import Chroma
from langchain.embeddings import FastEmbedEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# KoAlpaca-Polyglot 모델 및 토크나이저 로드
model_name_or_path = "beomi/KoAlpaca-Polyglot-12.8B"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=False, revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
koalpaca_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# PDF 내용을 처리하는 클래스
class ChatPDF:
    def __init__(self):
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )

    def ingest(self, pdf_file_path):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        self.vector_store = Chroma.from_documents(
            documents=chunks, embedding=FastEmbedEmbeddings()
        )

    def ask(self, question):
        if not self.vector_store:
            return "PDF 문서를 먼저 업로드해주세요."
        
        # 검색기를 사용하여 유사한 문서를 검색
        search_results = self.vector_store.search(
            query=question, 
            search_type="similarity",  # "similarity" 또는 "mmr"를 사용
            k=3  # k는 반환할 최대 문서 수
        )
        
        # 검색 결과가 3개보다 적다면 결과 수에 맞게 업데이트
        n_results = min(len(search_results), 3)
        search_results = search_results[:n_results]
        
        context = " ".join([result.text.decode('utf-8') for result in search_results])  # 텍스트를 가져올 때 디코딩
        
        # KoAlpaca-Polyglot 모델을 사용하여 답변 생성
        response = koalpaca_pipe(f"질문: {question}\n\n맥락: {context}\n\n답변:")
        return response[0]['generated_text']

# ChatPDF 인스턴스 생성
chat_pdf = ChatPDF()

# PDF 파일 경로 설정
pdf_file_path = "EDIYA_ko.pdf"  # PDF 파일 경로를 이곳에 지정하세요.

# 메인 로직
if __name__ == "__main__":
    # PDF 파일 처리
    chat_pdf.ingest(pdf_file_path)
    print("PDF 파일 처리 완료.")

    # 사용자 질문에 대한 반복적인 처리
    while True:
        user_input = input("질문을 입력하세요: ")
        answer = chat_pdf.ask(user_input)
        print("답변:", answer)
