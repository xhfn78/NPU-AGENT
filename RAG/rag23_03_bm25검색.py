from langchain_core.documents import Document

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = '삼성전자의 반도체 사업'


# ############bm25검색 ########################

#BM25는 키워드 기반 검색.
from langchain_community.retrievers import BM25Retriever
#pip install rank_bm25
bm25_retriver = BM25Retriever.from_documents(texts)
bm25_retriver.k = 3

bm25_docs = bm25_retriver.invoke(query)
  
# print('===================BM25검색 결과===========')
# for doc in bm25_docs:
#     print(doc.page_content)


###################################################
################벡터 검색###########################
import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter,TextSplitter
from langchain_chroma import Chroma

#pip install faiss-cpu
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
#Faiss는 유클리드 거리 검색에 강력함(대규모)




from dotenv import load_dotenv
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip() #strip env에 저장한 키값 줄바꾸기 ,공백  인정안하게해줌
base_url= 'https://monogpt.kr/api/monorouter/v1/'


#3임베딩
from langchain_openai import OpenAIEmbeddings
embeddings =OpenAIEmbeddings(
    model ='text-embedding-3-small', #3072 차원갯수
    api_key=api_key,
    base_url= base_url,
    # dimensions=5,
)

from hashlib import sha256
from langchain_chroma import Chroma

DB_PATH = "./_db/Chroma17"

db = Chroma(
    collection_name="croma17",
    persist_directory=DB_PATH,
    embedding_function=embeddings,
)

# 같은 코드를 다시 실행해도 동일한 문서가 중복 저장되지 않도록 ID 지정
ids = [
    sha256(doc.page_content.encode("utf-8")).hexdigest()
    for doc in texts
]
db.add_documents(documents=texts, ids=ids)

# 검색
db_retriever = db.as_retriever(search_kwargs={"k": 3})
results = db_retriever.invoke(query)

# print("========== Chroma 검색 결과 ==========")
# for doc in results:
#     print(doc.page_content)

#################하이브리드 검색####################

from langchain_classic.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriver,
        db_retriever,
    ],
    weights=[0.5,0.5]
)


hybrid_docs = hybrid_retriever.invoke(query)
print('===================하이브리드 앙상블 검색 결과===========')
for doc in hybrid_docs:
    print(doc.page_content)