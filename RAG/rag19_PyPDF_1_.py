import os
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter,TextSplitter
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip() #strip env에 저장한 키값 줄바꾸기 ,공백  인정안하게해줌
base_url= 'https://monogpt.kr/api/monorouter/v1/'



path = './_data/'
pdf_loader = PyPDFLoader(path + 'attention is all you need.pdf')
# pdf_docs = pdf_loader.load()

# # print(type(pdf_docs))
# # print(len(pdf_docs))
# # print(pdf_docs)
# print('==============================')
# print(pdf_docs[0])
# print('==============================')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size =500,
    chunk_overlap =100,
    separators =['\n\n', '\n','',''], #디폴트값
)
#데이터 자르기 /청킹

split_doc = pdf_loader.load_and_split(text_splitter) #청크 300,오버랩 100 기준으로 짜름
#pip install faiss-cpu
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
#Faiss는 유클리드 거리 검색에 강력함(대규모)


#3임베딩
from langchain_openai import OpenAIEmbeddings
embedding =OpenAIEmbeddings(
    model ='text-embedding-3-small', #3072 차원갯수
    api_key=api_key,
    base_url= base_url,
    # dimensions=5,
)

####################FAISS 변경부분############################
faiss_index = faiss.IndexFlatL2(len(embedding.embed_query('hello world')))
print('faiss 인덱스 초기화 준비 완료')



print(faiss_index.d) #1536
#db정의
faiss_db = FAISS(
    embedding_function=embedding,
    index=faiss_index,
    docstore=InMemoryDocstore(), 
    index_to_docstore_id={},

)
#저장된 문서의 갯수 확인.
print(faiss_db.index.ntotal) #현재 db에 저장되어있지않아서 0나옴-> 준비과정
#####################준비완료################


db = FAISS.from_documents(
    documents=split_doc,
    embedding=embedding,
)
DB_PATH = './_db/PyPdf'
db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index17'
)