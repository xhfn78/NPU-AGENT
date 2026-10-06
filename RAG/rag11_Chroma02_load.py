import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter,TextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()


api_key = os.environ['MONOROUTER_API_KEY'].strip() #strip env에 저장한 키값 줄바꾸기 ,공백  인정안하게해줌
base_url= 'https://monogpt.kr/api/monorouter/v1/'
#데이터 불러오기
path = './_data/rag_data/'

# loader1 = TextLoader(path + 'samsung_outlook.txt',encoding='utf-8')
# loader2 = TextLoader(path + 'nvidia_outlook.txt',encoding='utf-8')

# #문서 자르기
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size =300,
#     chunk_overlap =100,
#     separators =['\n\n', '\n','',''], #디폴트값
# )
# #데이터 자르기 /청킹

# split_doc1 = loader1.load_and_split(text_splitter) #청크 300,오버랩 100 기준으로 짜름
# split_doc2 = loader2.load_and_split(text_splitter) #청크 300,오버랩 100 기준으로 짜름

# # print(split_doc1)
# # print(len(split_doc1),len(split_doc2)) #9 9 <<<#documents 갯수

from langchain_openai import OpenAIEmbeddings
embedding =OpenAIEmbeddings(
    model ='text-embedding-3-small', #3072 차원갯수
    api_key=api_key,
    base_url= base_url,
    # dimensions=5,
)

DB_PATH = './_db/Chroma11/' 


db = Chroma(
    embedding_function=embedding,
    persist_directory=DB_PATH,
    collection_name='croma11'
)
 
#저장된 데이터 확인 
print('==============================')
print(db.get)
print('==============================')

aaa = db.similarity_search('삼성전자 사업전망에 대해 알려줘', k=2)# 디폴트 4 문서 4개가져옴

print(aaa)