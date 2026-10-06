import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter,TextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()


api_key = os.environ['MONOROUTER_API_KEY'].strip() #strip env에 저장한 키값 줄바꾸기 ,공백  인정안하게해줌
base_url= 'https://monogpt.kr/api/monorouter/v1/'

from glob import glob
path = './_data/rag_data/'
#폴더에서 텍스트 파일 목록 가져오기 
txt_files = glob(os.path.join(path,'*.txt'))

# print(txt_files)#폴더속 txt파일 전부다 가져옴
# ['./_data/rag_data\\2026_AI_for_All.txt', 
#  './_data/rag_data\\nvidia_outlook.txt', 
#  './_data/rag_data\\samsung_outlook.txt']

#리스트로 저장된 파일을 이터레이터(반복문)으로 가져오기
# txt_files에서 파일 경로를 하나씩 꺼내어 문서로 불러온다.
data = []
for text_file in txt_files:  
    loader = TextLoader(text_file,encoding= 'utf-8')
    # data.append(loader)
    data += loader.load()                                   

# print('==============================')
# print(data[0])
# print('==============================')
# print(len(data))
# print(data[0].page_content)

char_count = [len(doc.page_content)for doc in data]
# print(char_count) #[8158, 2049, 1898]


#문서 자르기
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size =500,
    chunk_overlap =100,
    separators =['\n\n', '\n','',''], #디폴트값
)
#데이터 자르기 /청킹

texts = text_splitter.split_documents(data)
# print('생성된 텍스트 청크수:',len(texts)) #30>>> 벡터의 갯수가 30개 생길예정
# print('각 청크의 길이:',list(len(text.page_content)for text in texts))
'''
# 각 청크의 길이: [259, 154, 150, 282, 128, 276, 288, 258, 268, 262, 271, 226, 
# 268, 182, 213, 281, 257, 182, 162, 208, 259, 188, 
# 225, 229, 219, 214, 258, 207, 284, 297, 198, 245, 
# 177, 272, 215, 9, 269, 293, 290, 236, 289, 209, 222, 230, 254, 249, 299, 181, 
# 247, 243, 185, 219, 239, 235, 298, 299, 172, 187, 249]

# 각 청크의 길이 출력 결과
# RecursiveCharacterTextSplitter는 chunk_size=300으로 설정했으므로
# 청크 길이가 최대 300자에 가깝게 생성된다.
#
# 단, 모든 청크가 정확히 300자가 되는 것은 아니다.
# separators=['\n\n', '\n', '', ''] 설정에 따라 문단이나 줄바꿈처럼
# 자연스러운 경계에서 먼저 나누기 때문에, 경계 지점에 따라 300자보다 짧은
# 청크가 만들어질 수 있다.
#
# 또한 chunk_overlap=100이므로 이전 청크의 마지막 최대 100자 정도를
# 다음 청크에 포함한다. 이는 문맥이 청크 사이에서 끊기는 것을 줄이기 위함이다.
#
# 예를 들어 길이가 9인 청크는 문서의 마지막에 남은 짧은 텍스트이거나,
# 줄바꿈 기준으로 분리된 매우 짧은 문장/문단일 가능성이 높다.
# 따라서 [259, 154, ..., 299, 172, ...]처럼 청크마다 길이가 다르게 나오는 것은 정상이다.
# '''
# print('첫번째 청크의 내용 :',texts[0].page_content)
# print('첫번째 청크의 내용 :',len(texts[0].page_content)) # 391
# print('두번째 청크의 내용 :',texts[1].page_content)
# print('두번째 청크의 내용 :',len(texts[1].page_content)) # 391
# metadata={'source': './_data/rag_data\\2026_AI_for_All.txt'}
#청킹 되어서 나온 doc 객체 안에는 metadata(경로),page_content(내용) 2가지가 나옴

#03 임베딩
from langchain_openai import OpenAIEmbeddings
embedding =OpenAIEmbeddings(
    model ='text-embedding-3-small', #3072 차원갯수                
    api_key=api_key,
    base_url= base_url,
    # dimensions=5,
)

sample_text = '삼성전자의 창업자는 누구인가요?'
vector = embedding.embed_query(sample_text)
# print(vector)
# print(len(vector)) #1536



DB_PATH = './_db/Chroma12/' 

#저장

vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=DB_PATH,
    collection_name='croma12'
)
print(f'벡터 저장소에 저장된 문서수:{vector_store._collection.count()}',)
#벡터 저장소에 저장된 문서수:30

query = '삼성전자의 창업자는 누구인가요?'
result = vector_store.similarity_search(query)

print(f'검색 결과의 길이: {len(result)}',) 
#디폴트값 4라 4개의 문장 찾아줌
#검색 결과의 길이: 4

#################Retrievers###############
#################검색기####################
retrivers =vector_store.as_retriever(search_kwargs={'k':2})
print(retrivers)
aaa = retrivers.invoke(query)
print(f'검색된 관련 문서 수 :{len(aaa)}')
print(f'첫번째 관련 문서 내용 미리보기: {aaa[0].page_content[:50]}')
