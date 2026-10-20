#13-1 카피
import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter,TextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()


api_key = os.environ['MONOROUTER_API_KEY'].strip() #strip env에 저장한 키값 줄바꾸기 ,공백  인정안하게해줌
base_url= 'https://monogpt.kr/api/monorouter/v1/'


#03 임베딩
from langchain_openai import OpenAIEmbeddings
embedding =OpenAIEmbeddings(
    model ='text-embedding-3-small', #3072 차원갯수                
    api_key=api_key,
    base_url= base_url,
    # dimensions=5,
)


DB_PATH = './_db/Chroma12/' 

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory=DB_PATH,
    collection_name='croma12'
)
# print(f'벡터 저장소에 저장된 문서수:{vector_store._collection.count()}',)
#벡터 저장소에 저장된 문서수:30

# query = '삼성전자의 창업자는 누구인가요?'
# result = vector_store.similarity_search(query)

# print(f'검색 결과의 길이: {len(result)}',) 
#디폴트값 4라 4개의 문장 찾아줌
#검색 결과의 길이: 4

#################Retrievers###############
#################검색기####################
retrivers =vector_store.as_retriever(search_kwargs={'k':2})
# print(retrivers)
# aaa = retrivers.invoke(query)
# print(f'검색된 관련 문서 수 :{len(aaa)}')
# print(f'첫번째 관련 문서 내용 미리보기: {aaa[0].page_content[:50]}')

########모델 연결###############################
# print('=================================================')
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    # model='gpt-5-nano',
    model_name='gpt-5.4-nano',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)


# response = model.invoke('엔비디아의 주요 사업은 무엇인가요?')
# print('model의 답변 : ',response.content)
# print('=================================================')

# query_with_context = f'''
#     {aaa[0].page_content}\n\n
#     위 내용에 근거하여 다음 질문에 답변하세요. \n\n{query}
# '''

# response = model.invoke(query_with_context)
# print('model의 응답:',response.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template('''
다음 컨텍스트를 바탕으로 질문에 답변해주세요. 컨텍스트 관련 정보가 없다면 ,
'주어진 정보로는 답변할 수 없습니다.'라고 말씀해 주세요.

컨텍스트: {context}
질문: {input}
답변:
''')

#체인만들기
docu_chain = create_stuff_documents_chain(model,prompt) #prompt | model 이랑 같은 역활
rag_chain = create_retrieval_chain(retrivers,docu_chain)#벡터db에있는 청크 문서와 docu_chain 연결 


#체인 실행

# query = '삼성전자의 창업자는 누구인가요?'
# response = rag_chain.invoke({'input': query})
# print(response)
# print(['================keys======================'])
# print(response.keys()) #dict_keys(['input', 'context', 'answer']
# print(['=================answer====================='])
# print(response['answer'])
# print(['=================context====================='])
# print(response['context'][0].page_content)

###############Gradio 챗봇 ###################
import gradio as gr
def answer_invoke(messge, history):
    response = rag_chain.invoke({'input': messge})
    return response['answer']

#Gradio 인터페이스 만들기
demo = gr.ChatInterface(fn= answer_invoke, title='chatbot')

#Gradio 실행
demo.launch()
###############Gradio 챗봇 ###################