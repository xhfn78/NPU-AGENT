#11-1카피
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
embedding =OpenAIEmbeddings(
    model ='text-embedding-3-small', #3072 차원갯수
    api_key=api_key,
    base_url= base_url,
    # dimensions=5,
)


DB_PATH = './_db/PyPdf'


db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
    embeddings=embedding,
    allow_dangerous_deserialization=True,
)



retrivers =db.as_retriever(search_kwargs={'k':2})
########모델 연결###############################

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    # model='gpt-5-nano',
    model_name='gpt-5.4-nano',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template('''


컨텍스트: {context}
질문: {input}
답변:
''')

#체인만들기
docu_chain = create_stuff_documents_chain(model,prompt) #prompt | model 이랑 같은 역활
rag_chain = create_retrieval_chain(retrivers,docu_chain)#벡터db에있는 청크 문서와 docu_chain 연결 


###############Gradio 챗봇 ###################
import gradio as gr
def answer_invoke(messge, history):
    response = rag_chain.invoke({'input': messge})
    return response['answer']

#Gradio 인터페이스 만들기
demo = gr.ChatInterface(fn= answer_invoke, title='chatbot')

#Gradio 실행
demo.launch(share=True)
###############Gradio 챗봇 ###################
