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


DB_PATH = './_db/Faiss17'


db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
    embeddings=embedding,
    allow_dangerous_deserialization=True,
)

# print('=============================')
# #문서 저장소 ID확인
# print(db.index_to_docstore_id)
# print('=============================')
# #저장된 결과 확인
# print(db.docstore._dict)
# print('=============================')
# #유사도 검색
# aaa = db.similarity_search('삼성전자 창업주에 대해 알려줘', k=2,)
# print(aaa)
# print('=============================')


################검색기#####################
# retrivers =db.as_retriever(search_kwargs={'k':2})
retrievers =db.as_retriever(
    search_type = 'similarity_score_threshold',
    search_kwargs={
        'k':2,
        'score_threshold':0.7
        },
)


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


from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage


prompt = ChatPromptTemplate.from_messages([
    ('system', """
당신은 벡터 DB에서 검색한 문서를 근거로 답변하는 챗봇입니다.

다음 정책을 반드시 지키세요.
1. 답변은 컨텍스트에 포함된 정보만 근거로 작성하세요.
2. 컨텍스트가 비어 있거나 질문과 관련된 정보가 없으면 정확히 '주어진 정보로는 답변할수없다'라고 답하세요.
3. 컨텍스트에 없는 사실을 추측하거나, 일반 지식 또는 대화 이력으로 보충하지 마세요.
4. 일부 정보만 확인되면 확인된 내용만 답하고, 확인되지 않은 내용은 알 수 없다고 명확히 밝히세요.

컨텍스트:
{context}

질문: {input}
"""),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')
])
   # {context}는 model과 prompt를 이용해서 알아서 생성됨

# 체인 만들기 (아래 두개 체인은 세트로 쓰임)
docu_chain = create_stuff_documents_chain(model, prompt)  # prompt | model
rag_chain = create_retrieval_chain(retrievers, docu_chain) # 검색 | docu_chain



import gradio as gr

def answer_invoke(message, history):
    chat_history = []

    for item in history[-10:]: #최근 10개의 메시지만 저장
        content = item['content']

        if item['role'] == 'user':
            chat_history.append(HumanMessage(content=content))
        elif item['role'] == 'assistant':
            chat_history.append(AIMessage(content=content))
    response = rag_chain.invoke({
        'input' : message,
        'chat_history': chat_history,
    })

    if not response['context']:
        return '주어진 정보로는 답변할수없다'

    #터미널 상으로 실제 검색된 문서 2개 확인.
    for i ,doc in enumerate(response['context'],start=1):
        print(f'\n 검색 문서 {i}')
        print('출처:',doc.metadata.get('source'))
        print('내용: ', doc.page_content)
        print('========================================')


    return response['answer']


# Gradio 인터페이스 만들자
demo = gr.ChatInterface(fn = answer_invoke, title = 'qwen3 FAISS!!')

# Gradio 실행
demo.launch()

'''
코드진행 순서
1.안녕
history = [
    {'role': 'user' , 'content' : '안녕'}
    {'role': 'assistant', "content":'안녕하세요'}
]

2.나는 진창훈이다. 
history = [
    {'role': 'user' , 'content' : '안녕'}
    {'role': 'assistant', "content":'안녕하세요'}
    1.안녕
history = [
    {'role': 'user' , 'content' : '나는 진창훈이다.'}
    {'role': 'assistant', "content":'당신은 진창훈이군요'}
]

3.
'''