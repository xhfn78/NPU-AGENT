from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()


api_key = os.environ['MONOROUTER_API_KEY'].strip() #strip env에 저장한 키값 줄바꾸기 ,공백  인정안하게해줌
base_url= 'https://monogpt.kr/api/monorouter/v1/'

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url= 'https://monogpt.kr/api/monorouter/v1/'
)

from typing import Annotated,TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage,HumanMessage


############1. state 정의####################
class State(TypedDict):
    messages : Annotated[list,add_messages]
    aaa : str
#위 클래스를 상세 분해하면 아래처럼 동작함

# state ={
#     'messages': [],
#     'aaa' : 'abc'
# }


#############2. node 정의####################

def prepare(state : State):
    return{'messages': 
           [AIMessage(content='질문을 접수했습니다.')]
           }

def chatbot(state : State):
    return{'messages': [llm.invoke(state['messages'])]}

def finish(state : State):
    return{'messages': 
           [AIMessage(content='답변 처리가 완료되었습니다.')]
           }
#############3.그래프 구성#############

graph_01 = StateGraph(State)
graph_01.add_node('prepare',prepare)
graph_01.add_node('chatbot',chatbot)
graph_01.add_node('finish',finish)


###########4.엣지 추가 ##############
graph_01.add_edge(START,'prepare')
graph_01.add_edge('prepare','chatbot')
graph_01.add_edge('chatbot','finish')
graph_01.add_edge('finish',END)

###############5.그래프 컴파일################
graph = graph_01.compile()

###############6. 그래프 실행################
query = '삼성전자의 주가를 예측해줘'

response = graph.invoke({
    'messages' :[('user',query)]
})
##################7.결과 출력##########################
for i ,msg in enumerate(response['messages']):
    print(f'[{i}] {msg.type} : {msg.content}')
#             [0] human : content 내용 나옴