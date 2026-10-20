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


############1. state 정의###############
class State(TypedDict):
    messages : Annotated[list,add_messages]
    aaa : str
#위 클래스를 상세 분해하면 아래처럼 동작함

# state ={
#     'messages': [],
#     'aaa' : 'abc'
# }


#############2. node 정의####################
def chatbot(state : State):
    return{'messages': [llm.invoke(state['messages'])]}


#############3.그래프 구성#############

graph_01 = StateGraph(State)
graph_01.add_node('chatbot',chatbot)


###########4.엣지 추가 ##############
graph_01.add_edge(START,'chatbot')
graph_01.add_edge('chatbot',END)

###############5.그래프 컴파일################
graph = graph_01.compile()

###############6. 그래프 실행################
query = '삼성전자의 주가를 예측해줘'

response = graph.invoke({
    'messages' :[('user',query)]
})

print(response)
print(response['messages'][0]) #휴먼 메세지
print(response['messages'][1]) #Ai답변 메세지
print(response['messages'][1].content) 
print(response['messages'][0].type) #휴먼
print(response['messages'][1].type)  #ai
