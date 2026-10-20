# tool calling : 외부  기능을 가져와서 사용하기!!
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
    base_url= base_url,
)



##################### 0.toll 정의####################
# Tavily는 AI 에이전트·LLM 애플리케이션을 위해 만들어진 웹 검색 API입니다.
# 최신 웹 정보를 검색하고, AI가 답변에 활용하기 좋은 형태로 결과를 제공합니다.
# LangChain/LangGraph에서 도구(tool)로 연결해
# "질문 분석 → Tavily 웹 검색 → 검색 결과를 바탕으로 답변 생성" 흐름에 자주 사용합니다.
#pip install langchain-tavily
# from langchain_tavily import TavilySearch

# web_search = TavilySearch(
#     max_results= 3,
# )
# result = web_search.invoke({
#     'query' : '삼성전자의 오늘 주가 알려줘'
# })

############################################################################
#pip install -U ddgs 이걸로 pip install 해야함
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
web_search = DuckDuckGoSearchRun(
    max_results =5,
)
#Tavily에서 DuckDuckGo로 Tool 변경
############################################################################
result = web_search.invoke({
    'query' : '삼성전자의 오늘 주가 알려줘'
})
# print(result)
# print(result['results'][-1]['content']) #이거 키면 데이터 배열이 달라서 에러남
#DDG 공짜라서 다같이 접속하면 DNS 에러 뜰수도있는데 계속시도하면 됨


# exit()
#챗봇에 tavliy 연결

tools = [web_search]

llm_with_tools = llm.bind_tools(tools)

from typing import Annotated,TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages  #리듀서
from langgraph.prebuilt import ToolNode,tools_condition

#################1. state 정의###############
class State(TypedDict):
    messages : Annotated[list,add_messages]

#################2.노드 정의#################
def chatbot(state : State):
    return{'messages': [llm_with_tools.invoke(state['messages'])]}

###############3. 그래프 구성,노드추가 ####################
#그래프 생성

graph_builder  = StateGraph(State)

graph_builder.add_node('chatbot',chatbot)


# tool_node = ToolNode(tools)
# graph_builder.add_node('tools',tool_node)
graph_builder.add_node('tools',ToolNode(tools)) #위에 코드 이렇게 줄여서 쓸수 있음

##############4.엣지추가#####################################
graph_builder.add_edge(START,'chatbot')

# graph_builder.add_node('chatbot','tools')
graph_builder.add_conditional_edges( #.add_conditional_edges>>>조건문 노드 추가
    'chatbot',
    tools_condition #조건문 (if랑 같음)
    #tool_calls 있으면 'tools' 반환
    #tool_calls 없으면 END 반환
    #tools_condition에  END포함됨 그래서 밑에End 넣어줄 필요없음>>바로 컴파일로 넘어감
)

graph_builder.add_edge('tools','chatbot')

################5.컴파일 #####################
graph = graph_builder.compile()


################6.그래프 실행#################
query = '삼성전자의 내일 주가 알려줘'

response = graph.invoke({
    'messages' : [('user',query)]
})

print('AI :', response['messages'][-1].content)
