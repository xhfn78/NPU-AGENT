# lg08-2 카피

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
model_name=' gpt-5.6-terra',
temperature=0,
max_tokens=5000,
api_key=api_key,
base_url=base_url,
)

# 0. Tool 정의
# pip install langchain-tavily
from langchain_tavily import TavilySearch

# 웹 검색 도구 생성 (검색창에서 검색하는 것과 같은 단계다.)
web_search = TavilySearch(
max_results=5,
)
# result = web_search.invoke({
#     'query' : '삼성전자의 오늘 주가 얼마'
# })
# print(result)
# print(result['results'][-1]['content'])

tools = [web_search]

llm_with_tools = llm.bind_tools(tools)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# 메모리 기능 추가 첫번째
from langgraph.checkpoint.memory import InMemorySaver

# 1. State 정의
class State(TypedDict):
    messages : Annotated[list, add_messages]
    search_result : str
    analysis_result : str

# 2. node 정의
def search_agent(state:State):
    print('검색 에이전트 실행')
    question = state['messages'][-1].content

    result = web_search.invoke({
        'query' : question
})
    return {'search_result' : str(result)}

def analysis_agent(state:State):
    print('분석 에이전트 실행')
    question = state['messages'][-1].content

    response = llm.invoke(
        f'다음 질문을 기존 지식을 바탕으로 분석해줘.\n{question}'
    )
    return {'analysis_result' : response.content}

def merge_agent(state:State): #결과 통합 에이전트 부분 변경
    # result = (
    #     state['search_result'] + '\n\n' +
    #     '='*25 + '\n' +
    #     state['analysis_result']
    # )
    # return {'messages' : [('assistant', result)]}
    print('결과 통합 에이전트 실행')

    prompt = f'''
    다음  두 결과를 종합하여 최종 답변을 작성해주세요.
    [웹 검색 결과]
    {state["search_result"]}

    [AI 분석결과]
    {state["analysis_result"]}]

    검색 결과를 우선적으로 참고하고,
    최신 정보와 기존 분석을 구분해서 설명해줘.
    '''
    response = llm.invoke(prompt)

    return{'messages' : [response]}


# 3. 그래프 구성
builder = StateGraph(State)

builder.add_node('search',search_agent)
builder.add_node('analysis',analysis_agent)
builder.add_node('merge',merge_agent)

###################### 병렬 실행 ######################
builder.add_edge(START,'search')
builder.add_edge(START,'analysis')

###################### 병합 ######################
builder.add_edge('search', 'merge')
builder.add_edge('analysis', 'merge')

builder.add_edge('merge', END)

# 4. 컴파일
graph = builder.compile()

# 5. 실행
query = '엔비디아의 미래 전망에 대해 알려줘.'

response = graph.invoke(
{'messages':[('user', query)]},
)

print('Assistant :', response['messages'][-1].content)



# exit()



###################### 그림 그려 볼까요잇!! ######################
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))

# 그림 출력
plt.figure(figsize=(12,8))
plt.imshow(img)
plt.axis('off')
plt.show()