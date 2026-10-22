from typing import TypedDict
from langgraph.graph import StateGraph,START,END

#1.state 정의 
class State(TypedDict):
    question : str
    search_result : str
    analysist_result : str
    final_result : str

#2 검색노드 

def search_agent(state : State):
    print('검색 에이전트 실행')
    return{'search_result' : '엔비디아는  ai gpu 시장의 주요 기업이다'}


#3 분석노드

def analysist_agent(state : State):
    print('분석 에이전트 실행')
    return{'analysist_result' : 'ai 반도체 수요 증가가 성장요인이다'}

#4 결과 통합 머지노드

def merge_agent(state : State):
    reslut = (
        state['search_result'] + '\n' +
        state['analysist_result']
    )
    return{'final_result' : reslut}


#5. 그래프 구성
builder = StateGraph(State)

builder.add_node("search",search_agent)
builder.add_node('analysist',analysist_agent)
builder.add_node('merge',merge_agent)

###########병령실행####################################
builder.add_edge(START,'search')
builder.add_edge(START,'analysist')
###################################################
##############병합###########################
builder.add_edge('search','merge')
builder.add_edge('analysist','merge')

builder.add_edge('merge',END)

#6.컴파일

graph = builder.compile()


#실행
response = graph.invoke({
    'question': '엔비디아의 미래 전망은'
})


print(response['final_result'])


import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))

# with open(path + 'lg07_ToolNode.png','wb') as file:
#     file.write(png_data)


#그림출력
plt.figure(figsize=(12,8))
plt.imshow(img)
plt.axis('off')
plt.show()
