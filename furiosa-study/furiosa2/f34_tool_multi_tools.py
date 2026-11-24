from kingkeras import get_llm
from langchain_core.tools import tool

#1.함수들 정의

@tool
def add(a:int, b:int )-> int :
    '''두 정수를 더합니다.'''

    return a+b


@tool
def multiply(a:int, b:int )-> int :
    '''두 정수를 곱합니다.'''

    return a*b

@tool
def subtract(a:int, b:int )-> int :
    '''두 정수를 뺍니다.'''

    return a - b


#2. 도구 목록

tools =[add,multiply,subtract]  #만든 도구들 tools에 담기
tool_map = {tool.name : tool for tool in tools}
# print(tool_map)
# print(tool_map['add'])

#3 llm+ 도구 연결

llm = get_llm(max_tokens=300)


llm_with_tools = llm.bind_tools(tools)

#4 질문
questions =[
    '15와 27을 더해줘',
    '12와 8을 곱해줘',
    '200에서 150을 빼줘',

]


for question in questions:
    response = llm_with_tools.invoke(question)

    print('\n 질문 :',question)

    for call in response.tool_calls:
        print('선택한 도구 :',call['name'])#llm이 선택한 툴 (함수의이름 나옴)
        print('매개변수:',call['args'])

        result =tool_map[call['name']].invoke(call['args'])
        print('실행결과',result)


'''
# 질문 : 15와 27을 더해줘
선택한 도구 : add
매개변수: {'a': 15, 'b': 27}
실행결과 42

 질문 : 12와 8을 곱해줘
선택한 도구 : multiply
매개변수: {'a': 12, 'b': 8}
실행결과 96

 질문 : 200에서 150을 빼줘
선택한 도구 : subtract
매개변수: {'a': 200, 'b': 150}
실행결과 50
'''



