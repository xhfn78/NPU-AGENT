from kingkeras import get_llm
from langchain_core.tools import tool

#데코레이터: 기존의 함수에 기능을 더해서 다시 만들겠다
#한마디로 함수를 불러오는 함수
@tool
def add(aaa:int, bbb:int)-> int:
    '''두  정수를 더합니다'''

    return aaa + bbb
print(add.name)
print(add.description)
print(add.args)

#2. llm 생성
llm = get_llm(max_tokens=1000)

#3.도구 등록
llm_with_tools = llm.bind_tools([add])

#4.질문
response =llm_with_tools.invoke(
    '237341과 3123321을 add로 더해줘'
)


#5.결과확인 
for call in response.tool_calls:
    print('tool 이름:',call ['name'])
    print('매개 변수:',call['args'])
    print('호출ID:',call['id'])
    print('tpye:',call['type'])

# add
# 두  정수를 더합니다
# {'aaa': {'title': 'Aaa', 'type': 'integer'}, 'bbb': {'title': 'Bbb', 'type': 'integer'}}
# tool 이름: add
# 매개 변수: {'aaa': 237341, 'bbb': 3123321}
# 호출ID: call-259be955608f4bb78cee8b5352424f2b
# tpye: tool_call