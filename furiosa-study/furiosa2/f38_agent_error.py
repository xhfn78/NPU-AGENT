from kingkeras import get_llm
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

#pip install langchain

@tool
def add(a:int, b:int) ->int:
    '''두 정수를 더합니다.'''
    return a+b

@tool
def subtract(a:int, b:int )-> int :
    '''첫번째에서 두번째 정수를 뺍니다.'''

    return a-b


@tool
def multiply(a:int, b:int )-> int :
    '''두 정수를 곱합니다.'''

    return a*b
@tool
def divide(a:int, b:int )-> float :
    '''두 정수를 나눕니다.'''

    return a / b

llm = get_llm(max_tokens=500)

tools =[add,multiply,divide,subtract]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        '계산이 필요하면 도구를 사용하세요.'
        '도구 실행에 실패하면 오류를 설명하세요.'
    )
    #에이전트한테 역활과 행동지침을 설정.
)

questions ='10을 0으로 나눠줘'


# for question in questions:
#     print('\n질문:',question)

try:
    result = agent.invoke(
        {
        'messages':[
            {'role':'user','content': questions}
        ]
    },config={'recursion_limit':20}
    )

    for i,msg in enumerate(result['messages']):
        print(f'{i+1}번째 메세지')
        print('타입:',type(msg).__name__)
        print('내용:',msg.content)

        if isinstance(msg,AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print('도구 이름:',call['name'])
                print('전달 인자:',call['args'])

    print('=================================')
    print('최종답변:',result['messages'][-1].content)

except ValueError as e:
    print('계산 오류:',e)

except Exception as e:
    print('agent 실행 오류:' ,type(e).__name__, e)