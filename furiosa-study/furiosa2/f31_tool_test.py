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
print('일반응답',response.content)
print('========================')
print('tool calls',response.tool_calls)
print('메타데이터 :',response.response_metadata)

'''
{'aaa': {'title': 'Aaa', 'type': 'integer'}, 'bbb': {'title': 'Bbb', 'type': 'integer'}}
일반응답 <think>
Okay, the user wants to add 237341 and 3123321 using the add function. Let me check the function parameters. The function requires two integers, aaa and bbb. I need to make sure both numbers are integers, which they are. So I'll call the add function with aaa as 237341 and bbb as 3123321. The sum should be correct. Let me double-check the numbers to avoid any typos. Yep, that's right. Time to format the tool call.
</think>


========================
tool calls [{'name': 'add', 'args': {'aaa': 237341, 'bbb': 3123321}, 'id': 'call-cc60d9efd0f6483cbb203a6ab8d2f2fe', 'type': 'tool_call'}]
메타데이터 : {'token_usage': {'completion_tokens': 161, 'prompt_tokens': 179, 'total_tokens': 340, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': None, 'cached_tokens': 178}}, 'model_provider': 'openai', 'model_name': 'furiosa-ai/Qwen3-8B-FP8', 'system_fingerprint': None, 'id': 'chat-0811f60b7f4d4e2c836af932d81d7228', 'finish_reason': 'tool_calls', 'logprobs': None}
(furiosa02) root@hancom-furi-agent-20:~/furiosa-study# 
'''