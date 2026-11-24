'''
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 
furiosa-llm serve furiosa-ai/gpt-oss-20B
furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct<<이건 안먹힘
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic

llm 가동 
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8\
    --devices npu:7\
    --host 0.0.0.0\
    --port 8000    

'''
from langchain_openai import ChatOpenAI
max_tokens =1000
# from openai import OpenAI
# client = OpenAI(
#     base_url="http://127.0.0.1:8000/v1", 
#     api_key='EMPTY',
# )

llm = ChatOpenAI(
     base_url="http://127.0.0.1:8000/v1", 
        api_key='EMPTY',
        max_tokens = max_tokens,
        temperature=1,
        top_p=0.5,
)



#서버에 로드된 모델확인
# model = client.models.list().data[0].id


#추론
# streaming = client.chat.completions.create(
#     model = model,
#     messages=[
#         {
#             'role' : 'user',
#             'content': '알렉산더 왕(meta)은 어떤사람이야'
#         }
#     ],
#     max_tokens=1000,
#     stream=True
# )


# for chunk in streaming:
#     if chunk.choices:
#         content = chunk.choices[0].delta.content
#         if content:
#             print(content,end='',flush=True)
for chunk in llm.stream('알렉산더 왕(meta)은 어떤사람이야'):
    print(chunk.content,end='',flush=True)


