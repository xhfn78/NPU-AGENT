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
from openai import OpenAI
from langchain_openai import ChatOpenAI
# client = OpenAI(
#     base_url="http://127.0.0.1:8000/v1", 
#     api_key='EMPTY',
# )




# #서버에 로드된 모델확인
# model = client.models.list()
# model_id = model.data[0].id

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1", 
    api_key='EMPTY',
)


# #추론
# response = client.chat.completions.create(
#     model = model_id,
#     messages=[
#         {
#             'role' : 'user',
#             'content': '알렉산더 왕(meta)은 어떤사람이야'
#         }
#     ],
#     max_tokens=10000,
# )

# print(response.choices[0].message.content)

response =llm.invoke('알렉산더 왕(meta)은 어떤사람이야')
print(response.content)


