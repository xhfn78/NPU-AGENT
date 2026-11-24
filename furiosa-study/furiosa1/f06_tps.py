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

import time
from langchain_openai import ChatOpenAI
max_tokens =1000

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1", 
    api_key='EMPTY',
    max_tokens = max_tokens,
    temperature=1,
    top_p=0.5,
)

start = time.perf_counter()
response =llm.invoke('알렉산더 왕(meta)은 어떤사람이야')
end = time.perf_counter()

elapsed = round(end - start,2)

# print(response.content)

print('추론시간 :',elapsed,'초')
print('출력토큰 :',max_tokens)
print('평균 처리량 :',round(max_tokens/elapsed,2),'tokens/s')





