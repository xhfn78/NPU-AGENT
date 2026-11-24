from kingkeras import get_llm


llm = get_llm(max_tokens=10000,)


prompt ='npu와 gpu차이 설명해줘'

response = llm.invoke(prompt)

print(response.content)