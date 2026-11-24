from kingkeras import get_llm
llm = get_llm(max_tokens=1000)

response = llm.invoke('npu란 무엇인가')
print(response.content)
print('=============================')
print(response.usage_metadata)