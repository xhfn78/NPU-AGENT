from a_kingkeras import get_llm

llm = get_llm()


response =llm.invoke('알렉산더 왕(meta)은 어떤사람이야')
print(response.content)