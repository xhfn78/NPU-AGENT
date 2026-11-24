from a_kingkeras import get_llm

llm = get_llm()

for chunk in llm.stream('알렉산더 왕(meta)은 어떤사람이야'):
    print(chunk.content,end='',flush=True)
