from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
)


# response = llm.invoke('나는 진창훈이야 이해했니?')
response = llm.invoke('내가 누구게?')

print(response.content)