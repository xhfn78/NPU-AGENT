from langchain_openai import ChatOpenAI
import os

# os.environ['OPENAI_API_KEY'] = 
llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    openai_api_key= openai_api_key,
)


# response = llm.invoke('나는 진창훈이야 이해했니?')
response = llm.invoke('내가 누구게?')

print(response.content)