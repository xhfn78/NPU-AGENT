from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()


api_key = os.environ['MONOROUTER_API_KEY'].strip() #strip env에 저장한 키값 줄바꾸기 ,공백  인정안하게해줌
base_url= 'https://monogpt.kr/api/monorouter/v1/'

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url= 'https://monogpt.kr/api/monorouter/v1/'
)


# response = llm.invoke('나는 진창훈이야 이해했니?')
response = llm.invoke('내가 누구게?')

print(response.content)