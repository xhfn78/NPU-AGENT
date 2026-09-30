# chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

prompt = PromptTemplate.from_template('{topic}에 대해 쉽게 설명해주세요.')

api_key = os.environ['MONOROUTER_API_KEY'].strip() #strip env에 저장한 키값 줄바꾸기 ,공백  인정안하게해줌
base_url= 'https://monogpt.kr/api/monorouter/v1/'

model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url= 'https://monogpt.kr/api/monorouter/v1/'
)

chain = prompt | model

input = {'topic' : '양자컴퓨터 학습 원리'}

response = chain.invoke(input)
print(response.content)
