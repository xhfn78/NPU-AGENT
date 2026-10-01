#8-2카피
# chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()


api_key = os.environ['MONOROUTER_API_KEY'].strip() #strip env에 저장한 키값 줄바꾸기 ,공백  인정안하게해줌
prompt = '삼성전자의 창업주는 누구인가요?'
base_url= 'https://monogpt.kr/api/monorouter/v1/'



from langchain_openai import OpenAIEmbeddings
embedding =OpenAIEmbeddings(
    model ='text-embedding-3-small',
    api_key=api_key,
    base_url= base_url,
)

vector = embedding.embed_query(prompt)
print(vector)
print('===============================')
print('임베딩 벡터 차원 :',len(vector)) #1536


