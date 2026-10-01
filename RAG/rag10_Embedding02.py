#10-1카피
# chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()


api_key = os.environ['MONOROUTER_API_KEY'].strip() #strip env에 저장한 키값 줄바꾸기 ,공백  인정안하게해줌
prompt = 'Salaan, way fiican tahay inaan kula kulmo.?'
base_url= 'https://monogpt.kr/api/monorouter/v1/'



from langchain_openai import OpenAIEmbeddings
embedding =OpenAIEmbeddings(
    model ='text-embedding-3-large', #3072 차원갯수
    api_key=api_key,
    base_url= base_url,
    # dimensions=5,
)

vector = embedding.embed_query(prompt)
print(vector)
print('===============================')
print('임베딩 벡터 차원 :',len(vector)) #1536 차원갯수 


