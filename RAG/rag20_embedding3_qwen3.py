# rag 20-2 카피
# huggingface 임베딩모델 Qwen/Qwen3-Embedding-0.6B 활용해서 vector DB 구성

# LCEL = Langchain Expression Language
# chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()


api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

prompt = "삼성전자의 창업주는 누구인가요?"


# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(
#     model = "text-embedding-3-small",
#     api_key = api_key,
#     base_url= base_url,
# )

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# pip install langchain_huggingface 
# pip install sentence-transformers
from langchain_openai import OpenAIEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name = "Qwen/Qwen3-Embedding-0.6B",  # 크기 조절해서 성능도 조절할 수 있음
    model_kwargs = {
        "device" : "cpu",    # gpu 쓰고 싶으면 "cuda"
        # "local_files_only" : True,  # 처음실행할때 내려받게되고 그이후로는 True옵션 켜놓고 사용하면 됨
    }
)

vector = embeddings.embed_query(prompt)
print(vector)
print("================================")
print("임베딩 벡터의 차원 :", len(vector))  # 임베딩 벡터의 차원 : 1024
# 현재상태에서 벡터는 1개이고 그 벡터의 차원은 1024이다.
 


