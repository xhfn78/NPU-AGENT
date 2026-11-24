from furiosa_llm import LLM, SamplingParams

#pip install furiosa-llm

with LLM("furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic") as llm:
    params=SamplingParams(max_tokens=1000,top_p=0.3, top_k=100)
#     일반적인 샘플링 구현에서는 다음과 같이 동작해.
# 1. 모델이 다음 토큰의 확률을 계산한다.
# 2. top_k=100으로 확률이 높은 최대 100개를 남긴다.
# 3. top_p=0.3으로 누적 확률이 30% 이상이 되는 최소 후보 집합만 남긴다.
# 4. 남은 후보들의 확률을 다시 정규화하고 하나를 샘플링한다.
# 단, 실제 필터 적용 순서와 경계 처리는 추론 엔진의 구현에 따라 달라질 수 있어.

# 여기서 중요한 점은 top_p=0.3이 토큰을 30% 확률로 선택한다는 의미는 아니라는 것이야.

    message= [{\
        'role':"user",
        'content':'오늘 엔비디아 주가 알려줘'
               }]
    prompt = llm.tokenizer.apply_chat_template(message,tokenize=False)
    response = llm.generate([prompt],params)
    print(response[0].outputs[0].text)