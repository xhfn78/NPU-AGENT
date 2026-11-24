#동시요청수별 성능 비교

import time
import statistics
from kingkeras import get_llm
from concurrent.futures import ThreadPoolExecutor

llm = get_llm(max_tokens=100)

prompts = [
    f'npu 추론 최적화 방법{i}번을 설명해줘'
    for i in range(16)
]

def ask(prompt):
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start
    usage = response.usage_metadata or {}
    tokens =usage.get('output_tokens',0)

    return elapsed,tokens

for workers in [1,2,4,8]:

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as excutor:
        results = list(excutor.map(ask,prompts))
    total_time = time.perf_counter() - start

    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)

    print(f'\n======Workers:{workers}==================')

    print('전체 시간:', round(total_time,2),'초')
    print('평균 지연:', round(statistics.mean(latencies),2),'초')
    print('요청 처리량:', round(len(results)/total_time,2),'req/s')
    print('요청 처리량:', round(len(results)/total_time,2),'req/s')
    print('output tps',round(total_tokens/total_time,3))
print()    


# 전체 출력 토큰: 1600
# 전체 시간: 5.395 초
# output TPS: 296.552 초
# RPS 2.966 req/s