'''
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8\
    --devices npu:7\
    --host 0.0.0.0\
    --port 8000 
'''
#동시요청수별 성능 비교

import time
import statistics
from kingkeras import get_llm
from concurrent.futures import ThreadPoolExecutor

llm = get_llm(max_tokens=500)

prompts = [
    f'npu 추론 최적화 방법{i}번을 설명해줘'
    for i in range(20)
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



####################eager모드###########################


# ======Workers:1==================
# 전체 시간: 18.91 초
# 평균 지연: 1.18 초
# 요청 처리량: 0.85 req/s
# 요청 처리량: 0.85 req/s
# output tps 84.624

# ======Workers:2==================
# 전체 시간: 10.43 초
# 평균 지연: 1.3 초
# 요청 처리량: 1.53 req/s
# 요청 처리량: 1.53 req/s
# output tps 153.375

# ======Workers:4==================
# 전체 시간: 5.45 초
# 평균 지연: 1.36 초
# 요청 처리량: 2.94 req/s
# 요청 처리량: 2.94 req/s
# output tps 293.731

# ======Workers:8==================
# 전체 시간: 3.11 초
# 평균 지연: 1.54 초
# 요청 처리량: 5.14 req/s
# 요청 처리량: 5.14 req/s
# output tps 514.481

#################overlab 모드##########################

# ======Workers:1==================
# 전체 시간: 117.58 초
# 평균 지연: 5.88 초
# 요청 처리량: 0.17 req/s
# 요청 처리량: 0.17 req/s
# output tps 85.049

# ======Workers:2==================
# 전체 시간: 61.94 초
# 평균 지연: 6.19 초
# 요청 처리량: 0.32 req/s
# 요청 처리량: 0.32 req/s
# output tps 161.451

# ======Workers:4==================
# 전체 시간: 32.41 초
# 평균 지연: 6.48 초
# 요청 처리량: 0.62 req/s
# 요청 처리량: 0.62 req/s
# output tps 308.52

# ======Workers:8==================
# 전체 시간: 20.95 초
# 평균 지연: 7.07 초
# 요청 처리량: 0.95 req/s
# 요청 처리량: 0.95 req/s
# output tps 477.342


# [--enable-overlap-scheduling | --no-enable-overlap-scheduling]

