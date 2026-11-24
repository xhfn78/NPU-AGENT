import time
import statistics
from concurrent.futures import ThreadPoolExecutor

from kingkeras import run_once

prompts =[
    f'npu 추론 최적화 방법 {i}번을 자세히 설명해줘.'
    for i in range(8)
]

# print(prompts)
# print(len(prompts)) #8

# aaa = run_once(prompt='npu가 뭐게?',max_tokens=500)

# print(aaa['text'])

# def ask(prompt):
#     return run_once(prompt,500)

for workers in [1,2,4,8]:
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(
                lambda prompts: run_once(prompts,50),
                prompts)  #프롬프트의 8개의 값을 하나씩 ask에 넣겠다 map 함수 동작원리
        )
    elapsed =time.perf_counter() - start

    total_tokens =sum(
        r['output_tokens'] for r in results
    )

       

    avg_latency =statistics.mean(
        r['elapsed'] for r in results
    )

    print(f'\n동시 요청 수 :{workers}')
    print(f'전체시간 :{elapsed:.2f}초')
    print(f'전체 출력 토큰 :{total_tokens}')
    print(f'처리량 :{total_tokens/elapsed}tps')
    print(f'평균 요청 지연 :{avg_latency:.2f}초')


# 이 코드는 NPU 추론 서버에 요청을 여러 개 동시에 보냈을 때 처리 성능이 어떻게 달라지는지 측정하는 실험이에요. 즉,
# 연속 배칭(continuous batching)을 직접 구현하는 코드는 아니고, 동시 요청 수에 따른 서버 성능을 비교하는 코드입니다.