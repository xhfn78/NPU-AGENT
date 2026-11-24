#time to first chunk
from kingkeras import get_llm
import time

llm = get_llm()

prompt = ["npu란?",
          'npu와 gpu의 차이를 5가지 설명해줘'
          'npu의구조와 장점 활용 분야를 자세히 설명해줘'
]

for prompt in prompt:
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed =time.perf_counter()- start

    print('질문:', prompt)
    print('시간:', round(elapsed,2),'초')


