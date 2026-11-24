from kingkeras import run_once, get_cache_metrics

prefix = 'npu의 특징을 설명해줘'*150

for i in range(1,5):
    prompt = prefix + f'\n 질문 {i}:kv cache란?'

    print(f'\n======{i}회 요청 =====')
    result = run_once(prompt,1000)

    print('응답시간 :',round(result['elapsed'],3))
    print('입력 토큰 :',result['input_tokens'])

    print('\nPrefix Cache 매트릭')
    get_cache_metrics()

    