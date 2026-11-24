from kingkeras import run_once, get_cache_metrics

prefix = 'npu의 특징을 설명해줘'*150

for i in range(1,5):
    prompt = prefix + f'\n 질문 {i}:kv cache란?'

    print(f'\n======{i}회 요청 =====')
    result = run_once(prompt,1000)

    before = get_cache_metrics() #요청 전 캐시 상태
    result = run_once(prompt,100)
    after = get_cache_metrics() #요청 후 캐시 상태

    # 이번 요청에서 증가한 횟수
    hits = after['hits'] - before['hits']
    queries = after['queries'] - before['queries']

    if queries > -0:
        hit_rate = hits/queries *100
    else:
        hit_rate = 0

    print(hit_rate)    
    print(f'\n======{i}회 요청 결과  =====')


    print('응답시간 :',round(result['elapsed'],3))
    print('입력 토큰 :',result['input_tokens'])
    print('이번 조회 횟수:',int(queries))
    print('이번 적중 횟수:',int(hits))
    print('이번 적중 적중률:',round(hit_rate,2),'%')

    print('\nPrefix Cache 매트릭')
    get_cache_metrics()

    