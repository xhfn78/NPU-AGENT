'''
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8\
    --devices npu:7\
    --host 0.0.0.0\
    --port 8000 \
    --enable-overlap-scheduling
'''

import time
from kingkeras import get_llm

llm = get_llm(max_tokens=1000)

prompts = [
    f"npu 추론 최적화 방법 {i}번을 설명해줘"
    for i in range(8)
]

start = time.perf_counter()
total_tokens = 0 

for prompt in prompts:
    response = llm.invoke(prompt)

    usage = response.usage_metadata or {}
    tokens = usage.get('output_tokens',0)

    total_tokens += tokens
elapsed = time.perf_counter() - start 

print('전체 요청:',len(prompts))
print('전체 출력 토큰',total_tokens)
print('전체 소요 시간:',round(elapsed,3),'초')
print('RPS:',round(len(prompt)/elapsed,3))
print('TPS:', round(total_tokens/elapsed,3))


####################eager모드###########################
##전체 요청: 8
# 전체 출력 토큰 800
# 전체 소요 시간: 9.421 초
# RPS: 2.335
# TPS: 84.915

# 전체 요청: 8
# 전체 출력 토큰 8000
# 전체 소요 시간: 93.55 초
# RPS: 0.235
# TPS: 85.516

#################overlab 모드##########################
# 전체 요청: 8
# 전체 출력 토큰 800
# 전체 소요 시간: 9.84 초
# RPS: 2.236
# TPS: 81.299





# 전체 요청: 8
# 전체 출력 토큰 8000
# 전체 소요 시간: 94.852 초
# RPS: 0.232
# TPS: 84.342


#  [--reasoning-parser {deepseek_r1,deepseek_v3,exaone4,gemma4,qwen3,solar_open}]
    #                          [--enable-responses-api-store]
    #                          [--responses-api-store-max-entries RESPONSES_API_STORE_MAX_ENTRIES]
    #                          [--responses-api-store-ttl RESPONSES_API_STORE_TTL]
    #                          [--structured-outputs-backend {auto,guidance,xgrammar}]
    #                          [--response-role RESPONSE_ROLE] [-pp PIPELINE_PARALLEL_SIZE]
    #                          [-dp DATA_PARALLEL_SIZE] [--max-model-len MAX_MODEL_LEN]
    #                          [--devices DEVICES] [--scheduler-kind SCHEDULER_KIND]
    #                          [--npu-queue-limit NPU_QUEUE_LIMIT]
    #                          [--max-processing-samples MAX_PROCESSING_SAMPLES]
    #                          [--profile-guided-decode-batching | --no-profile-guided-decode-batching]
    #                          [--batch-scheduling-budget-ms <positive-int>]
    #                          [--enable-prefix-caching | --no-enable-prefix-caching]

    #                          [--enable-overlap-scheduling | --no-enable-overlap-scheduling]

    #                          [--data-parallel-routing-policy {scoring,round-robin}]
    #                          [--data-parallel-scoring-profile {balanced,locality,load}]
    #                          [--max-concurrency MAX_CONCURRENCY]
    #                          [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS]
    #                          [--prefix-cache-lookahead-requests PREFIX_CACHE_LOOKAHEAD_REQUESTS]
    #                          [--experimental-enable-npu-sampling] [--max-io-memory-mb MAX_IO_MEMORY_MB]
    #                          [--uvicorn-log-level {debug,info,warning,error,critical,trace}]
    #                          [--disable-uvicorn-access-log | --no-disable-uvicorn-access-log]
    #                          [--allowed-local-media-path ALLOWED_LOCAL_MEDIA_PATH]
    #                          [--image-limit-per-prompt IMAGE_LIMIT_PER_PROMPT]
    #                          [--video-limit-per-prompt VIDEO_LIMIT_PER_PROMPT]
    #                          [--allowed-media-domains ALLOWED_MEDIA_DOMAINS [ALLOWED_MEDIA_DOMAINS ...]]
    #                          [--interleave-mm-strings] [--fxb FXB] [--media-io-kwargs MEDIA_IO_KWARGS]
    #                          [--mm-processor-kwargs MM_PROCESSOR_KWARGS]
    #                          [--mm-processor-cache-gb MM_PROCESSOR_CACHE_GB]