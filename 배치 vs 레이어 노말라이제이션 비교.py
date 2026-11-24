
"""
배치 정규화(BN) vs 레이어 정규화(LN) 실습
- 1~4단계: NumPy만으로 직접 계산 (설명에서 본 숫자와 똑같이 나옴)
- 5단계: Keras(TensorFlow)가 설치돼 있으면 BatchNormalization / LayerNormalization 결과와 비교
 
실행: python bn_vs_ln.py
"""
import numpy as np
 
np.set_printoptions(precision=2, suppress=True)
 
EPS = 1e-5  # 0으로 나누는 걸 막는 아주 작은 값 (PyTorch 기본값과 동일)
 
 
def batch_norm(x):
    """세로(열) 방향: 같은 특성을 배치 안의 여러 샘플끼리 비교"""
    mean = x.mean(axis=0, keepdims=True)   # 특성별 평균 (샘플들을 가로질러)
    var = x.var(axis=0, keepdims=True)     # 특성별 분산
    return (x - mean) / np.sqrt(var + EPS), mean, var
 
 
def layer_norm(x):
    """가로(행) 방향: 샘플 하나가 자기 특성들끼리만 비교"""
    mean = x.mean(axis=1, keepdims=True)   # 샘플별 평균 (특성들을 가로질러)
    var = x.var(axis=1, keepdims=True)     # 샘플별 분산
    return (x - mean) / np.sqrt(var + EPS), mean, var
 
 
def title(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)
 
 
# ─────────────────────────────────────────────────────────────
# 1단계: 데이터 준비 (샘플 3개 × 특성 3개)
# ─────────────────────────────────────────────────────────────
x = np.array([
    [1.0, 2.0, 6.0],   # 샘플1
    [3.0, 4.0, 2.0],   # 샘플2
    [5.0, 6.0, 4.0],   # 샘플3
])
title("1단계: 원본 데이터 (행=샘플, 열=특성)")
print(x)
 
# ─────────────────────────────────────────────────────────────
# 2단계: 배치 정규화 - 열(세로)마다 평균/분산
# ─────────────────────────────────────────────────────────────
bn_out, bn_mean, bn_var = batch_norm(x)
title("2단계: 배치 정규화 (열마다 계산)")
print("특성별 평균   :", bn_mean.ravel())
print("특성별 표준편차:", np.sqrt(bn_var).ravel())
print("결과:\n", bn_out)
 
# ─────────────────────────────────────────────────────────────
# 3단계: 레이어 정규화 - 행(가로)마다 평균/분산
# ─────────────────────────────────────────────────────────────
ln_out, ln_mean, ln_var = layer_norm(x)
title("3단계: 레이어 정규화 (행마다 계산)")
print("샘플별 평균   :", ln_mean.ravel())
print("샘플별 표준편차:", np.sqrt(ln_var).ravel())
print("결과:\n", ln_out)
 
# ─────────────────────────────────────────────────────────────
# 4단계: 핵심 실험 - 샘플1의 '배치 친구'를 바꾸면?
# ─────────────────────────────────────────────────────────────
title("4단계: 샘플1은 그대로, 같은 배치에 묶인 샘플만 바꾸기")
batches = {
    "샘플1 + 샘플2 + 샘플3": x,
    "샘플1 + 샘플2      ": x[:2],
    "샘플1 + 엉뚱한 샘플 ": np.array([[1.0, 2.0, 6.0],
                                     [100.0, -50.0, 30.0]]),
}
print(f"{'배치 구성':<22}{'BN 결과(샘플1)':<26}{'LN 결과(샘플1)'}")
for name, b in batches.items():
    bn_s1 = batch_norm(b)[0][0]
    ln_s1 = layer_norm(b)[0][0]
    print(f"{name:<20}{str(bn_s1):<26}{ln_s1}")
print("\n→ BN은 옆에 누가 있느냐에 따라 샘플1의 결과가 계속 바뀐다.")
print("→ LN은 샘플1 혼자 계산하므로 항상 같다.")
 
# ─────────────────────────────────────────────────────────────
# 보너스: 패딩이 섞이면? (문장 길이가 다른 경우)
# ─────────────────────────────────────────────────────────────
title("보너스: 짧은 문장의 빈칸을 0(패딩)으로 채웠을 때")
padded = np.array([
    [1.0, 2.0, 6.0],   # 실제 토큰
    [3.0, 4.0, 2.0],   # 실제 토큰
    [0.0, 0.0, 0.0],   # 패딩 (의미 없는 값)
    [0.0, 0.0, 0.0],   # 패딩
])
print("BN 특성별 평균 (패딩 없음):", batch_norm(x[:2])[1].ravel())
print("BN 특성별 평균 (패딩 섞임):", batch_norm(padded)[1].ravel())
print("→ 의미 없는 0이 평균을 끌어내려 통계가 왜곡된다.")
print("LN 샘플1 결과는 패딩과 무관:", layer_norm(padded)[0][0])
 
# ─────────────────────────────────────────────────────────────
# 5단계: Keras 공식 구현과 비교 (설치돼 있을 때만)
# ─────────────────────────────────────────────────────────────
title("5단계: Keras 구현과 비교")
try:
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow 로그 메시지 숨김
    from tensorflow import keras
 
    t = x.astype("float32")
 
    # center=False, scale=False → 학습되는 β, γ를 끔 (순수 정규화만)
    bn = keras.layers.BatchNormalization(axis=-1, epsilon=EPS,
                                         center=False, scale=False)
    ln = keras.layers.LayerNormalization(axis=-1, epsilon=EPS,
                                         center=False, scale=False)
 
    # training=True → 지금 들어온 배치의 평균/분산으로 정규화
    keras_bn = bn(t, training=True).numpy()
    keras_ln = ln(t).numpy()
 
    print("Keras BatchNormalization:\n", keras_bn)
    print("직접 계산과 일치?", np.allclose(keras_bn, bn_out, atol=1e-4))
    print("Keras LayerNormalization:\n", keras_ln)
    print("직접 계산과 일치?", np.allclose(keras_ln, ln_out, atol=1e-4))
 
    # 배치 크기 1이면? 평균 = 자기 자신 → 분산 0 → 결과가 전부 0
    print("\n배치 크기 1 (문장 하나만 처리하는 상황):")
    print("  BN 결과:", bn(t[:1], training=True).numpy(),
          "← 정보가 다 사라짐")
    print("  LN 결과:", ln(t[:1]).numpy(), "← 그대로 잘 계산됨")
    print("→ 한 문장씩 생성하는 상황에서 LN이 유리한 이유")
except ImportError:
    print("TensorFlow가 없어서 건너뜀 (pip install tensorflow 후 다시 실행해 보세요)")