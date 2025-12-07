---
title: "Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation"
author: seokgi
date: 2025-12-07
categories: [Paper]
tags: [MLLM, representation learning , Matryoshka representaion learning, text-text retrieval, text-image retrieval]
pin: true
math: true
---
Written By. Tiansheng Wen, Yifei Wang, Zequn Zeng

# 1. 문제 정의

## 1.1. 배경: Adaptive Embedding이 필요한 이유

정보 검색(search), RAG, 벡터DB에서 임베딩의 길이는 **성능과 비용을 결정하는 핵심 파라미터**다.

- 긴 embedding → 정확도↑, 계산량↑, 비용↑
- 짧은 embedding → 정확도↓, 계산량↓

요즘 상황에 따라

> “embedding 길이를 다양하게 조절(adaptive)하고 싶다”
> 
> 
> 는 요구가 크다.
> 

---

## 1.2. Matryoshka Representation Learning(MRL)의 한계

MRL은 dense embedding의 **앞 N 차원을 쓰는 방식**으로 adaptive length를 구현했다.

그러나 다음과 같은 문제가 있다:

### (문제 1) 짧은 embedding에서는 성능이 급격히 떨어진다

- 8~16차원 같은 짧은 크기에서는 **10% 이상 정확도 하락**
- 구조적으로 “앞부분만 잘라내는 방식”이기 때문에 표현력이 부족

### (문제 2) backbone 전체를 다시 학습해야 한다

- CLIP, NV-Embed 같은 대형 모델을 다시 학습해야 하므로 매우 헤비하다.

### (문제 3) dense 연산의 복잡도 문제

- dense vector의 길이가 줄어도 항상 O(d) 수준의 연산을 수행
- 즉, 엄밀히 말해 계산량 감소는 제한적

---

## 1.3. 핵심 문제 정의

> 사전 학습된 임베딩 모델을 다시 학습시키지 않고,
sparse한 고차원 표현을 이용하여,
짧은 active dimension(K)에서도 높은 정확도를 유지하고,
더욱 빠른 검색을 가능하게 만드는 방법이 필요한가?
> 

이것이 CSR이 해결하려는 문제다.

---

# 2. 방법론

CSR(Contrastive Sparse Representation)은 다음 두 가지를 핵심 아이디어로 한다:

## 2.1: Dense → Sparse(high-dimensional) 변환

> “dense embedding을 길이를 줄이는(truncate) 대신
> 
> 
> 훨씬 큰 공간으로 올린 후(top-heavy) 일부 뉴런만 활성화(Top-K)한다.”
> 

이 아이디어는 sparse coding의 기본 철학과 같다:

- 많은 차원 중 일부만 사용 → 연산량 = O(K)
- h(차원 수)를 크게 만들수록 선택 가능한 feature 종류 증가 → 표현력↑
- K는 작으므로 계산은 매우 빠름

이 구조 덕분에:

- **짧은 실질 embedding(K=8~32)에서도 성능이 거의 유지됨**
- **검색 속도는 K에 비례하여 매우 빨라짐**

---

## 2.2: Sparse Autoencoder + Non-negative Contrastive Loss

CSR은 전체적으로 **SAE + NCL** 로 구성된다.

### 2.2.1. Sparse Autoencoder(SAE)

CSR의 SAE는 다음으로 구성된다:

- **Encoder(1 layer)**
    
    $$
    z = \text{TopK}(\text{ReLU}(W_{enc}(v - b_{pre}) + b_{enc}))
    $$
    
    - v: pretrained dense embedding
    - W_enc: d → h (h=4d가 기본)
    - ReLU: non-negative constraint
    - **TopK: K개만 남기고 0 → sparse**
- **Decoder(1 layer)**
    
    $$
    \hat{v} = W_{dec}z + b_{pre}
    $$
    

---

### 2.2.2 Non-negative Contrastive Loss(NCL):

CSR의 contrastive loss는 다음 식:

$$
L_{cl} = -\log \frac{e^{z_i^T z_i}}{e^{z_i^T z_i} + \sum_{j \ne i} e^{z_i^T z_j}}
$$

여기서 핵심은 **z_i ≥ 0인 sparse 벡터**, 즉 CSR latent의 특성을 활용한다는 점.

- NCL vs InfoNCE 비교 핵심 요약
    - CSR이 NCL을 쓰는 이유는 다음과 같다:
        1. sparse vector에서는 음수 내적이 불가능 → InfoNCE가 잘 작동하지 않음
        2. NCL은 차원 겹침을 줄이는 방향으로 학습 → sparse coding과 매우 잘 맞음
        3. latent dimension이 disentangled되어 표현력이 증가
        4. dead latent 감소 → 더 많은 차원이 의미 있게 사용됨

---

# 3. 최종 Loss: CSR Loss

$$
L_{CSR} = L_{recon} + \gamma L_{cl}
$$

- L_recon = SAE 재구성 보존
- L_cl = NCL을 통한 discriminative alignment
- γ = 1(default)

---

# 4. 실험

CSR은 Vision, Text, Multi-modal 실험에서 MRL을 압도하는 결과를 보여줌.

---

### 4.1. Retrieval Time — 최대 69× 가속

![image.png](https://seokilee0412.github.io/assets/img/BeyoundMRL/image.png)

dense(RN50) 대비 sparse(CSR)는 검색 시간이 크게 줄어든다:

- MRL: dense matmul → O(d)
- CSR: sparse dot-product → O(K)

실험에 따르면:

- **CSR는 69× 이상 빠르게** 1-NN 검색 수행
- h가 커질수록 성능이 올라가면서 속도는 빨라짐 (zero-skipping 효과)

---

### 4.2. Text(MTEB) — 기존 MRL embedding 모델보다 우수

![image.png](https://seokilee0412.github.io/assets/img/BeyoundMRL/image1.png)

### 동일 compute 기준(K=32):

- Jina-V3-64, Nomic-64, Google Gecko-256 등 최신 MRL embedding 모델보다 **약 10~15% 성능 우위**

### 동일 성능 기준:

- NV-Embed-V2와 성능이 비슷한데 속도는 **61× faster**

---

### 4.3. Multimodal(CLIP) — MRL fine-tuning보다 성능 우위

![image.png](https://seokilee0412.github.io/assets/img/BeyoundMRL/image2.png)

CSR은 backbone을 finetune하지 않는데도:

- In-distribution retrieval에서
    - **I2T +4~5%p 향상**
    - **T2I +6~10%p 향상**
- Zero-shot retrieval에서도
    - MRL보다 성능 우세

이는 sparse coding이 multimodal feature alignment에서도 효과적임을 의미.

---

### 4.4. Vision(ImageNet) — Sparse로도 Full-dim 수준 성능

![image.png](https://seokilee0412.github.io/assets/img/BeyoundMRL/image3.png)

- **K=8~32 같은 작은 K에서도 CSR > MRL 성능 차이가 10~20% 수준**
- encoder만 학습했는데도 MRL(full retraining)보다 성능 우위

이는 sparse 구조가 dense truncation보다 표현력이 훨씬 크다는 것을 의미한다.

---

# 5. 결론

CSR이 증명한 것은 다음이다:

### 1) Adaptive representation은 “dense truncation”이 아니라 **sparse selection**이 정답이다

MRL처럼 “embedding을 자른다”는 발상은 원래 정보를 훼손함.

CSR은 반대로, embedding을 더 큰 공간에 올리고 필요한 K개만 선택한다

이 방식이 훨씬 표현력이 크고 정확도가 유지된다.

### 2) Sparse representation은 dense representation보다 빠르고 효율적이다

- zero-skipping 덕분에 연산량 감소
- 실제 GPU 실행 시간도 orders-of-magnitude 수준으로 감소

이는 실무 벡터 검색 시스템에서 매우 큰 이점이다.

### 3) Pretrained embedding을 재학습하지 않고도 MRL보다 성능이 높다

CSR은:

- backbone freeze
- SAE만 학습

코스트 측면에서도 장점을 가진다.

### 4) NCL은 sparse representation의 표현력을 크게 끌어올린다

- dead latent 감소
- feature disentanglement
- sparse 구조와 최적의 조합

InfoNCE보다 훨씬 자연스럽고 효과적이다.