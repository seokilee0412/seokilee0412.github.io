---
title: "Training LLMs to be Better Text Embedders through Bidirectional Reconstruction"
author: seokgi
date: 2025-10-04
categories: [Paper]
tags: [LLM, dense retrieval, contrastive learning]
pin: true
math: true
---
Written By. Chang Su, Dengliang Shi, Siyuan Huang, Jintao Du

## 배경 및 문제점

- 기존의 LLM 기반 임베딩 기법은 주로 **[EOS]와 같은 마지막 토큰의 임베딩을 전체 문장의 대표 벡터**로 사용
- 하지만 이 [EOS] 토큰은 pretraining 중에 **의미 정보를 담도록 학습되지 않기 때문에**, information이나 re-ranking 같은 임베딩 기반 태스크에서 한계가 존재.

---

## 제안 방식: 2단계 학습 프레임워크 (Anchor Embedding)

![image.png](https://seokilee0412.github.io/assets/img/BetterTextEmbedders/image.png)

### Stage 1: **양방향 재구성 (Bidirectional Reconstruction)**

![image.png](https://seokilee0412.github.io/assets/img/BetterTextEmbedders/image1.png)

두 가지 새로운 재구성 태스크를 제안.

### **Stage 1: Bidirectional Reconstruction (양방향 재구성)**

- 목적
    - [EOS] 임베딩이 **문맥과 의미를 잘 요약한 표현이 되도록 훈련**하는 과정.
- 방법
    - Prefix로 $e_Q, e_D$ 를 디코딩 과정에 넣어줌
- 1.1 EBQ2D (Embedding-Based Query-to-Document)
    - **목적**:
        - Query의 [EOS] 임베딩으로 관련 문서를 생성하도록 학습
        - → Query 임베딩이 **의미를 요약**할 뿐 아니라 **연관된 문서의 힌트까지 담도록 강제**함.
    - **구조**
        - Query 시퀀스로부터 임베딩 $e_Q$ 추출
            - $e_X=f(x_1,x_2,...,x_n,[EOS])[−1]$ (equation 1)
        - 이 임베딩을 기반으로 관련 문서 D = {d₁, ..., dₘ}를 생성
        - teacher forcing 방식으로 학습 (예측한 문서와 실제 문서 간 cross-entropy loss 사용)
    - **수식**
        
        $$
        \mathcal{L}_{Q2D} = - \sum_{t=1}^{m} \log P_\Theta(d_t | e_Q, d_{<t})
        $$
        
- 1.2 EBD2Q (Embedding-Based Document-to-Query)
    - **목적**:
        - Document의 [EOS] 임베딩으로 해당 Query를 생성
        - → Document 임베딩이 사용자의 intent를 담아낼 수 있도록 학습
    - **구조**
        - 문서 시퀀스로부터 임베딩 $e_D$ 추출
        - $e_D$를 통해 Query Q = {q₁, ..., qₙ}를 생성
        - 역시 teacher forcing 및 cross-entropy loss 사용
    - **수식**
    
    $$
    \mathcal{L}_{D2Q} = - \sum_{t=1}^{n} \log P_\Theta(q_t | e_D, q_{<t})
    $$
    
- 1.3 최종 손실 함수 (Stage I 전체)
    - 두 손실 함수를 가중 평균하여 학습:
    
    $$
    \mathcal{L}_{\text{Stage I}} = \alpha \mathcal{L}_{Q2D} + (1 - \alpha) \mathcal{L}_{D2Q}, \quad \alpha=0.2
    $$
    
- 기존 LLM은 [EOS]를 그냥 “문장 끝 표시”로만 썼다면,
- 여기서는 [EOS]를 **"생성 시 활용 가능한 정보 요약 역할"로 재학습**시킴
- 결과적으로 **[EOS] = 의미 요약 + 관련성 내포 = 앵커(Anchor)**

---

### **Stage 2: Contrastive Learning**

- 목적
    - Stage 1에서 학습된 [EOS] 임베딩을 바탕으로,
    - **유사한 쌍은 가깝게, 다른 쌍은 멀게** 임베딩 공간을 조정.
- 손실 함수: InfoNCE

### 왜 Stage 1 → Stage 2 순서인가?

- Stage 1: [EOS] 임베딩에 핵심 의미를 더 잘 담도록 만들기
- Stage 2: 그 임베딩을 활용해 유사도 기준으로 정렬하기

---

## 성능 결과 (MTEB 기준)

![image.png](https://seokilee0412.github.io/assets/img/BetterTextEmbedders/image2.png)

![image.png](https://seokilee0412.github.io/assets/img/BetterTextEmbedders/image3.png)

- 특히 **Retrieval (+2.54%)**, Re-ranking (+1.71%)에서 높은 향상을 보여줌.
- 서브 테스크들에서도 전반적으로 **안정적이고 일관된 성능 향상**을 기록.

---

## Ablation Study

![image.png](https://seokilee0412.github.io/assets/img/BetterTextEmbedders/image4.png)

- D2Q, Q2D 각각만 적용해도 성능이 향상되며, **두 태스크를 함께 사용할 때 최상 성능**.

![image.png](https://seokilee0412.github.io/assets/img/BetterTextEmbedders/image5.png)

- α 파라미터(각 손실의 가중치 비율)는 0.2로 설정했을 때 가장 좋은 결과를 얻음.

![image.png](https://seokilee0412.github.io/assets/img/BetterTextEmbedders/image6.png)

- Stage I 적용 모델은 **초반 25~50 step만에 baseline 초월**
- 성능 상승과 수렴 속도 개선 모두 확인됨

![image.png](https://seokilee0412.github.io/assets/img/BetterTextEmbedders/image7.png)

![image.png](https://seokilee0412.github.io/assets/img/BetterTextEmbedders/image8.png)

- 추가 학습량이 아닌 **양방향 reconstruction 학습 자체의 효과**가 성능 개선 원인임을 확인

---

## 한계 및 향후 방향

- **Retrieval/Ranking 태스크에 최적화된 구조**이므로, 분류나 클러스터링에 대한 일반화는 추가 연구 필요.
- 영어 외 다국어 확장도 향후 과제.
- 전체적으로 **2단계 훈련으로 인해 훈련 시간 증가**는 한계 요소.