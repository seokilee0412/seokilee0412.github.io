---
title: "Differential-informed Sample Selection Accelerates Multimodal Contrastive Learning"
author: seokgi
date: 2025-09-21
categories: [Paper]
tags: [LLM, text-image matching]
pin: true
math: true
---
Written by. Zihua Zhao, Feng Hong, Mengxi Chen

## 논문 개요 및 배경

- 이미지-텍스트 쌍을 위한 **대규모 contrastive learning**은 성능이 뛰어나지만 **학습 비용이 매우 큼**.
- 학습 효율성을 높이기 위한 방법으로 sample selection이 주목받고 있음
- 기존 방식은:
    - **오프라인 방식**: 사전학습 모델을 사용한 coreset selection → cold-start 상황에 적합하지 않음
    - **온라인 방식**: 실시간 모델 예측에 기반하지만 noisy correspondence(잘못된 이미지-텍스트 쌍)를 충분히 고려하지 못함
- 이에 따라 저자들은 **Differential-Informed Sample Selection (DISSect)**를 제안함.

---

## 핵심 아이디어: DISSect

- *현재 모델 예측과 과거(히스토리) 모델 예측의 차이(Δ)**를 활용해 샘플의 품질을 판단.
- **클린한 쌍**은 훈련 초기 높은 유사도를 보이다가 일정 시점 이후 잊히며 점차 CLIPScore가 감소.
- **노이즈 있는 쌍**은 초기엔 낮은 유사도지만 점차 기억되며 CLIPScore가 증가.
- 이 패턴을 기반으로, Δ(CLIPScore_hist - CLIPScore_curr)를 계산해 **노이즈 샘플을 효과적으로 배제**하고 클린 샘플을 우선 학습함.

### DISSect의 핵심 메커니즘

![image.png](https://seokilee0412.github.io/assets/img/Differential/image.png)

### 직관: “모델의 예측 변화량”으로 품질 판단

- 클린 샘플은 초기 학습에서 유사도가 올라간 후 **학습이 완료되어 변화가 작음**
- 노이즈 샘플은 **시간이 지나면서 유사도가 인위적으로 증가함 (과적합)**

이런 "기억되는 정도"를 수치화한 것이 바로 **CLIPScore의 Differential**임:

$$
\Delta = \text{CLIPScore}_{\text{hist}} - \text{CLIPScore}_{\text{curr}}
$$

- `CLIPScore_hist`: 과거 모델(초기 에폭의 모델)의 예측 유사도
- `CLIPScore_curr`: 현재 모델의 예측 유사도

> Δ가 크다 → 이 샘플은 클린인데 현재는 잊혀지고 있음.
> 
> 
> **Δ가 작거나 음수 → noisy하거나 이미 과적합된 샘플**
> 

---

### 실제 사용 방법

두 가지 방식으로 `CLIPScore_hist`를 얻을 수 있음:

1. **Warm-up 기반 방식** (초기 몇 에폭 사용):
    - 일정 warm-up epoch 동안 학습 진행 후 snapshot 저장
    - 이후 각 batch마다 `δ` 계산 후 상위 `r%`만 선택
2. **Momentum 기반 Temporal Ensembling**:
    
    $$
    \text{CLIPScore}_{\text{hist}}^{(t)} = \beta \cdot \text{CLIPScore}_{\text{hist}}^{(t-1)} + (1 - \beta) \cdot \text{CLIPScore}_{\text{curr}}^{(t)}
    $$
    
    - `β`는 모멘텀 계수 (예: 0.9)
    - **비용이 적고** 오라클 모델 없이 실시간 적용 가능

---

### 학습 루프 (Warm-up 버전 기준)

```
1. Warm-up Epoch 동안 전체 데이터 학습 → CLIPScore_hist 저장
2. 이후 Epoch마다:
   a. 각 배치의 현재 CLIPScore_curr 계산
   b. Δ = hist - curr 계산
   c. Δ 값 상위 r% 샘플만 선택해 학습

```

→ 이를 통해 **클린하고 중요한 샘플 위주로 학습 진행**, 노이즈 샘플은 배제됨

---

### 왜 잘 작동하는가?

- **Memorization Theory**
    
    ![image.png](https://seokilee0412.github.io/assets/img/Differential/image1.png)
    
    - 깨끗한 데이터는 먼저 학습되고, 노이즈는 나중에 과적합됨 (이전 논문 인용)
- **Gradient 관점**에서 보면:
    - 클린 샘플의 gradient는 초기 이후 거의 0
    - 노이즈 샘플은 계속 큰 gradient를 유지
- 이 차이를 CLIPScore의 변화량 δ가 **간접적으로 반영**함

$$
\frac{\partial L}{\partial f(I), g(T)} \propto \exp(\text{CLIPScore}) - 1
$$

---

### 어려운 샘플은 그 자체로 학습이 힘들 수 있지 않을까?에 대한 DISSect의 대응 전략:

1. **warm-up 기간 설정**:
    - 초기에 모든 샘플을 균등하게 학습함으로써 “어려운 샘플도 충분히 노출”됨
    - 이후 Δ 계산 시 기준이 보다 안정화됨
2. **Momentum Ensembling 방식**:
    - 단일 시점이 아닌 여러 에폭을 평균낸 결과를 사용 → 일시적인 급등락이 반영되지 않음
3. **우선순위 기반 선택**:
    - 완전히 제거하는 게 아니라 상위 r%만 선택 → 일부 어려운 샘플도 포함될 가능성이 있음

---

## 실험 결과 요약

### 성능 비교 실험 (Retrieval Task)

![CC3M 사전학습 후 평가 (30%, 50%, 70% 선택 비율)](https://seokilee0412.github.io/assets/img/Differential/oimage.png)

CC3M 사전학습 후 평가 (30%, 50%, 70% 선택 비율)

- 평가 지표: Recall@1, @10 for IR/TR
- 대상: MS-COCO, Flickr30K 테스트 데이터셋

주요 결과:

- **DISSect는 CLIPScore, CLIPLoss (oracle 기반)보다 성능이 높음**
- **30% 샘플만으로 full training과 거의 동등한 성능**
- 랜덤, Small Loss, Big Loss 방식보다 확연히 우수

---

### 대규모/노이즈 환경 실험

![image.png](https://seokilee0412.github.io/assets/img/Differential/image2.png)

CC12M, YFCC15M(노이즈 심한 데이터셋) 실험

- **선택 비율: 50%**
- 평가: MS-COCO, Flickr30K

주요 결과:

- YFCC15M에서 DISSect가 **full data보다 성능이 높음**
- 이는 DISSect가 **노이즈 제거에 효과적임**을 시사

---

### 다른 다운스트림 과제

![image.png](https://seokilee0412.github.io/assets/img/Differential/image3.png)

- **BLIP-backbone** 기반으로 실험
- **NLVR2** (시각추론) & **COCO Captioning** 실험

주요 결과:

- DISSect가 SCAN 및 Random 방식보다 더 높은 정확도, BLEU, METEOR, CIDEr 달성
- BLIP 모델에서도 **샘플 선택 효과가 잘 작동함**

---

## Ablation

![image.png](https://seokilee0412.github.io/assets/img/Differential/image4.png)

- 선택된 샘플의 CLIPScore 분포 분석
    - 초기에는 다양한 샘플 선택 → 후반에는 cleaner 샘플만 선택
    - DISSect는 **보다 날카로운(high contrast) 분포** 학습

![image.png](https://seokilee0412.github.io/assets/img/Differential/image5.png)

- 기존 noisy correspondence 제거 방식과 비교
    - GSC, NCR 등 dual-network 기반 방법보다 더 정확하면서 **연산 비용 적음**

![image.png](https://seokilee0412.github.io/assets/img/Differential/image6.png)

- 선택 비율 별 성능 변화
    - **60% 선택 비율에서 최고 성능**
    - 30%에서도 full training에 근접

![image.png](https://seokilee0412.github.io/assets/img/Differential/image7.png)

- 학습 시간 비교 (CC3M, CC12M)
    - DISSect는 **최대 74% 학습 시간 단축** 가능

---

## 결론 및 기여

- **DISSect는 오라클 모델 없이도** 클린한 샘플을 선택하고 학습을 빠르게 할 수 있는 강력한 온라인 샘플 선택 전략.
- 다양한 실험에서 기존 방법보다 **우수한 정확도와 효율성**을 입증.
- 실제 대규모 noisy 데이터셋 (YFCC15M)에서도 매우 좋은 성능을 보임.