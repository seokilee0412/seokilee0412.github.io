---
title: "Randomly Removing 50% of Dimensions in Text Embeddings has Minimal Impact on Retrieval and Classification Tasks"
author: seokgi
date: 2025-10-22
categories: [Paper]
tags: [LLM, retrieval]
pin: true
math: true
---
EMNLP Oral
Written By. Sotaro Takeshita, Yurina Takeshita, Daniel Ruffinelli, Simone Paolo Ponzetto

## 1. 연구 개요

- **텍스트 임베딩의 차원을 무작위로 제거했을 때** 다운스트림 작업(검색·분류·생성)의 성능에 어떤 영향을 미치는지를 체계적으로 분석

### 주요 관찰 결과

![image.png](https://seokilee0412.github.io/assets/img/RandomlyRemoving50/image.png)

1. **연구 목표**
    
    텍스트 임베딩의 차원 축소가 검색 및 분류 성능에 미치는 영향을 실험적으로 평가.
    
2. **핵심 발견**
    
    6개의 최신 텍스트 인코더와 26개의 다운스트림 작업에서,
    
    **임베딩 차원의 최대 50%를 무작위로 제거해도 성능 하락이 10% 미만**으로 유지됨.
    
3. **성능 유지 수준**
    
    절반 이상 차원 제거 후에도
    
    - 분류 작업: **원본 대비 95% 성능 유지**
    - 검색 작업: **원본 대비 90% 성능 유지**

---

## 2. 기존 이론 검토 및 새로운 관점 제시

### 2.1. 기존 이론 검토

기존 연구의 세 가지 대표적 가설을 실험적으로 검증:

1. **Anisotropy (비등방성)** — 임베딩이 좁은 원뿔 공간에 몰림
2. **Redundancy (중복성)** — 차원 간 정보 중복
3. **Outlier Dimensions (이상치 차원)** — 특정 차원이 성능 결정

→ 세 현상 모두 관찰되지만, **무작위 제거 시 성능 유지와 직접적인 상관관계는 없음**.

### 2.2. Degrading Dimensions

- **입력 기여도(Attribution) 분석** 방식을 활용해 각 차원의 성능 기여도를 독립적으로 평가.
- **결과:** 모든 모델에서 성능에 부정적 영향을 미치는 다수의 차원이 존재함.
    - E5-large: **1024개 중 430개(약 42%)가 성능 저하 차원**
- **특징:** 저하 차원은 임베딩 전역에 균일하게 분포되어 있으며,
    
    무작위로 차원을 제거할 때 긍정·부정 차원이 함께 제거되어 **전반적 성능 하락이 작게 나타남.**
    
- **효과:** 저하 차원만 제거하면 성능이 유지되거나 오히려 향상되는 경우도 있음.
- **공유성:** 여러 다운스트림 작업 간에 공통적으로 존재하는 저하 차원이 확인됨.

---

## 3. 실험 설정 및 주요 결과

### 3.1. 모델 및 데이터셋

- **모델:** MPNet, Contriever, E5 (Large), E5-Mistral, Paraphrase-MiniLM, Sentence-T5 6종
- **데이터셋:**
    - **검색:** BEIR (14개)
    - **분류:** MTEB (12개)

### 3.2. 차원 축소 방식

1. **Last K% Truncation:** 마지막 K% 차원 제거
2. **Random K% Truncation:** 무작위 K% 차원 제거

### 3.3. 결과

![image.png](https://seokilee0412.github.io/assets/img/RandomlyRemoving50/image1.png)

- **Last K% 제거:**
    - 대부분 모델이 **80% 제거 시에도 성능 80% 이상 유지**
    - E5-Mistral은 **90% 차원 제거 시 90% 성능 유지**
- **Random K% 제거:**
    - 결과 패턴이 유사하며, **10회 반복 간 표준편차가 작음** → 어떤 차원을 제거해도 성능 유지됨.

---

## 4. 표현 공간 사용 효율성 분석

![image.png](https://seokilee0412.github.io/assets/img/RandomlyRemoving50/image2.png)

### 4.1. Anisotropy 분석

- **실험:** BERT, T5 모델을 대조 학습으로 훈련하며 anisotropy 변화 추적
- **지표:** Uniform Loss(↓), IsoScore(↑)
- Anisotropy (비등방성) 확인 방법
    - **실험 설계**
        - **BERT**와 **T5** 모델을 **contrastive learning 방식으로 학습**하면서, 각 학습 단계에서 임베딩의 anisotropy 정도와 성능을 함께 측정.
        - **측정 지표**:
            - **Uniform Loss** (↓일수록 임베딩이 균일하게 분포)
                - 개념적 정의
                    - 임베딩이 representation space에 얼마나 **균일하게 분포되어 있는가**를 측정.
                    - 값이 **작을수록 균일하게 퍼져 있음 (덜 anisotropic)**
                    - 값이 **클수록 특정 방향에 몰려 있음 (anisotropic)**
                - 수학적 정의
                    - 임베딩 벡터들의 L2 정규화된 집합을 ${z_i}_{i=1}^N$라 할 때,
                    - 식:

                    $$
                    \mathcal{L}_{uniform} = \log \mathbb{E}_{i \neq j} \left[ e^{-2 \| z_i - z_j \|^2} \right]
                    $$
                    
                    - $\|z_i - z_j \|^2$: 임베딩 간 유클리드 거리 제곱
                    - $e^{-2 \| z_i - z_j \|^2}$: 서로 가까울수록 큰 값 (즉, 군집화 정도)
            - **IsoScore** (↑일수록 고르게 분포)
                - 개념적 정의
                    - 임베딩 공간이 **모든 방향에 걸쳐 고르게 분산되어 있는지**를 측정.
                    - 즉, 임베딩 벡터들의 공분산 행렬의 고유값(eigenvalues)이 얼마나 균일한가를 보는 척도.
                - 수학적 정의
                1. 임베딩 벡터 집합의 공분산 행렬 $C = \text{Cov}(Z)$계산
                2. 고유값 $\lambda_1, \lambda_2, \dots, \lambda_d$ 추출
                3. 이를 정규화한 후, **평균 분산의 균등성**을 평가: $\text{IsoScore} = 1 - \frac{\text{Var}(\lambda)}{\text{Var}(\lambda_{uniform})}$
                - $\text{Var}(\lambda)$: 실제 고유값들의 분산
                - $\text{Var}(\lambda_{uniform})$: 이론적 최대 분산, $Var(λ_{uniform}​)=\frac{1}{d1}​(1−\frac{1}{d1}​)$
                
                결과적으로:
                
                - **IsoScore → 1:** 완전 균일 (등방적, 덜 anisotropic)
                - **IsoScore → 0:** 특정 방향에 몰림 (anisotropy 강함)
    - **결과**
        - Full embedding 성능은 향상, anisotropy는 감소 → 기대한 대로
        - **그러나**: 50% 차원 축소된 embedding의 상대 성능은 **거의 일정**
            
            → 즉, 덜 anisotropic하다고 해도 차원 축소 시 성능이 떨어지지 않음
            
- 결론: **Anisotropy와 성능 유지 간에 상관관계 없음**

### 4.2. Dimensional Collapse 분석

- **지표:** 차원 간 평균 상관계수(Corr Mean)
- Dimensional Collapse (차원 붕괴) 확인 방법
    - **실험 설계**
        - 동일하게 BERT와 T5의 contrastive 학습 중 **각 차원 간 상관관계(평균)** 측정
        - 상관관계가 높을수록 차원 붕괴 가능성이 높음
        - 측정 지표: **Corr Mean**
            - 개념적 정의
                - Dimensional Collapse(차원 붕괴)의 측정 지표.
                - 즉, 임베딩의 각 차원들이 서로 **얼마나 상관되어 있는지**의 평균값.
                    - **Corr Mean이 높다:**
                        - 차원 간 상관성이 높아 정보가 중복 → 일부 차원만 사용 (collapse)
                    - **Corr Mean이 낮다:**
                        - 차원들이 독립적으로 정보 표현 → 더 효율적 사용
            - 수학적 정의
            1. 임베딩 행렬 $Z \in \mathbb{R}^{N \times d}$ 의 각 차원을 정규화
            2. Pearson 상관계수를 $\rho_{i,j} = \frac{\text{Cov}(Z_i, Z_j)}{\sigma_{Z_i} \sigma_{Z_j}}$ 차원 간 쌍으로 계산
            3. 전체 상관계수 평균을 구함: $\text{Corr Mean} = \frac{1}{d(d-1)} \sum_{i \neq j} |{\rho_{i,j}}|$
    - **결과**
        - 학습 진행되면서 Corr Mean은 감소 → 차원 붕괴 완화
        - 그러나 50% 차원 제거 후 **성능 유지 비율은 변화 없음**
- 결론: **차원 붕괴와 성능 유지 간에도 유의미한 상관관계 없음**

### 4.3. Outlier Dimensions 분석

![image.png](https://seokilee0412.github.io/assets/img/RandomlyRemoving50/image3.png)

- **정의:** 평균 임베딩 기준 ±3σ 벗어난 차원
- Outlier Dimensions (이상값 차원) 검토 방법
    - **실험 설계**
        - NanoBEIR 데이터셋의 평균 임베딩에서 **표준편차 기준으로 이상값(±3σ 이상)** 차원들을 추출
        - 이들 차원을 제거한 결과와, 동일 개수의 무작위 차원을 제거한 결과를 비교
    - **결과**
        - 대부분 모델에서 **이상값 차원 제거 시 성능 변화 거의 없음**
        - 일부 모델(E5-Mistral)에서는 제거 후 약간 성능 향상되었으나 미미함 (nDCG@10에서 0.025)
- 결론: **이상값 차원은 소수이고, 성능 유지의 주된 원인이 아님**

---

## 5. 차원 기여도 분석 (Dimension Attribution Analysis)

![image.png](https://seokilee0412.github.io/assets/img/RandomlyRemoving50/image4.png)

### 5.1. 방법

- 각 차원을 **하나씩 독립적으로 제거**하면서 다운스트림 성능 변화를 측정.
- 제거 시 성능이 향상 → 저하 차원
- 성능이 하락 → 기여 차원

### 5.2. 결과

- 모델 대부분에서 절반 이상이 **저하 차원**으로 판별됨.
- **저하 차원만 제거 시:** 성능 향상 또는 완만한 감소.
- **기여 차원만 제거 시:** 성능 급격히 하락.
- **분포:** 저하 차원은 특정 영역에 몰리지 않고 전역적으로 분포.

### 5.3. PCA와의 비교

![image.png](https://seokilee0412.github.io/assets/img/RandomlyRemoving50/image5.png)

- 임베딩 절반 축소 시 **Random Truncation vs PCA** 비교.
- **결과:** 무작위 절단은 학습·추론 비용 없이 PCA와 유사하거나 더 나은 성능.
    - 예: E5-Mistral(MTEB) → Random: 99.6%, PCA: 100.4%
        
        → **무작위 절단은 단순하고 비용 효율적인 대안**.
        

---

## 6. Causal LLM 실험

![image.png](https://seokilee0412.github.io/assets/img/RandomlyRemoving50/image6.png)

### 실험 개요

- **모델:** Llama 3.1 8B, Qwen 2.5 7B
- **방법:** 마지막 은닉 표현의 절반(첫 절반 or 마지막 절반) 제거 후 평가
- **테스트 세트:** MMLU, SQuAD-v2, GSM8K 등 6개

### 결과

- 6개 중 3개 작업에서 **원본 성능의 80% 이상 유지**
- 제거 방식(앞/뒤 절반)에 따른 차이 거의 없음 → 비효율적 표현 공간 사용 추정
- 단, **GSM8K 등 수리적 추론 과제에서는 성능 급락** → LLM에서는 과제별 민감도가 큼.

---

## 7. 결론 및 한계점

### 7.1. 결론

1. **핵심 발견:**
    
    텍스트 임베딩의 절반을 무작위로 제거해도 90% 이상 성능 유지.
    
2. **원인:**
    
    임베딩 내부에는 성능 저하를 유발하는 **다수의 저하 차원**이 존재하며,
    
    이들이 무작위로 함께 제거되면 성능 하락이 미미하게 나타남.
    
3. **적용 확장:**
    
    일부 LLM에서도 유사한 경향이 관찰됨.
    

### 7.2. 한계점

1. **저하 차원 발생 원인 미상** — 어느 훈련 단계에서 생기는지 불명.
2. **단일 차원 분석 한계** — 다차원 상호작용 미고려.
3. **언어적 제한** — 영어 데이터만 실험.
4. **MRL 모델과의 비교 부재** — 예비 실험에서는 유사 성능이었으나 대규모 실험 미실시.