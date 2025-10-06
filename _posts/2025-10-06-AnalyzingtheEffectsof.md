---
title: "Analyzing the Effects of Supervised Fine-Tuning on Model Knowledge from Token and Parameter Levels"
author: seokgi
date: 2025-10-06
categories: [Paper]
tags: [LLM, SFT]
pin: true
math: true
---
Written By. Junjie Ye, Yuming Yang, Yang Nan, Shuo Li, Qi Zhang

## 연구 목적

- LLM은 사전학습(pre-training) 과정에서 대규모 지식을 학습하고, 이후 SFT나 RLHF로 조정됨.
- 하지만 **SFT가 실제로 모델의 지식에 어떤 영향을 미치는지에 대한 체계적인 분석은 부족**함.
- 본 논문은 token-level (로짓 분포 변화)과 parameter-level (모델 파라미터 변화)의 양 측면에서 SFT의 영향을 분석함.

---

## 실험 개요

### 모델

- LLaMA-2 (7B, 13B, 70B) 및 LLaMA-3 (8B, 70B) 모델 사용.

### 데이터

- CBQA(Closed-book QA) 데이터 사용.
    - 질문–정답 쌍(QA pair)으로 구성.
    - 각 주제 10개으로 서브 셋들을 나눔.
- 모델이 얼마나 사전 지식을 알고 있는지를 기반으로 **5단계 mastery level**로 데이터를 나눔.
- 사전학습 모델이 얼마나 잘 답변하는지를 기준으로 각 샘플을 분류함 ($R^k_M$ 스코어).

### 실험 방식

- 서로 다른 mastery level 및 데이터 양 (60~1920 샘플)으로 모델을 SFT 수행.
- 성능은 in-domain 및 out-of-domain QA 정답률로 측정.

---

## 주요 결과

### 현상 1: 더 많은 데이터가 항상 성능 향상을 유도하지 않음

![image.png](https://seokilee0412.github.io/assets/img/AnalyzingtheEffectsof/image.png)

- 240개 샘플까진 성능 향상, 이후 오히려 **성능 저하** 발생.
- 예: LLaMA-3-8B는 240개 학습 시보다 1920개 학습 시 정확도 **최대 14% 감소**.

---

### 현상 2: 학습 데이터의 mastery 수준이 성능에 큰 영향

- “mastery 수준 (knowledge mastery level)”의 정의
    - 모델이 fine-tuning 이전에 그 데이터를 얼마나 “이미 알고 있는지”를 수치화한 값
        - 먼저 사전학습된 LLaMA 모델 MMM이 각 질문 k에 대해 답을 맞히는 확률을 측정.
        - 이 확률을 여러 템플릿(질문 형태 변형)을 통해 평균낸 뒤 모델이 이미 알고 있는 정도 $R_M^k$로 정의
        - 모델이 각 질문에 대해 여러 표현으로 물어봤을 때 정답을 맞힌 비율이 mastery score $R_M^k$.
        
        $$
        R_M^k = \frac{\sum_{i=1}^{N_{map}} \sum_{j=1}^{N_{sample}} I(y_i \subset M_j(x_i))}{N_{map} \times N_{sample}}
        $$
        
        | 기호 | 의미 |
        | --- | --- |
        | $x_i$ | i번째 질문 템플릿으로 작성된 입력 |
        | $y_i$ | 정답 토큰 |
        | $M_j(x_i)$ | j번째 샘플링된 모델 출력 |
        | $I(\cdot)$ | 정답 포함 여부 (맞으면 1, 틀리면 0) |
        | $N_{map}=21$ | 질문 템플릿 수 |
        | $N_{sample}=10$ | 샘플링 횟수 (온도 0.7) |
- Mastery level 구분
    
    ![image.png](https://seokilee0412.github.io/assets/img/AnalyzingtheEffectsof/image1.png)
    
    - 이렇게 분류된 데이터를 knowledge mastery별 SFT dataset으로 사용.
        
        
        | 구분 | mastery level 조건 | 의미 |
        | --- | --- | --- |
        | $D^{M}_{train-0}$ | $R_M^k = 0$ | 모델이 전혀 모르는 데이터 |
        | $D^{M}_{train-1}$ | $0 < R_M^k ≤ 0.25$ | 거의 모름 |
        | $D^{M}_{train-2}$ | $0.25 < R_M^k ≤ 0.50$ | 절반 정도 앎 |
        | $D^{M}_{train-3}$ | $0.5 < R_M^k ≤ 0.75$ | 대부분 앎 |
        | $D^{M}_{train-4}$ | $0.75 < R_M^k ≤ 1.0$ | 이미 거의 완벽히 앎 |

![image.png](https://seokilee0412.github.io/assets/img/AnalyzingtheEffectsof/image2.png)

- 낮은 mastery 수준의 데이터로 SFT를 수행하면 **성능이 더 크게 감소**.
- 고수준 데이터를 사용해도 **낮은 수준 문제에선 성능이 떨어짐**.
- 즉, 중간 수준 mastery 데이터를 활용하는 것이 가장 **균형 잡힌 성능**.

---

## Token-Level 분석 (KL Divergence)

![image.png](https://seokilee0412.github.io/assets/img/AnalyzingtheEffectsof/image3.png)

- KL divergence를 통해 fine-tuned vs. pretrained 모델의 로짓 분포 차이를 측정.
- 60~240개 학습 시 divergence 감소 → 안정적
- 240개 초과하면 divergence 급증 → 성능 저하와 **직접적 상관관계**

---

## Parameter-Level 분석 (파라미터 복원)

![image.png](https://seokilee0412.github.io/assets/img/AnalyzingtheEffectsof/image4.png)

- SFT 과정에서 **전체 파라미터의 90%는 지식 향상에 기여하지 않음**.
    - 전체 파라미터 중 **상위 1%만이 전체 변화량의 약 70%를 차지**
    - SFT 과정에서 극히 일부 파라미터만 강하게 갱신되고, 나머지는 거의 영향이 없음.
    - SFT가 실제로는 모델 전체를 재최적화하지 않고 일부 특정 부분만 과도하게 덮어씀(overwrite)

![image.png](https://seokilee0412.github.io/assets/img/AnalyzingtheEffectsof/image5.png)

- 가장 많이 변한 파라미터를 **순차적으로 원래 값으로 복원** → 성능 향상.

![image.png](https://seokilee0412.github.io/assets/img/AnalyzingtheEffectsof/image6.png)

- 파라미터 복원이 SFT, LoRA보다 높은 성능

![image.png](https://seokilee0412.github.io/assets/img/AnalyzingtheEffectsof/image7.png)

- 어텐션 메커니즘의 역할
    - LLM의 어텐션 메커니즘은 입력 시퀀스 내의 각 토큰이 다른 토큰들과 얼마나 관련이 있는지를 파악하고, 그 관계를 바탕으로 중요한 정보를 추출하고 통합하는 역할
    - Query(Q)는 "내가 찾고 있는 것", Key(K)는 "내가 가지고 있는 정보", Value(V)는 "실제 정보 내용"
    - 낮은 업데이트 비율의 의미
        - 이 레이어들에서 불필요한 파라미터 업데이트 비율이 낮다는 것은, 사전 훈련 단계에서 이미 광범위한 세계 지식을 학습하면서 구축된 지식 검색(retrieval) 및 정보 통합(integration)의 핵심 메커니즘이 파인튜닝 과정에서 비교적 안정적으로 유지된다는 것을 의미.
        - **모델이 정보를 '어떻게' 찾아내고 '어떻게' 연결할지에 대한 기본적인 능력은 SFT를 통해 크게 흔들리거나 비효율적으로 변경되지 않는다는 것.**
- 위의 불필요한 파라미터 간의 어텐션과 FFN 레이어 간의 차이는 LLM 내에서 지식의 유형별 저장 및 처리에 대한 일종의 '분업'이 존재할 수 있음을 시사
    - 어텐션 레이어 (Query, Key, Value)
        - 입력 토큰들 간의 관계 학습 및 관련성 판단에 특화되어 있으며, 이는 지식의 구조나 접근 방식과 더 밀접하게 관련 프리 트레이닝을 통해 이미 전반적인 언어 구조와 의미론적 관계를 잘 학습했기 때문에, 특정 CBQA 작업에 대한 파인튜닝 시에도 그 본질적인 역할은 크게 변할 필요가 없었을 가능성
    - FFN 레이어
        - 어텐션에서 통합된 정보를 바탕으로 구체적인 사실적 지식(factual knowledge)을 처리하거나 새로운 추론을 수행하는 역할을 담당.
        - SFT는 모델이 특정 형식의 질문에 특정 방식으로 답변하도록 조절하므로, FFN에 저장된 지식의 '표현'이나 '활성화' 방식에 더 많은 변화를 유도
        - 이러한 변화가 항상 지식 강화에 긍정적인 영향을 미치지 않아 불필요한 업데이트가 많이 발생.
- SFT는 모델에게 새로운 사실적 지식을 직접 '주입'하기보다는, 기존에 사전 훈련된 모델이 가지고 있는 방대한 지식을 특정 작업에 맞게 '활용'하고 '표현'하는 방법을 가르치는 데 더 중점
- 따라서 어텐션 메커니즘은 지식에서 필요한 정보를 찾는 '도구'의 역할을 하고, FFN은 건져 올린 정보를 특정 목적에 맞게 가공하는 역할.
- 이는 SFT 전략을 설계할 때, 어텐션과 같은 핵심 지식 접근 메커니즘은 보존하면서 FFN을 통한 지식 처리 및 표현 방식을 효율적으로 조정하는 데 초점을 맞춰야 함을 시사.

---

## 결론

- SFT는 무분별한 파라미터 업데이트를 유발하여 **지식 손실** 및 **성능 저하**를 유도할 수 있음.
- 향후 연구는 “불필요한 파라미터 업데이트를 억제하는 fine-tuning 전략” 개발 필요.
- 파라미터 복원 방식은 단순하면서도 효과적인 보정 방법.