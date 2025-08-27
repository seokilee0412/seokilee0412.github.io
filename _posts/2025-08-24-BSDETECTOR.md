---
title: "Quantifying Uncertainty in Answers from Any Language Model and Enhancing Their Trustworthiness"
author: seokgi
date: 2025-08-24
categories: [Paper]
tags: [LLM, haullcination]
pin: true
math: true
---
Written by. Jiuhai Chen, Jonas Mueller

## 배경 및 문제의식

- LLM은 강력한 성능을 보이지만, 여전히 **신뢰성 부족**으로 인해 고위험(high-stakes) 분야에 사용하기 힘듬.
- 특히 **hallucination** 문제로 인해 그럴듯하지만 **틀린 답변**을 자주 생성함.
- 기존 불확실성 추정(uncertainty estimation) 기법은 대부분 모델의 내부 접근(token-level likelihood, training data 등)을 필요로 하는데, **GPT 계열 모델은 블랙박스 API**이므로 적용할 수 없음.

---

## Related Work

- **Supervised Learning에서의 불확실성 추정**: Bayesian dropout, Deep ensemble, Calibration 기법 등. → 하지만 LLM 블랙박스에는 적용 불가.
- **LLM 불확실성 추정 관련 연구**:
    - Semantic entropy: token probability 필요 → 블랙박스에 제한.
    - Self-evaluation prompting: 모델에게 직접 확률을 묻는 방식.
    - Sampling 기반 hallucination 탐지(SelfCheckGPT).
- **본 논문의 차별점**:
    - **추가 학습 불필요**, **token probability 접근 불필요**, **블랙박스 모델 적용 가능**.

---

## 제안된 기법: **BSDETECTOR**

### 목표

![image.png](https://seokilee0412.github.io/assets/img/BSDETECTOR/image.png)

- 어떤 블랙박스 LLM이든 응답에 대해 **수치화된 신뢰도(confidence score)** 를 부여.
- 사용자가 응답을 맹신하지 않고, **‘믿을 만한지 아닌지’** 판단할 수 있게 함.

### 구성 요소

BSDETECTOR는 **2가지 핵심 컴포넌트**를 조합하여 confidence score를 산출:

![image.png](https://seokilee0412.github.io/assets/img/BSDETECTOR/image1.png)

### Observed Consistency (외재적 평가)

- **아이디어**: 신뢰할 수 있는 답변이라면 여러 번 생성해도 크게 달라지지 않는다.
- **방법**:
    1. LLM에 동일 질문을 여러 번 던져 {y₁, y₂, ..., yₖ} 생성 (k=5 사용).
    2. 답변 간 의미적 유사도를 측정 (Natural Language Inference, NLI 활용 → entailment/neutral/contradiction).
    3. NLI 한계(특히 단답형 문제) 보완 위해 단순 indicator function(y=yi 여부)도 결합.
        - 논문에서는 **Observed Consistency**(응답 일관성 평가)를 위해 LLM이 생성한 여러 답변을 비교할 때, **NLI(Natural Language Inference) 모델**을 사용합니다.
        - NLI 모델은 두 문장을 보고 **entailment(함의), neutral(중립), contradiction(모순)**을 판정.
        - 예:
            - “The pancreas produces insulin.”
            - “Insulin is produced by the pancreas.”
                
                → entailment (의미적으로 동일).
                
        
        그런데 단답형 문제에서는 문제가 생김.
        
        예를 들어, 수학 문제에서 정답이 “20”인데 모델이 응답으로
        
        - y = "20"
        - yi = "24"
        
        NLI 모델은 이런 **숫자-숫자 비교**에는 학습이 충분치 않아서, **정확히 모순을 잘 잡아내지 못하는 경우**가 많음
        
        → “20”과 “24”의 관계를 entailment/contradiction으로 안정적으로 분류하지 못할 수 있음.
        
        그래서 저자들은 **보조 지표**로 단순한 **indicator function**을 추가.
        
        - 정의:
            
            $$
            r_i=1[y=y_i]
            $$
            
            - (즉, 원래 답변 y와 비교 답변 yi가 **완전히 동일하면 1, 다르면 0**)
        - 최종 similarity 점수 계산식:
            
            $$
            o_i=αs_i+(1−α)r_i
            $$
            
        
        여기서
        
        - s_i = NLI 기반 유사도 (1-contradiction 확률)
        - r_i = indicator function (정확히 같은지 여부)
        - α = 0.8 (실험에서 고정)
        
        즉, NLI의 불안정성을 보완하기 위해 **정확히 같은 답이면 확실히 신뢰도를 높여주는 보정 역할**.
        
    4. 최종 similarity score 계산 후 평균 → Observed Consistency Score O 도출.

### Self-reflection Certainty (내재적 평가)

- **아이디어**: LLM에게 자신의 답이 맞는지 스스로 평가하게 하자.
- **방법**:
    - 후속 질문 프롬프트: "이 답변은 맞습니까?"
    - 선택지: A) Correct, B) Incorrect, C) I am not sure.
    - 점수 부여: Correct=1.0, Incorrect=0.0, Not sure=0.5
    - 여러 번 반복해 평균 → Self-reflection Certainty S 도출.
- **중요한 점**: token probability 사용 X, 오직 프롬프트 기반 자기반성.

### **최종 Confidence Score**

$$
C=βO+(1−β)S
$$

- O: Observed Consistency
- S: Self-reflection Certainty
- β는 실험에서 0.7로 고정

---

## Applications

### 더 신뢰할 수 있는 답변 생성

- 동일 질문에 대해 여러 응답을 생성 후, **confidence score가 가장 높은 것 선택**.
- 결과적으로 **기존 응답보다 더 정확한 답변**을 자동 선택 가능.

### 자동 평가(Auto-eval) 신뢰성 향상

- GPT-4 같은 모델이 다른 답변을 평가할 때조차 불확실성이 존재.
- GPT-4 평가에도 BSDETECTOR를 적용하여,
    - confidence가 낮은 평가 결과는 **human-in-the-loop**으로 보완.
    - fully-automated 평가에서는 confidence 낮은 평가를 제거.

---

## 실험

### 데이터셋

- GSM8K (수학 단어문제)
- SVAMP (수학 문제 변형)
- CSQA (Commonsense QA)
- TriviaQA

### 실험 모델

- OpenAI Text-Davinci-003
- GPT-3.5 Turbo

### 주요 결과

- **AUROC (Uncertainty Quality 평가)**
    
    ![image.png](https://seokilee0412.github.io/assets/img/BSDETECTOR/image2.png)
    
    - BSDETECTOR가 모든 baseline (Likelihood-based, Temperature sampling, Self-reflection 단독) 대비 우수.
    - 예: GPT-3.5 Turbo, GSM8K AUROC = **0.951** (baseline보다 월등히 높음).
- **더 정확한 답변 선택**
    
    ![image.png](https://seokilee0412.github.io/assets/img/BSDETECTOR/image3.png)
    
    - BSDETECTOR 적용 시 LLM 답변 정확도 상승.
    - 예: GPT-3.5 Turbo, GSM8K 정확도 **47.47% → 69.44%**.
- **LLM 평가자의 신뢰성 개선**
    
    ![image.png](https://seokilee0412.github.io/assets/img/BSDETECTOR/image4.png)
    
    - human-in-the-loop 시 인간이 리뷰해야할 데이터를 선정할 때, confidence selection이 더 높은 종합적 정확도를 보임.
    - confidence 기반 human-in-the-loop 추가 시 더 높은 정확도 달성.
    - 하지만, 요약 같은 주관적 업무에 대해서는 random selection과 크게 차이 나지 않음.

---

## 결론

- **BSDETECTOR = 단순 + 범용적 + 추가 학습 불필요**
- 블랙박스 LLM에서도 불확실성 추정 가능.
- 실제 응용:
    1. **LLM 답변 신뢰도 평가**
    2. **더 정확한 답변 자동 선택**
    3. **LLM 기반 자동 평가 신뢰성 향상**
- 향후 과제: 계산 비용 줄이기 (특히 consistency 측정 부분).