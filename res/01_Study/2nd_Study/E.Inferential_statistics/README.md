# 추론통계

## #01. 추론통계

### [1] 정의

모집단으로부터 추출된 표본의 표본통계량으로 부터 모집단의 특성인 모수에 관해 통계적으로 추론하는 절차

> 실질적인 데이터 분석(통계)를 의미하는 과정

- 자료의 정보를 이용해 집단에 관한 추측, 결론을 이끌어내는 과정
- 수집된 자료를 이용해 대상 집단(모집단)에 대한 의사결정을 하는 것으로 Sample을 통해 모집 단을 추정하는 것을 의미
- 제한된 표본을 바탕으로 모집단에 대한 일반적인 결론을 유도하려는 시도이므로 본질적으로 불확실성을 수반함.

### [2] 추론 통계의 결론

- 성별에 따른 월급의 차이가 **우연히 나타날 확률이 작다**면 통계적으로 **유의하다(statistically signficant)**라고 결론 내린다.
- 성별에 따른 월급의 차이가 **우연히 나타날 확률이 크다**면 통계적으로 **유의하지 않다(not statistically signficant)**고 결론 내린다.

> 일반적으로 통계 분석을 수행했다는 것은 추론 통계를 이용해 가설 검정을 했다는 의미.

### 변수의 종류에 따른 분석 방법 종류

![analysis](res/analysis.svg)

## #02. 데이터마이닝(머신러닝)

### 1) 개요

- 대표적인 고급 데이터 분석법
- 대용량의 자료로부터 정보를 요약
- 미래에 대한 예측
- 관계, 패턴, 규칙 등을 탐색
- 모형화
- 유용한 지식을 추출

### 2) 방법론

#### 데이터베이스에서의 지식탐색

- 데이터웨어하우스에서 데이터마트를 생성
- 각 데이터들의 속성을 사전분석을 통해 지식을 얻음

#### 기계학습

- 컴퓨터가 학습할 수 있도록 알고리즘과 기술을 개발하는 분야
- 인공신경망, 의사결정나무, 클러스터링, 베이지안분류, SVM 등

#### 패턴인식

- 원자료를 이용해서 사전지식과 패턴에서 추출된 통계 정보를 기반으로 자료 또는 패턴을 분류
- 장바구니 분석, 연관규칙 등

### 3) 활용분야

| 분야                        | 예시                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| 데이터베이스 마케팅         | 방대한 고객의 행동정보를 활용`예) 목표 마케팅, 고객세분화, 장바구니 분석, 추천 시스템 등` |
| 신용평가 및 조기경보 시스템 | 신용카드 발급, 보험, 대출 업무 등                                                                |
| 생물정보학                  | 유전자 분석, 질병 진단, 치료법/신약 개발                                                         |
| 텍스트마이닝                | 전자우편, SNS 등 디지털 텍스트 정보를 통한 고객 성향 분석, 감성 분석, 사회관계망 분석 등         |

## #03. 확률과 통계

### [1] 사건과 확률

#### (1) 확률의 이해

- 특정 사건이 일어날 가능성의 척도
- 모든 사건의 확률값은 `0`과 `1`사이
- 표본공간 $S$에 부분집합인 각 사상에 대해 실수값을 가지는 함수의 확률값이 `0`과 `1`사이에 있고，전체 확률의 합이 `1`인 것을 의미
- 표본공간 $Q$의 부분집합인 사건 $E$의 확률은 표본공간의 원소의 개수에 대한 사건 표의 개수의 비율로 확률을 $P(E)$라고 할 때

$P(E) = \frac{n(E)}{n(\Omega)}$

> $\Omega$는 전체 사건, $n(\Omega)$는 모든 경우의 수

#### (2) 확률의 용어

##### 실험 또는 시행

 여러 가능한 결과 중 하나가 일어나도록 하는 행위

##### 표본공간

- 통계적 실험을 실시할 때 타나날 수 있는 모든 결과들의 집합
- 표본공간에서 임의의 사건 $A$가 일어날 확률 $P(A)$는 항상 0과 1 사이에 있다.

##### 사건

- 표본공간의 부분집합
- 서로 배반인 사건들의 합집합의 확률은 각 사건들의 확률의 합
- **두 사건 A, B가 독립이라면 사건 B의 확률은 A가 일어난다는 가정하에서의 B의 조건부 확률과 동일.**

##### 원소

- 나타날 수 있는 개별의 결과들

##### 수학적 확률

$\frac{일어날\,수\,있는\,모든\,경우의\,수}{사건\,A가\,일어나는\,경우의\,수}$

##### 통계적 확률

- 한 사건 $A$가 일어날 확률을 $P(A)$라 할 때 $n$번의 반복시행에서 사건 $A$가 일어날 횟수를 $r$이라고 하면, 상대도수 $\frac{n}{r}$은 $n$이 커점에 따라 확률 $P(A)$에 가까워짐을 알 수 있다. 이러한 $P(A)$를 통계적 확률이라 한다.

##### 조건부 확률

- 사건 $A$가 일어났다는 가정하의 사건 $B$의 확률
- $P(B|A) = \frac{P(A \cap B)}{P(A)}$

### [2] 경우의 수

사건의 기본적인 연산

- A의 여사건: 사건 A에 포함되지 않는 집합
- A와 B의 합사건 : A or B
- A와 B의 곱사건 : A and B
- 배반사건 : 동시에 일어날 수 없는 두 사건, A and B = 0인 두 사건

#### 경우의 수의 계산

##### 합의 법칙

두 사건 A와 B가 일어나는 경우의 수가 각각 m과 n

- 두 사건이 동시에 일어나지 않음
- 사건 A 또는 사건 B가 일어나는 경우의 수는 m+n

##### 곱의 법칙

이 때 경우의 수는 m x n

##### 팩토리얼(!)

1부터 어떤 양의 정수 n까지의 정수를 모두 곱한 것, n! = nx(n-1)!

> 예) 4명의 학생을 순서대로 세우는 경우의 수는 4!