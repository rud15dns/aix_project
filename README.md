## Title :인구센서스 데이터를 활용한 소득 예측 모델 구축
## Members: 
- 손주희 진수림 김채원 이상엽

## I. Proposal (Option 1 or 2) – This should be filled by sometime in early May.
- Motivation: Why are you doing this? - What do you want to see at the end?

- Motivation :
인구센서스 데이터에는 가구 소득, 교육 수준, 직업 등 다양한 변수가 포함되어 있어 소득 수준 예측이 가능하다
소득은 국민의 생활수준과 국가경제발전을 반영하는 중요한 지표이며 소득수준을 개선하고 삶의 질을 향상시키는 것은 각국 정부와 국민의 공동의 목표이다.
소득에 영향을 미치는 요인을 탐색하는 것은 국가에 매우 중요할 뿐만 아니라 개인의 자아 향상을 위해 기업이 목표 소비자 그룹을 식별하는 데 중요한 참고 가치가 있다.
따라서 위와 같이 인구센서스 데이터를 이용해 사회적으로 중요하게 활용 될 수 있는 소득 예측 모델 구축을 통하여 실제 데이터와 타겟 변수 간의 관계를 분석하고 최적의 모델을 선택


## II. Datasets
- Describing your dataset
- 소득에 영향을 미치는 요인은 복잡하고 다양하며 개인뿐만 아니라 가족, 심지어 국가의 다양한 요인에 의해 영향을 받으며 관련된 지표의 수가 방대하고 얻기 어려우며 모델 및 분석 과정을 단순화하기 위해 데이터는 UCI Machine Learning Repository 웹 사이트의 Adult Data Set 데이터 세트(https://archive.ics.uci.edu/ml/datasets/adult )에서 가져오기로 하였다.
- adult.names : 데이터 세트의 설명과 데이터 세트의 다양한 지표에 대한 설명이 포함
- adult.test, adult.data :  adult.data와 adult.test는 데이터 세트의 작성자를 위한 훈련 세트와 테스트 세트로 구분되며, 데이터의 분류 및 분석을 용이하게 하기 위해 본 논문에서는 adult.data와 adult.test의 두 데이터 세트를 통합하여 모델을 훈련할 때 훈련 세트와 테스트 세트를 재분할하였다.

- 이 기사에 사용된 데이터 세트에는 총 48,842개의 샘플 15개의 지표가 포함되어 있습니다. 15개 지표 중 6개 지표는 연속형 지표이고 나머지 9개 지표는 이산형 지표로 명칭과 속성은 아래 표와 같다.
