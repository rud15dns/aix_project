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

- 
```
[데이터 테이블]

지표	        지표명	        지표유형	         변수값

age	        나이	        연속형	-
workclass	직업 유형	        이산형	"Private(사적), Self-emp-not-inc(자영업 비회사), Self-emp-inc(자영업 회사), Federal-gov(연방정부), Local-gov(지방정부), State-gov(주정부), Without-pay(무급), Never-worked(무직경험)"
final-weight	샘플 가중치	연속형	-
education	교육 수준	        이산형	"Bachelors(학사), Some-college(대학 중퇴), 11th(11학년), HS-grad(고졸), Prof-school(전문학교), Assoc-acdm(전문 대학), Assoc-voc(준전문학위), 9th(9학년),7th-8th(중학교 1-2학년), 12th(12학년), Masters(석사), 1st-4th(초등학교 1-4학년), 10th(10학년), Doctorate(박사), 5th-6th(초등학교 5-6학년), Preschool(유치원)"
education-num	교육 기간  	연속형	-
marital-status	결혼 상태	        이산형	"Married-civ-spouse(결혼-시민 배우자), Divorced(이혼), Never-married(미혼), Separated(별거), Widowed(과부), Married-spouse-absent(결혼-배우자 부재), arried-AF-spouse(결혼-군인 배우자)"
occupation	직업	        이산형	"Tech-support(기술 지원), Craft-repair(공예 수리), Other-service(기타 직업), Sales(판매), Exec-managerial(경영직), Prof-specialty(전문직), Handlers-cleaners(작업자 청소), Machine-op-inspct(기계 작업), Adm-clerical(행정 사무), Farming-fishing(농업 어업), Transport-moving(운송), Priv-house-serv(가정 서비스), Protective-serv(경호), Armed-Forces(군인)"
relationship	가족 역할	        이산형	"Wife(아내), Own-child(자녀), Husband(남편), Not-in-family(가족 이외), Other-relative(기타 관계), Unmarried(미혼)"
income-level	소득 수준  	이산형	<=50K; >50K
race	        인종	        이산형	"White(백인), Asian-Pac-Islander(아시아-태평양 섬주민), Amer-Indian-Eskimo(아메리카 원주민-에스키모), Black(흑인), Other(기타)"
sex	        성별	        이산형	Female(여성); Male(남성)
capital-gain	자본 이득	        연속형	-
capital-loss	자본 손실	        연속형	-
hours-per-week	주당             근로시연속형	-
country	        국적	        이산형	"United-States(미국), Cambodia(캄보디아), England(영국), Puerto-Rico(푸에르토리코), Canada(캐나다), Germany(독일), Outlying-US(Guam-USVI-etc) (미국 해외 속지), India(인도), Japan(일본), Greece(그리스), South(남미), China(중국), Cuba(쿠바), Iran(이란), Honduras(온두라스), Philippines(필리핀), Italy(이탈리아), Poland(폴란드), Jamaica(자메이카),Vietnam(베트남), Mexico(멕시코), Portugal(포르투갈), Ireland(아일랜드), France(프랑스), Dominican-Republic(도미니카 공화국), Laos(라오스), Ecuador(에콰도르), Taiwan(대만), Haiti(아이티), Columbia(콜롬비아), Hungary(헝가리), Guatemala(과테말라), Nicaragua(니카라과), Scotland(스코틀랜드), Thailand(태국), Yugoslavia(유고슬라비아), El-Salvador(엘살바도르)"

```
## III. Methodology
- Explaining your choice of algorithms (methods) - Explaining features (if any)
- 우리는 데이터의 다양한 유형을 알고 있지만 어떤 기계 학습 알고리즘을 적용하기에 가장 좋은지 결정할 수 없기 때문에 다양한 알고리즘(약 10개)을 적용해보기로 하였다.
- 여러 가지 다른 알고리즘을 통해 훈련하고 어떤 효과가 가장 좋은지 비교하고 상위 3개로 추려보고자 한다.

```ruby
# Gradient Boosting Trees 그레이디언트업 의사결정 트리 
start_time = time.time()  
train_pred_gbt, test_pred_gbt, acc_gbt, acc_cv_gbt, probs_gbt= fit_ml_algo(GradientBoostingClassifier(),   
              x_train,   
              y_train,   
              x_test,   
              10)  
gbt_time = (time.time() - start_time)  
print("Accuracy: %s" % acc_gbt)  
print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)  
print("Running Time: %s s" % datetime.timedelta(seconds=gbt_time).seconds)  
```


```ruby
# 로지스틱 회귀
# 하이퍼파라미터 설정 및 랜덤 서치 생성
n_iter_search = 10  # 10번 훈련, 값이 클수록 매개변수 정확도가 높아지지만 검색 시간이 더 오래 걸립니다.
param_dist = {'penalty': ['l2', 'l1'], 
              'class_weight': [None, 'balanced'],
              'C': np.logspace(-20, 20, 10000), 
              'intercept_scaling': np.logspace(-20, 20, 10000)} 
random_search = RandomizedSearchCV(LogisticRegression(),  # 사용할 분류기
                                   n_jobs=-1,  # 모든 CPU를 사용하여 훈련합니다. 기본값은 1로 1개의 CPU를 사용합니다.
                                   param_distributions=param_dist,
                                   n_iter=n_iter_search)  # 훈련 횟수
start = time.time()
random_search.fit(x_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time.time() - start), n_iter_search))
report(random_search.cv_results_)

```


```ruby

# AdaBoost Classifier  
start_time = time.time()  
train_pred_adb, test_pred_adb, acc_adb, acc_cv_adb, probs_adb= fit_ml_algo(AdaBoostClassifier(),   
              x_train,   
              y_train,   
              x_test,   
              10)  
adb_time = (time.time() - start_time)  
print("Accuracy: %s" % acc_adb)  
print("Accuracy CV 10-Fold: %s" % acc_cv_adb)  
print("Running Time: %s s" % datetime.timedelta(seconds=adb_time).seconds)  

```






<img width="510" alt="image" src="https://github.com/rud15dns/aix_project/assets/90837976/9d994548-47db-4fd0-aec2-e0be4a4995c4">

-  본 훈련에 사용된 10개 모델 중 경사도 상승 결정 트리, 에이다부스트, 랜덤 포레스트의 3개 모델의 정확도가 85% 이상으로 양호함을 알 수 있으며 F1 값도 다른 모델에 비해 우수한 수준이기 때문에 지표의 관점에서 이 세 모델이 본 논문에서 연구한 문제에 더 적합하다고 판단하였다.

## IV. Evaluation & Analysis
- Graphs, tables, any statistics (if any)
## V. Related Work (e.g., existing studies)
- Tools, libraries, blogs, or any documentation that you have used to do this project.
## VI. Conclusion: Discussion
