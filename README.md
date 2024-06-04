## Title :인구센서스 데이터를 활용한 소득 예측 모델 구축
## Members: 
- 손주희 | ICT융합학부 | star7613@naver.com
- 진수림
- 김채원
- 이상엽



## I. Proposal (Option 1 or 2) – This should be filled by sometime in early May.
- Motivation: Why are you doing this? - What do you want to see at the end?

- Motivation :
인구 센서스 데이터에는 가구 소득, 교육 수준, 직업 등 다양한 정보가 담겨 있어서 소득 수준을 예측할 수 있습니다. 소득은 국민의 삶의 질이 어떠한지, 국가의 경제가 어느정도 발전했는지를 나타내는 중요한 지표입니다. 소득을 향상시키는 것은 국가 뿐만 아니라 개인에게도 중요한 목표입니다.
게다가, 소득에 영향을 미치는 요인을 파악하는 것은 국가뿐만 아니라 기업들이 목표 소비자를 이해하고 정확한 마케팅 전략을 수립하는 데에도 중요한 역할을 합니다. 따라서 우리는 이러한 이유로 인구 센서스 데이터를 활용하여 소득 예측 모델을 구축하고, 실제 데이터와 목표 변수 사이의 관계를 분석하여 최적의 모델을 찾고자 합니다.


## II. Datasets
- Describing your dataset
- 소득에 영향을 미치는 요인은 복잡하고 다양하며 개인뿐만 아니라 가족, 심지어 국가의 다양한 요인에 의해 영향을 받으며 관련된 지표의 수가 방대하고 얻기 어려우며 모델 및 분석 과정을 단순화하기 위해 데이터는 UCI Machine Learning Repository 웹 사이트의 Adult Data Set 데이터 세트(https://archive.ics.uci.edu/ml/datasets/adult )에서 가져오기로 하였습니다.
- adult.names : 데이터 세트의 설명과 데이터 세트의 다양한 지표에 대한 설명이 포함
- adult.data, adult.test :  adult.data와 adult.test는  훈련 세트와 테스트 세트로 구분되며, 데이터의 분류 및 분석을 용이하게 하기 위해 본 논문에서는 adult.data와 adult.test의 두 데이터 세트를 통합하여 모델을 훈련할 때 훈련 세트와 테스트 세트를 재분할하였습니다.

```ruby
#데이터의 헤더 정의(인자/지표 이름 정의)
headers = ['age', 'workclass', 'final-weight', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'country','income-level']

#데이터 훈련 세트
adult_data = pd.read_csv('adult.data',header=None,names=headers,sep=',\s',na_values=["?"],engine='python')

#데이터 테스트 세트
adult_test = pd.read_csv('adult.test',header=None,names=headers,sep=',\s',na_values=["?"], engine='python',skiprows=1)

# 두 데이터 세트 병합
dataset = pd.concat([adult_data, adult_test], ignore_index=True) 
# 가져올 때 두 데이터 세트에 인덱스를 추가했기 때문에 병합된 DataFrame에 대해 새로운 순차적 인덱스를 부여합니다. 
dataset.reset_index(inplace=True, drop=True)
```

- 이 기사에 사용된 데이터 세트에는 총 48,842개의 샘플 15개의 지표가 포함되어 있습니다. 15개 지표 중 6개 지표는 연속형 지표이고 나머지 9개 지표는 이산형 지표로 명칭과 속성은 아래 표와 같습니다.

  
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
- 데이터셋에서 누락된 데이터가 있는지 python의 'missingno' 라이브러리를 사용하여 확인합니다.
```ruby
missingno.matrix(dataset, figsize = (20,5))
missingno.bar(dataset, sort='ascending', figsize = (20,5))
```

![image](https://github.com/rud15dns/aix_project/assets/90837976/d2b5d9fa-2b5b-40ea-8b79-2857dfeb1c3a)

![image](https://github.com/rud15dns/aix_project/assets/113186906/6f84d3dc-a047-4b3e-9bfe-c5a0bac7d86e)

> 데이터셋에서 누락된 데이터가 있는지 확인합니다.
  -> 'workclass','occupation','country' 부분에 누락된 데이터가 일부 존재한다는 것을 확인 할 수 있습니다.

- 데이터의 무결성을 위해 결측값이 포함된 행을 제거합니다
```ruby
dataset.dropna(axis=0, how='any', inplace=True)  
dataset.describe(include='all')
```


***빼도 될 것 같은데 어케 생각...??
- 훈련 세트와 테스트 세트에서 연수입 지표 나타내는 표기가 달라 표기 방식을 통일합니다 
- 필요없는 final-weight 지표를 삭제합니다. (나중에 이유 쓸 수 있으면 쓰면 좋을 것 같습니다)
```ruby
dataset.loc[dataset['income-level'] == '>50K.', 'income-level'] = '>50K'  
dataset.loc[dataset['income-level'] == '<=50K.', 'income-level'] = '<=50K'
# final-weight 지표 삭제  
dataset = dataset.drop(['final-weight'],axis=1)  
```
- 각 변수에 대한 분포를 시각화합니다. 그래프의 종류는 변수의 데이터 유형이 수치형인지 범주형인지에 따라 정의됩니다.
- 범주형(이산형) 데이터인 경우, 각 변수에 대한 빈도수를 세로 막대 그래프로 그립니다.
- 수치형(연속형) 데이터인 경우, 히스토그램과 KDE 그래포로 그립니다. 
```ruby
# 각 변수의 분포 상태 그리기  
def plot_distribution(dataset, cols, width, height, hspace, wspace):  
    plt.style.use('seaborn-whitegrid')  
    fig = plt.figure(figsize=(width,height))  
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)  
    rows = math.ceil(float(dataset.shape[1]) / cols)  
    for i, column in enumerate(dataset.columns):  
        ax = fig.add_subplot(rows, cols, i + 1)  
        ax.set_title(column)  
        if dataset.dtypes[column] == np.object:  
            g = sns.countplot(y=column, data=dataset)  
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]  
            g.set(yticklabels=substrings)  
            plt.xticks(rotation=25)  
        else:  
            g = sns.distplot(dataset[column])  
            plt.xticks(rotation=25)  
      
plot_distribution(dataset, cols=3, width=20, height=20, hspace=0.45, wspace=0.5)
```
![image](https://github.com/rud15dns/aix_project/assets/113186906/c3be7e22-ebd2-40a6-be17-24e6de624a43)

- 변수 간의 상관 관계 검사를 위한 종속변수와 독립변수의 분리 ( 데이터에 따른 income-level(소득수준)의 변화)
```ruby
# 분할 독립 변수와 종속 변수 
y_data=dataset_num['income-level']  # 소득수준income-level열   종속변수
x_data=dataset_num.drop(['income-level'],axis=1)  # 소득수준 income-level 열을 제외한 나머지 열을 X_data로  독립변수

```

- 모델 학습 준비
- 훈련 세트와 테스트 세트를 분할

```ruby
# 훈련 세트와 테스트 세트를 분할하다.  
x_train,x_test,y_train,y_test = train_test_split(  
    x_data,  
    y_data,  
    test_size=0.2,  
    random_state=1,  
    stratify=y_data)  

```

  ----수정 ing,...


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
- ![image](https://github.com/rud15dns/aix_project/assets/90837976/411f5d56-b5eb-4400-a904-c168e9d5cb9c)
- ![image](https://github.com/rud15dns/aix_project/assets/90837976/82da2d30-1275-48e7-b231-fc1f1d267e32)


## V. Related Work (e.g., existing studies)
- Tools, libraries, blogs, or any documentation that you have used to do this project.
## VI. Conclusion: Discussion
