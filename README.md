## Title :인구센서스 데이터를 활용한 소득 예측 모델 구축
>AI+X 딥러닝 Final Project


## Members: 
- 손주희 | ICT융합학부 | star7613@naver.com | 깃허브, 코드 작성
- 진수림 | 인공지능학과 | sl695969@outlook.com | 데이터셋 수집 , 모델 구축
- 김채원 | ICT융합학부 | rud14dns@hanyang.ac.kr | 깃허브, 코드 작성
- 이상엽 | 화학공학과  | ben20141220@gmail.com | 발표, 영상 제작



## Index:
#### I. Proposal<br/>
#### II. Datasets<br/>
[1] 데이터 가져오기    [2] 결측 값 처리   [3] 각 변수의 분포 시각화   [4] 변수의 범주 단순화 <br/>[5] 범주형 변수 인코딩   [6] 히트맵을 통한 시각화  [7] 모델 훈련 준비 <br/>
#### III. Methodology<br/>
[1] 알고리즘 성능 비교를 위한 함수 설계  
 [2] 모델 학습 및 평가 <br/>
#### IV. Evaluation & Analysis<br/>
[1] 모델 성능 비교를 위한 데이터프레임 생성 
 [2] ROC 곡선 시각화 
 [3] Precision-Recall 곡선 시각화 <br/>
#### V. Related Work
<br/>

## Video Link :<br/>
https://www.youtube.com/watch?v=cEOBPfzEGpQ

<br/><br/>


## I. Proposal (Option A )<br/>


### Motivation (동기) :

인구 센서스 데이터에는 가구 소득, 교육 수준, 직업 등 다양한 정보가 포함되어 있어 소득 수준을 예측할 수 있습니다. <br/>
소득은 국민의 삶의 질과 국가의 경제 발전 정도를 나타내는 중요한 지표로, 소득 향상은 국가뿐만 아니라 개인에게도 중요한 목표입니다. <br/>
또한, 기업들이 목표 소비자를 이해하고 정확한 마케팅 전략을 수립하는 데에도 큰 역할을 합니다. <br/>
이러한 이유로 소득 측정의 중요성을 인식하고, 수업 시간에 배운 딥러닝 기법을 활용하여 인구 센서스 데이터를 기반으로 소득 예측 모델을 구축하고자 합니다.





<br/>

### project goal (프로젝트 목표) : 
#### 소득 예측 모델을 구축하고 실제 데이터와 목표 변수 사이의 관계를 분석하여 최적의 모델을 찾고자 합니다.
<br/>


## II. Datasets

### [1] 데이터 가져오기

- 소득에 영향을 미치는 요인은 복잡하고 다양하며 개인뿐만 아니라 가족, 심지어 국가의 다양한 요인에 의해 영향을 받으며 관련된 지표의 수가 방대하고 얻기 어렵기 떄문에 모델 및 분석 과정을 단순화하기 위해 데이터는 UCI Machine Learning Repository 웹 사이트의 Adult Data Set 데이터 세트(https://archive.ics.uci.edu/ml/datasets/adult )에서 가져오기로 하였습니다.

- adult.names : 데이터 세트의 설명과 데이터 세트의 다양한 지표에 대한 설명이 포함되어있습니다.
- adult.data, adult.test :  adult.data와 adult.test는 훈련 세트와 테스트 세트로 구분되며, 데이터의 분류 및 분석을 용이하게 하기 위해 본 내용에서는 adult.data와 adult.test의 두 데이터 세트를 통합하여 모델을 훈련할 때 훈련 세트와 테스트 세트를 재분할하였습니다. <br/><br/>

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
<br/><br/>  

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
<br/><br/>  
### [2] 결측 값 처리<br/>  
- 데이터셋에서 누락된 데이터가 있는지 python의 'missingno' 라이브러리를 사용하여 확인합니다.
```ruby
missingno.matrix(dataset, figsize = (20,5))
missingno.bar(dataset, sort='ascending', figsize = (20,5))
```

![image](https://github.com/rud15dns/aix_project/assets/90837976/d2b5d9fa-2b5b-40ea-8b79-2857dfeb1c3a)

![image](https://github.com/rud15dns/aix_project/assets/113186906/6f84d3dc-a047-4b3e-9bfe-c5a0bac7d86e)

> 데이터셋에서 누락된 데이터가 있는지 확인합니다.
  -> 'workclass','occupation','country' 부분에 누락된 데이터가 일부 존재한다는 것을 확인 할 수 있습니다.

<br/><br/>


- 데이터의 무결성을 위해 결측값이 포함된 행을 제거합니다
```ruby
dataset.dropna(axis=0, how='any', inplace=True)  
dataset.describe(include='all')
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>final-weight</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>country</th>
      <th>income-level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>45222.000000</td>
      <td>45222</td>
      <td>4.522200e+04</td>
      <td>45222</td>
      <td>45222.000000</td>
      <td>45222</td>
      <td>45222</td>
      <td>45222</td>
      <td>45222</td>
      <td>45222</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222</td>
      <td>45222</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>7</td>
      <td>NaN</td>
      <td>16</td>
      <td>NaN</td>
      <td>7</td>
      <td>14</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41</td>
      <td>4</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>Private</td>
      <td>NaN</td>
      <td>HS-grad</td>
      <td>NaN</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>33307</td>
      <td>NaN</td>
      <td>14783</td>
      <td>NaN</td>
      <td>21055</td>
      <td>6020</td>
      <td>18666</td>
      <td>38903</td>
      <td>30527</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41292</td>
      <td>22654</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.547941</td>
      <td>NaN</td>
      <td>1.897347e+05</td>
      <td>NaN</td>
      <td>10.118460</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1101.430344</td>
      <td>88.595418</td>
      <td>40.938017</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.217870</td>
      <td>NaN</td>
      <td>1.056392e+05</td>
      <td>NaN</td>
      <td>2.552881</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7506.430084</td>
      <td>404.956092</td>
      <td>12.007508</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>NaN</td>
      <td>1.349200e+04</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.000000</td>
      <td>NaN</td>
      <td>1.173882e+05</td>
      <td>NaN</td>
      <td>9.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
      <td>NaN</td>
      <td>1.783160e+05</td>
      <td>NaN</td>
      <td>10.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>47.000000</td>
      <td>NaN</td>
      <td>2.379260e+05</td>
      <td>NaN</td>
      <td>13.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>NaN</td>
      <td>1.490400e+06</td>
      <td>NaN</td>
      <td>16.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>99999.000000</td>
      <td>4356.000000</td>
      <td>99.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
<br/><br/> 

### [3] 각 변수의 분포 시각화
- 각 변수에 대한 분포를 시각화합니다. 그래프의 종류는 변수의 데이터 유형이 수치형인지 범주형인지에 따라 정의됩니다.
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
> - 범주형(이산형) 데이터인 경우, 각 변수에 대한 빈도수를 세로 막대 그래프로 그립니다.
> - 수치형(연속형) 데이터인 경우, 히스토그램과 KDE 그래프로 그립니다. 

<br/><br/>  

### [4] 변수의 범주 단순화
> 보다 효율적인 모델 학습을 위해 데이터셋 내의 범주형 변수를 더 일반적이고 일관성 있는 범주로 변환합니다.

<br/>

-  데이터 셋 'workclass' 의 범주 단순화


``` ruby
#위의 막대 그래프에서 민간 작업이 표본에서 차지하는 비중이 상대적으로 크고 비작업 및 비수입 작업 표본의 수는 매우 적음을 알 수 있으며 실제 상황에 따라 5가지 범주로 요약됩니다.
dataset.loc[dataset['workclass'] == 'Without-pay', 'workclass'] = 'Not Working'  
dataset.loc[dataset['workclass'] == 'Never-worked', 'workclass'] = 'Not Working'  
dataset.loc[dataset['workclass'] == 'Federal-gov', 'workclass'] = 'Fed-gov'  
dataset.loc[dataset['workclass'] == 'State-gov', 'workclass'] = 'Non-fed-gov'  
dataset.loc[dataset['workclass'] == 'Local-gov', 'workclass'] = 'Non-fed-gov'  
dataset.loc[dataset['workclass'] == 'Self-emp-not-inc', 'workclass'] = 'Self-emp'  
dataset.loc[dataset['workclass'] == 'Self-emp-inc', 'workclass'] = 'Self-emp'  
dataset.loc[dataset['workclass'] == ' Private', 'workclass'] = ' Private'  

plt.style.use('seaborn-whitegrid')  
fig = plt.figure(figsize=(15, 4))   
sns.countplot(y="workclass", data=dataset); 


```
![image](https://github.com/rud15dns/aix_project/assets/113186906/f6bd1d2e-5aad-4a79-bd30-6cd72b110e36)

> - 위의 막대 그래프에서 민간 작업이 표본에서 차지하는 비중이 상대적으로 크고 비작업 및 비수입 작업 표본의 수는 매우 적음을 확인하였습니다.
> - 해당 동향을 파악하여 비슷한 범주를 통합함으로써 5가지 범주로 요약하였습니다.

<br/>

- 데이터 셋 'Occupation' 의 범주 단순화
  

``` ruby
dataset.loc[dataset['occupation'] == 'Adm-clerical', 'occupation'] = 'Admin'  # 행정 사무
# 군대 관련 직업과 보호 서비스 직업을 하나의 군사 관련으로 묶었습니다.
dataset.loc[dataset['occupation'] == 'Armed-Forces', 'occupation'] = 'Military'  #군대
dataset.loc[dataset['occupation'] == 'Protective-serv', 'occupation'] = 'Military'
#육체 노동을 필요로 하는 직업들을 하나의 그룹으로 묶었습니다.
dataset.loc[dataset['occupation'] == 'Craft-repair', 'occupation'] = 'Manual Labour'# 육체노동자  
dataset.loc[dataset['occupation'] == 'Transport-moving', 'occupation'] = 'Manual Labour'   
dataset.loc[dataset['occupation'] == 'Farming-fishing', 'occupation'] = 'Manual Labour'   
dataset.loc[dataset['occupation'] == 'Handlers-cleaners', 'occupation'] = 'Manual Labour'    
dataset.loc[dataset['occupation'] == 'Machine-op-inspct', 'occupation'] = 'Manual Labour'
#주로 사무실에서 이루어지는 일을 하는 직업들을 하나의 그룹으로 묶었습니다   
dataset.loc[dataset['occupation'] == 'Exec-managerial', 'occupation'] = 'Office Labour'  # 문서 작업
dataset.loc[dataset['occupation'] == 'Sales', 'occupation'] = 'Office Labour' 
dataset.loc[dataset['occupation'] == 'Tech-support', 'occupation'] = 'Office Labour'
#서비스 직업들을 하나의 그룹으로 묵었습니다.
dataset.loc[dataset['occupation'] == 'Other-service', 'occupation'] = 'Service'#  서비스 직종
dataset.loc[dataset['occupation'] == 'Priv-house-serv', 'occupation'] = 'Service'  
dataset.loc[dataset['occupation'] == 'Prof-specialty', 'occupation'] = 'Professional'# 기술자
  
plt.style.use('seaborn-whitegrid')  
fig = plt.figure(figsize=(20,3))  
sns.countplot(y="occupation", data=dataset);
```
![image](https://github.com/rud15dns/aix_project/assets/113186906/7633dc7f-064d-4edd-916e-1d7238be6b1b)
> - 위의 막대 그래프에서 비슷한 범주를 통합함으로써 6가지 범주로 요약하였습니다.
<br/>

-  데이터 셋 'Country' 의 범주 단순화
``` ruby
#총 41개 국가와 지역이 데이터 셋에 있고, 미국을 제외한 대부분의 국가 및 지역의 샘플이 거의 없으므로 나머지 국가는 대륙별로 범주를 단순화시켰습니다.
dataset.loc[dataset['country'] == 'China', 'country'] = 'East-Asia'  
dataset.loc[dataset['country'] == 'Hong', 'country'] = 'East-Asia'  
dataset.loc[dataset['country'] == 'Taiwan', 'country'] = 'East-Asia'  
dataset.loc[dataset['country'] == 'Japan', 'country'] = 'East-Asia'  
  
dataset.loc[dataset['country'] == 'Thailand', 'country'] = 'Southeast-Asia'  
dataset.loc[dataset['country'] == 'Vietnam', 'country'] = 'Southeast-Asia'  
dataset.loc[dataset['country'] == 'Laos', 'country'] = 'Southeast-Asia'  
dataset.loc[dataset['country'] == 'Philippines', 'country'] = 'Southeast-Asia'  
dataset.loc[dataset['country'] == 'Cambodia', 'country'] = 'Southeast-Asia'  
  
dataset.loc[dataset['country'] == 'Columbia', 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'Cuba', 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'Dominican-Republic', 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'Ecuador', 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'Guatemala', 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'El-Salvador', 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'Haiti', 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'Honduras', 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'Mexico', 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'Nicaragua', 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'Outlying-US(Guam-USVI-etc)'  , 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'Peru', 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'Jamaica', 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'Puerto-Rico', 'country'] = 'South-America'  
dataset.loc[dataset['country'] == 'Trinadad&Tobago', 'country'] = 'South-America'  
  
dataset.loc[dataset['country'] == 'Canada', 'country'] = 'British-Commonwealth'  
dataset.loc[dataset['country'] == 'England', 'country'] = 'British-Commonwealth'  
dataset.loc[dataset['country'] == 'India', 'country'] = 'British-Commonwealth'  
dataset.loc[dataset['country'] == 'Ireland', 'country'] = 'British-Commonwealth'  
dataset.loc[dataset['country'] == 'Scotland', 'country'] = 'British-Commonwealth'  
  
dataset.loc[dataset['country'] == 'France', 'country'] = 'Europe'  
dataset.loc[dataset['country'] == 'Germany', 'country'] = 'Europe'  
dataset.loc[dataset['country'] == 'Italy', 'country'] = 'Europe'  
dataset.loc[dataset['country'] == 'Holand-Netherlands', 'country'] = 'Europe'  
dataset.loc[dataset['country'] == 'Greece', 'country'] = 'Europe'  
dataset.loc[dataset['country'] == 'Hungary', 'country'] = 'Europe'  
dataset.loc[dataset['country'] == 'Iran', 'country'] = 'Europe'  
dataset.loc[dataset['country'] == 'Yugoslavia', 'country'] = 'Europe'  
dataset.loc[dataset['country'] == 'Poland', 'country'] = 'Europe'  
dataset.loc[dataset['country'] == 'Portugal', 'country'] = 'Europe'  
dataset.loc[dataset['country'] == 'South', 'country'] = 'Europe'  
  
dataset.loc[dataset['country'] == 'United-States', 'country'] = 'United-States'  
#국가를 지역으로 통합한 후에도 미국과 비교하면 여전히 큰 차이가 있지만 각 범주의 표본 수가 초기 데이터 세트보다 균일해졌습니다.

plt.style.use('seaborn-whitegrid')  
fig = plt.figure(figsize=(15,4))   
sns.countplot(y="country", data=dataset);  
```
![image](https://github.com/rud15dns/aix_project/assets/113186906/aa5351d8-4295-4e14-8365-01c39d9262a7)
<br/>

- 데이터 셋 'Education'의 범주 단순화

``` ruby
#위의 막대 그래프에서 확인해 본 결과, 교육 수준은 총 16가지로 나뉘며,
# 교육수준이 낮은 각 카테고리의 수가 적음을 알 수 있었습니다.
# 고등학교를 기준으로 하여 그 아래의 교육수준은 dropout으로 통합하고,
# Assoc-acdm과 Assoc-voc는 Associate로 통합하고,
# HS-Grad와 Some-college는 HS-Graduate으로 통합하였습니다. 
dataset.loc[dataset['education'] == 'Preschool', 'education'] = 'Dropout'# 중퇴
dataset.loc[dataset['education'] == '1st-4th', 'education'] = 'Dropout'  
dataset.loc[dataset['education'] == '5th-6th', 'education'] = 'Dropout'  
dataset.loc[dataset['education'] == '7th-8th', 'education'] = 'Dropout'  
dataset.loc[dataset['education'] == '9th', 'education'] = 'Dropout' 
dataset.loc[dataset['education'] == '10th', 'education'] = 'Dropout'  
dataset.loc[dataset['education'] == '11th', 'education'] = 'Dropout'  
dataset.loc[dataset['education'] == '12th', 'education'] = 'Dropout'

dataset.loc[dataset['education'] == 'Assoc-acdm', 'education'] = 'Associate'  # 전문대학 
dataset.loc[dataset['education'] == 'Assoc-voc', 'education'] = 'Associate'

dataset.loc[dataset['education'] == 'HS-Grad', 'education'] = 'HS-Graduate'  #고등 학교  
dataset.loc[dataset['education'] == 'Some-college', 'education'] = 'HS-Graduate'

dataset.loc[dataset['education'] == 'Prof-school', 'education'] = 'Professor'  # 직업학교.  
dataset.loc[dataset['education'] == 'Bachelors', 'education'] = 'Bachelors'  # 학사.  
dataset.loc[dataset['education'] == 'Masters', 'education'] = 'Masters'  # 석사.  
dataset.loc[dataset['education'] == 'Doctorate', 'education'] = 'Doctorate'  # 박사.
plt.style.use('seaborn-whitegrid')  
fig = plt.figure(figsize=(15,4))   
sns.countplot(y="education", data=dataset); 
```
![image](https://github.com/rud15dns/aix_project/assets/113186906/6feaba9d-0862-4c00-953f-54f22e7611f8)

<br/>

- 데이터 셋 'marital-status'의 범주 단순화

```ruby
# 7가지의 혼인 상태를 5가지로 통합하였다: 
# 한 번도 결혼한 적이 없는 상태 / 이혼한 상태 / 기혼인 상태 / 사별한 상태 / 별거 중인 상태 
dataset.loc[dataset['marital-status'] == 'Never-married', 'marital-status'] = 'Never-Married'  # 한 번도 결혼한 적이 없는 상태
dataset.loc[dataset['marital-status'] == 'Divorced', 'marital-status'] = 'Divorced'# 이혼한 상태  
dataset.loc[dataset['marital-status'] == 'Widowed', 'marital-status'] = 'Widowed'# 사별한 상태
dataset.loc[dataset['marital-status'] == 'Married-spouse-absent', 'marital-status'] = 'Separated'  # 별거 중인 상태
dataset.loc[dataset['marital-status'] == 'Separated', 'marital-status'] = 'Separated'
dataset.loc[dataset['marital-status'] == 'Married-AF-spouse', 'marital-status'] = 'Married'  # 기혼인 상태
dataset.loc[dataset['marital-status'] == 'Married-civ-spouse', 'marital-status'] = 'Married'    
  
plt.style.use('seaborn-whitegrid')  
fig = plt.figure(figsize=(10,3))   
sns.countplot(y="marital-status", data=dataset);  


```
![image](https://github.com/rud15dns/aix_project/assets/113186906/2ab24e8b-a7a8-4ab2-9460-ef4fb9f397dd)
<br/><br/>

- 범주를 단순화시킨 작업을 끝낸 후, 데이터 세트의 전체 분포 상황을 확인하였습니다.
``` ruby
plot_distribution(dataset, cols=3, width=20, height=20, hspace=0.45, wspace=0.5)
```
![image](https://github.com/rud15dns/aix_project/assets/113186906/a199b23b-d6f1-4f26-8563-542221c6f453)

<br/><br/>

### [5] 범주형 변수 인코딩<br/>
- 데이터 세트를 머신 러닝 모델에 적합한 형식으로 바꾸기 위해 범주형 변수를 숫자형 변수로 변환합니다.
```ruby
dataset_num = dataset.copy() # 데이터 세트 복사  
#범주형 변수를 숫자형 변수로 변환
dataset_num['workclass'] = dataset_num['workclass'].factorize()[0]  
dataset_num['education'] = dataset_num['education'].factorize()[0]  
dataset_num['marital-status'] = dataset_num['marital-status'].factorize()[0]  
dataset_num['occupation'] = dataset_num['occupation'].factorize()[0]  
dataset_num['relationship'] = dataset_num['relationship'].factorize()[0]  
dataset_num['race'] = dataset_num['race'].factorize()[0]  
dataset_num['sex'] = dataset_num['sex'].factorize()[0]  
dataset_num['country'] = dataset_num['country'].factorize()[0]  
dataset_num['income-level'] = dataset_num['income-level'].factorize()[0]  
```
> - 각 범주형 변수에 대해 'factorize()' 함수를 사용하여 숫자형으로 변환합니다.
> - factorize() 함수는 각 고유한 범주의 값을 정수로 맵핑합니다.
> - 예를 들어서, workclass 열의 고유 값들이 ['Private', 'Self-emp-not-inc', 'Logical-gov', 'Private', 'Private...]이라면,<br/>
>  이들을 각각 [0, 1, 2, 0, 0, ...]와 같이 정수로 변환합니다.

<br/><br/>

### [6] 히트맵을 통한 시각화 <br/>
- 데이터셋의 숫자형 변수들 간의 상관 관계를 히트맵을 통해 확인합니다.

```ruby
 #변수 간의 상관 관계 검사
plt.style.use('seaborn-whitegrid')  
fig = plt.figure(figsize=(15, 15))   
  
mask = np.zeros_like(dataset_num.corr(), dtype=np.bool)  
mask[np.triu_indices_from(mask)] = True  
sns.heatmap(dataset_num.corr(), vmin=-1, vmax=1, square=True,   
            cmap=sns.color_palette("RdBu_r", 100),   
            mask=mask, annot=True, linewidths=.5);  
```  
  ![image](https://github.com/rud15dns/aix_project/assets/90837976/c10b4a5c-662d-42e6-b457-0890b6d3035d)
  >데이터에 따른 income-level(소득수준)의 변화를 확인 할 수 있습니다.
<br/>

### [7] 모델 훈련 준비 <br/>
 

-  데이터셋에 x_data에는   'income-level' 열을 제외한 나머지 열들을 독립 변수 즉 입력 값들로 설정하고 y_data에는 종속 변수인 'income-level' 열을 저장합니다.

```ruby
#  독립 변수와 종속 변수를 분리
x_data=dataset_num.drop(['income-level'],axis=1)  # 소득수준 income-level 열을 제외한 나머지 열을 X_data로  독립변수
y_data=dataset_num['income-level']  # 소득수준income-level열   종속변수

```
<br/>

- 모델을 훈련시키기 위해 필요한 입력 데이터(x_data)와 예측해야 하는 목표 값(y_data)을 분리해 줍니다.

```ruby
# 훈련 세트와 테스트 세트를 분할
x_train,x_test,y_train,y_test = train_test_split(  
    x_data,  
    y_data,  
    test_size=0.2,  
    random_state=1,  
    stratify=y_data)  

```


<br/><br/>

## III. Methodology <br/> 

- 이전 과정에서 데이터를 시각화하여 다양한 유형을 파악했지만, 어떤 기계 학습 알고리즘이 가장 적합할지 결정할 수 없었습니다. 따라서 약 10개의 다양한 알고리즘을 적용해보기로 했습니다.
- 여러 가지 알고리즘으로 모델을 훈련한 후, 그 성능을 비교하여 가장 효과적인 상위 3개의 알고리즘을 선정하고자 합니다.
  <br/><br/>

### [1] 알고리즘 성능 비교를 위한 함수 설계 <br/>
- 다양한 머신러닝 알고리즘을 쉽게 실험하고, 그 성능을 비교할 수 있도록 함수를 설계합니다. 사용자는 이 함수를 호출할 때, 다양한 머신러닝 알고리즘과 함께 훈련 및 검증 할 데이터를 전달하기만 하면 됩니다.
```ruby
# 모델 세트용 템플릿을 구성하고, 자동으로 훈련 세트를 호출하여 들어오는 모델을 훈련하고, 검증 세트를 사용하여 모델을 검증하고, 관련 지표를 출력하도록 설계합니다.
def fit_ml_algo(algo, X_train, y_train, X_test, cv):  
    model = algo.fit(X_train, y_train)  
    test_pred = model.predict(X_test)  
    try:  
        probs = model.predict_proba(X_test)[:,1]  
    except Exception as e:  
        probs = "Unavailable"  
        print('Warning: Probs unavaliable.')  
        print('Reason: ', e)  
          
      
    acc = round(model.score(X_test, y_test) * 100, 2)   
    
    # CV -> 모델 여러번 학습 및 검증  
    train_pred = model_selection.cross_val_predict(algo,   
                                                  X_train,   
                                                  y_train,   
                                                  cv=cv,   
                                                  n_jobs = -1)  
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)  
    return train_pred, test_pred, acc, acc_cv, probs  

```
> 훈련 예측값(train_pred), 검증 데이터에 대한 예측값(test_pred), 검증 데이터에 대한 정확도(acc), 교차 검증 정확도(acc_cv), 그리고 예측 확률(probs)을 반환합니다.

<br/>

### [2] 모델 학습 및 평가 <br/>
> 각 코드를 실행하면 해당 모델을 사용하여 학습하고, 테스트 세트에 대한 정확도, 10-Fold 교차 검증 정확도, 실행 시간을 출력하게 됩니다.


<br/>

- (1) Random Forest Classifier(랜덤 포레스트) 모델을 사용하여 데이터셋을 학습시키고, 모델의 성능을 평가합니다.
> - 배깅 기법을 사용하여 여러 학습기를 병렬로 학습시킵니다. 각 학습기는 서로 다른 데이터 샘플과 특징을 사용하여 학습됩니다. 그 후, 각 학습기의 예측을 평균내어 최종 예측을 만듭니다. 
> - 배깅은 여러 개의 모델을 독립적으로 학습시키고, 이들의 예측을 결합하여 최종 예측을 만드는 방법입니다. 
  
```ruby
# 랜덤 탐색기를 사용하여 계산한 최적의 하이퍼 파라미터 모델을 사용하여 계산하다  
import datetime
start_time = time.time()  

now = datetime.datetime.now()
print(now)

rfc = random_search.best_estimator_  
train_pred_rf, test_pred_rf, acc_rf, acc_cv_rf, probs_rf = fit_ml_algo(  
                                                             rfc,   
                                                             x_train,   
                                                             y_train,   
                                                             x_test,   
                                                             10)  
rf_time = (time.time() - start_time)  
print("Accuracy: %s" % acc_rf)  
print("Accuracy CV 10-Fold: %s" % acc_cv_rf)  
print("Running Time: %s s" % datetime.timedelta(seconds=rf_time).seconds)  

```
<br/>

- (2) Gradient Boosting Trees (그레이디언트업 의사결정 트리) 모델을 사용하여 데이터셋을 학습시키고, 모델의 성능을 평가합니다.
> - 새로운 트리는 이전 트리들의 오차를 줄이는 방향으로 학습되는데, 이 때 오차를 줄이기 위해 경사하강법을 이용하여 모델을 최적화합니다.
> - 최종 트리는 각 트리의 예측값을 모두 합산하여 만듭니다. 

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

<br/>

-  (3) AdaBoost Classifier(에이다부스트) 모델을 사용하여 데이터셋을 학습시키고, 모델의 성능을 평가합니다.
> - Scikit-Learn 에서 제공하는 AdaBoost Classifier입니다.
> - 여러 개의 약한 학습기를 결합하여 분류 성능을 향상시킵니다. 
> - 각 학습기는 이전 학습기의 오차를 보완하는 방식으로 학습됩니다. 잘못 분류된 데이터에 더 큰 가중치를 부여하여 다음 학습기가 이를 올바르게 분류하도록 학습합니다. 
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
-  (4) Naive Bayes(GaussianNB) 모델을 사용하여 데이터셋을 학습시키고, 모델의 성능을 평가합니다.
> - 머신러닝의 Naive Bayes 알고리즘 중 하나입니다.
> - 모든 특징이 서로 독립적이라고 가정합니다. 
> - 연속형 데이터를 처리하기 위해 각 특징이 정규분포를 따른다고 가정합니다.
> - 주어진 트레이닝 데이터를 이용해서 각 클래스에 대한 각 특징의 평균과 분산을 계산합니다. 새로운 데이터가 주어졌을 때, 각 클래스에 대한 조건부 확률을 계산하여 가장 높은 확률을 가진 클래스로 예측합니다.
```ruby
train_pred_gaussian, test_pred_gaussian, acc_gaussian, acc_cv_gaussian, probs_gau= fit_ml_algo(GaussianNB(),   
              x_train,   
              y_train,   
              x_test,   
              10)
gaussian_time = (time.time() - start_time)  
print("Accuracy: %s" % acc_gaussian)  
print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)  
print("Running Time: %s s" % datetime.timedelta(seconds=gaussian_time).seconds) 
```
-  (5) Linear SVC 모델을 사용하여 데이터셋을 학습시키고, 모델의 성능을 평가합니다.
> - SVM은 클래스를 구분하는 분류 문제에서, 각 클래스를 잘 구분하는 선을 그어주는 방식입니다.
> - 클래스 간의 선을 그어주게 되고, 가장 가까이 있는 점들을 Support Vector라고 하고, 찾은 직선과 서포트 벡터 사이의 처리를 최대 마진이라고 합니다. 마진을 최대로 하는 서포트벡터와 직선을 찾는 것이 목표입니다.
> - SVC는 SVM을 구현하는 Scikit-Learn의 클래스이며, 그 중 선형 커널을 사용하는 것이 Linera SVC모델입니다. 선형 커널은 데이터가 선형적으로 구분될 수 있는 경우에 적합합니다. 
```ruby
# Linear SVC  
start_time = time.time()  
# kernel = ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’  
svc_clf = SVC(probability=True, max_iter=1000, kernel='linear')  
train_pred_svc, test_pred_svc, acc_linear_svc, acc_cv_linear_svc, probs_svc= fit_ml_algo(svc_clf,  
              x_train,   
              y_train,  
              x_test,   
              10)  
linear_svc_time = (time.time() - start_time)  
print("Accuracy: %s" % acc_linear_svc)  
print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)  
print("Running Time: %s s" % datetime.timedelta(seconds=linear_svc_time).seconds)  
```
- (6) Stochastic Gradient Descent 모델을 사용하여 데이터셋을 학습시키고, 모델의 성능을 평가합니다.
> - 매 반복마다 무작위로 선택된 데이터를 사용하여 가중치를 업데이트합니다.
> - 경사하강법의 변형으로, 매우 큰 데이터셋에서도 빠르고 효율적으로 학습을 수행할 수 있습니다.
```ruby
# Stochastic Gradient Descent 무작위 구배 하강
start_time = time.time()  
train_pred_sgd, test_pred_sgd, acc_sgd, acc_cv_sgd, probs_sgd= fit_ml_algo(  
              SGDClassifier(n_jobs = -1, loss='log'),   
              x_train,   
              y_train,   
              x_test,   
              10)  
sgd_time = (time.time() - start_time)  
print("Accuracy: %s" % acc_sgd)  
print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)  
print("Running Time: %s s" % datetime.timedelta(seconds=sgd_time).seconds)  
```
-  (7) Voting Classifier 모델을 사용하여 데이터셋을 학습시키고, 모델의 성능을 평가합니다.
> - 여러 개의 다른 머신러닝 모델(예 : 로지스틱 회귀, 결정 트리 등)을 결합하여 최종 예측을 만드는 학습 기법입니다.
> - voting = 'soft'를 하여, 소프트 보팅을 사용합니다. 각 모델의 예측 확률을 평균내어 최종 예측을 결정합니다. 
```ruby
# Voting Classifier  
start_time = time.time()  
voting_clf = VotingClassifier(estimators=[  
    ('log_clf', LogisticRegression()),   
    ('gnb_clf', GaussianNB()),  
    ('rf_clf', RandomForestClassifier(n_estimators=10)),  
    ('gb_clf', GradientBoostingClassifier()),  
    ('dt_clf', DecisionTreeClassifier(random_state=666))],  
                             voting='soft', n_jobs = -1)  
train_pred_vot, test_pred_vot, acc_vot, acc_cv_vot, probs_vot= fit_ml_algo(voting_clf,   
              x_train,   
              y_train,   
              x_test,   
              10)  
vot_time = (time.time() - start_time)  
print("Accuracy: %s" % acc_vot)  
print("Accuracy CV 10-Fold: %s" % acc_cv_vot)  
print("Running Time: %s s" % datetime.timedelta(seconds=vot_time).seconds)  
```
-  (8) K-NN 모델을 사용하여 데이터셋을 학습시키고, 모델의 성능을 평가합니다.
> - 어떤 데이터가 주어지면, 그 주변(이웃)의 데이터 k개를 살펴본 뒤 더 많은 데이터가 포함되어 있는 범주로 분류합니다.
> - 모델 훈련이 별도로 필요하지 않습니다.
> - n_neighbors = 3으로 하여, 예측을 위해 참조할 이웃의 수를 3으로 설정하였습니다.
```ruby
# k-Nearest Neighbors  
start_time = time.time()  
train_pred_knn, test_pred_knn, acc_knn, acc_cv_knn, probs_knn  = fit_ml_algo(KNeighborsClassifier(n_neighbors = 3,  
                                   n_jobs = -1),   
                                   x_train,   
                                   y_train,   
                                   x_test,   
                                   10)  
knn_time = (time.time() - start_time)  
print("Accuracy: %s" % acc_knn)  
print("Accuracy CV 10-Fold: %s" % acc_cv_knn)  
print("Running Time: %s s" % datetime.timedelta(seconds=knn_time))  
```

-  (9) Logistic Regression 모델을 사용하여 데이터셋을 학습시키고, 모델의 성능을 평가합니다.
> - 주어진 입력(독립 변수)에 대해 특정 클래스에 속할 확률을 예측합니다.
> - 예측된 확률은 로지스틱 함수를 통해서 결정되며, 이 모델은 결과가 0과 1 사이의 값으로 제한됩니다.
```ruby
# 로지스틱 회귀
# 하이퍼파라미터 설정 및 랜덤 서치 생성
n_iter_search = 10  # 10번 훈련, 값이 클수록 매개변수 정확도가 높아지지만 검색 시간이 더 오래 걸립니다.
param_dist = {'penalty': ['l2', 'l1'], 
              'class_weight': [None, 'balanced'],
              'C': np.logspace(-20, 20, 10000), 
              'intercept_scaling': np.logspace(-20, 20, 10000)}
# RandomizedSearchCV를 사용하여 로지스틱 회귀 모델의 최적 하이퍼파라미터를 탐색합니다.
random_search = RandomizedSearchCV(LogisticRegression(),  # 사용할 분류기
                                   n_jobs=-1,  # 모든 CPU를 사용하여 훈련합니다. 기본값은 1로 1개의 CPU를 사용합니다.
                                   param_distributions=param_dist,
                                   n_iter=n_iter_search)  # 훈련 횟수
start = time.time()
random_search.fit(x_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time.time() - start), n_iter_search))
report(random_search.cv_results_)

# 랜덤 탐색기에서 얻은 매개변수가 가장 좋은 로지스틱 회귀 모델을 호출하여 훈련합니다.  
start_time = time.time()  
train_pred_log, test_pred_log, acc_log, acc_cv_log, probs_log = fit_ml_algo(  
                                                                 random_search.best_estimator_,   
                                                                 x_train,   
                                                                 y_train,   
                                                                 x_test,   
                                                                 10)  
log_time = (time.time() - start_time)  
print("Accuracy: %s" % acc_log)  
print("Accuracy CV 10-Fold: %s" % acc_cv_log)  
print("Running Time: %s s" % datetime.timedelta(seconds=log_time).seconds)  


```
- (10) Decision Tree Classifier 모델을 사용하여 데이터셋을 학습시키고, 모델의 성능을 평가합니다.
> - 이 모델은 데이터의 특성을 기반으로 Decision nodes와 lead nodes를 포함한 트리 구조를 생성합니다. 각 내부 노드는 데이터의 속성에 대한 결정 규칙을 나타내며, 각 leaf node는 결과의 클래스 레이블(최종적인 예측)을 나타냅니다.
> - 데이터를 가장 잘 나눌 수 있는 특성과 그 임계값을 찾기 위하여 엔트로피(데이터의 불확실성)와 같은 기준을 사용합니다. 
```ruby
# Decision Tree Classifier  
start_time = time.time()  
train_pred_dt, test_pred_dt, acc_dt, acc_cv_dt, probs_dt= fit_ml_algo(DecisionTreeClassifier(),   
              x_train,   
              y_train,   
              x_test,   
              10)  
dt_time = (time.time() - start_time)  
print("Accuracy: %s" % acc_dt)  
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)  
print("Running Time: %s s" % datetime.timedelta(seconds=dt_time).seconds)  
```

<br/><br/>



## IV. Evaluation & Analysis

### [1] 모델 성능 비교를 위한 데이터프레임 생성

- 주어진 코드는 여러 머신러닝 모델의 성능을 비교하기 위해 정확도, 교차 검증 정확도, 정밀도, 재현율, F1 점수를 포함한 데이터프레임을 생성합니다.  <br/> 이를 통해 각 모델의 성능을 쉽게 비교할 수 있습니다.

```ruby
models = pd.DataFrame({  
    'Model': ['KNN', 'Logistic Regression',   
              'Random Forest', 'Naive Bayes',   
              'Stochastic Gradient Decent', 'Linear SVC',   
              'Decision Tree', 'Gradient Boosting Trees',   
              'AdaBoost', 'Voting'],  
    'Acc': [  
        acc_knn, acc_log, acc_rf,   
        acc_gaussian, acc_sgd,   
        acc_linear_svc, acc_dt,  
        acc_gbt, acc_adb, acc_vot  
    ],  
    'Acc_cv': [  
        acc_cv_knn, acc_cv_log,   
        acc_cv_rf, acc_cv_gaussian,   
        acc_cv_sgd, acc_cv_linear_svc,   
        acc_cv_dt, acc_cv_gbt,  
        acc_cv_adb, acc_cv_vot  
    ],  
    'precision': [  
        round(precision_score(y_test,test_pred_knn), 3),  
        round(precision_score(y_test,test_pred_log), 3),  
        round(precision_score(y_test,test_pred_rf), 3),  
        round(precision_score(y_test,test_pred_gaussian), 3),  
        round(precision_score(y_test,test_pred_sgd), 3),  
        round(precision_score(y_test,test_pred_svc), 3),  
        round(precision_score(y_test,test_pred_dt), 3),  
        round(precision_score(y_test,test_pred_gbt), 3),  
        round(precision_score(y_test,test_pred_adb), 3),  
        round(precision_score(y_test,test_pred_vot), 3),    
    ],  
    'recall': [  
        round(recall_score(y_test,test_pred_knn), 3),  
        round(recall_score(y_test,test_pred_log), 3),  
        round(recall_score(y_test,test_pred_rf), 3),  
        round(recall_score(y_test,test_pred_gaussian), 3),  
        round(recall_score(y_test,test_pred_sgd), 3),  
        round(recall_score(y_test,test_pred_svc), 3),  
        round(recall_score(y_test,test_pred_dt), 3),  
        round(recall_score(y_test,test_pred_gbt), 3),  
        round(recall_score(y_test,test_pred_adb), 3),  
        round(recall_score(y_test,test_pred_vot), 3),    
    ],  
    'F1': [  
        round(f1_score(y_test,test_pred_knn,average='binary'), 3),  
        round(f1_score(y_test,test_pred_log,average='binary'), 3),  
        round(f1_score(y_test,test_pred_rf,average='binary'), 3),  
        round(f1_score(y_test,test_pred_gaussian,average='binary'), 3),  
        round(f1_score(y_test,test_pred_sgd,average='binary'), 3),  
        round(f1_score(y_test,test_pred_svc,average='binary'), 3),  
        round(f1_score(y_test,test_pred_dt,average='binary'), 3),  
        round(f1_score(y_test,test_pred_gbt,average='binary'), 3),  
        round(f1_score(y_test,test_pred_adb,average='binary'), 3),  
        round(f1_score(y_test,test_pred_vot,average='binary'), 3),      
    ],  
})  
models.sort_values(by='Acc', ascending=False)  


```

<img width="510" alt="image" src="https://github.com/rud15dns/aix_project/assets/90837976/9d994548-47db-4fd0-aec2-e0be4a4995c4">

> 본 훈련에 사용된 10개 모델 중 랜덤 포레스트(RandomForest),경사도 상승 결정 트리(Gradient Boosting Trees), 에이다부스트(AdaBoost)의 3개 모델의 정확도가 85% 이상으로 양호함을 알 수 있으며 F1 값도 다른 모델에 비해 우수한 수준이기 때문에 지표의 관점에서 이 세 모델이 본 논문에서 연구한 문제에 더 적합하다고 판단하였습니다.

<br/>

### [2] ROC 곡선 시각화 
<br/>

- 각 모델에 대해 ROC 곡선을 그려 여러 종류의 분류 모델들의 성능을 시각적으로 비교 할 수 있습니다.

```ruby
plt.style.use('seaborn-whitegrid')  
fig = plt.figure(figsize=(10,10))   
models = [  
    'KNN',   
    'Logistic Regression',   
    'Random Forest',   
    'Naive Bayes',   
    'Decision Tree',   
    'Gradient Boosting Trees',  
    'AdaBoost',  
    'Linear SVC',  
    'Voting',  
    'Stochastic Gradient Decent'  
]  
probs = [  
    probs_knn,  
    probs_log,  
    probs_rf,  
    probs_gau,  
    probs_dt,  
    probs_gbt,  
    probs_adb,  
    probs_svc,  
    probs_vot,  
    probs_sgd  
]  
colormap = plt.cm.tab10 #nipy_spectral, Set1, Paired, tab10, gist_ncar  
colors = [colormap(i) for i in np.linspace(0, 1,len(models))]  
plt.title('Receiver Operating Characteristic')  
plt.plot([0, 1], [0, 1],'r--')  
plt.xlim([-0.01, 1.01])  
plt.ylim([-0.01, 1.01])  
plt.ylabel('True Positive Rate')  
plt.xlabel('False Positive Rate')  
def plot_roc_curves(y_test, prob, model):  
    fpr, tpr, threshold = metrics.roc_curve(y_test, prob)  
    roc_auc = metrics.auc(fpr, tpr)  
    label = model + ' AUC = %0.2f' % roc_auc  
    plt.plot(fpr, tpr, 'b', label=label, color=colors[i])  
    plt.legend(loc = 'lower right')    
for i, model in list(enumerate(models)):  
    plot_roc_curves(y_test, probs[i], models[i])


```
![image](https://github.com/rud15dns/aix_project/assets/90837976/411f5d56-b5eb-4400-a904-c168e9d5cb9c)

<br/>


### [3] Precision-Recall 곡선 시각화
- Precision-Recall 곡선을 통해서도 다양한 분류 모델의 성능을 시각적으로 비교 할 수 있습니다.

- ![image](https://github.com/rud15dns/aix_project/assets/90837976/82da2d30-1275-48e7-b231-fc1f1d267e32)


> 세 가지 성능 평가 방법 모두에서 10개의 모델 중 랜덤 포레스트(RandomForest), 경사하강 결정 트리(Gradient Boosting Trees), 에이다부스트(AdaBoost) 모델이 가장 우수한 것으로 평가되었습니다.
<br/><br/>

## V. Related Work

- https://www.kaggle.com/datasets/uciml/adult-census-income
- https://dacon.io/competitions/official/235892/talkboard
- https://scikit-learn.org/stable/supervised_learning.html
- https://github.com/search?q=census+income+prediction&type=repositories
- https://www.kaggle.com/code/humagonen/adult-income-data-cleaning-eda

<br/><br/>
## VI. Conclusion: Discussion
- 이번 프로젝트를 통해 다양한 머신러닝 알고리즘을 탐구하고 활용할 수 있었으며, 모델을 학습시키는 과정을 다시 한번 정확하게 익히는 계기가 되었습니다.
-  인구센서스 데이터를 활용하여 소득 예측 모델을 구축과정에서 다양한 머신러닝 알고리즘을 적용한 결과, 랜덤 포레스트(RandomForest), 경사하강 결정 트리(Gradient Boosting Trees), 에이다부스트(AdaBoost) 모델이 가장 우수한 성능을 보였습니다. 이 모델들은 높은 정확도와 균형 잡힌 정밀도, 재현율, F1 점수를 기록하여 데이터의 복잡한 패턴을 효과적으로 학습할 수 있음을 입증했습니다. 또한, ROC 곡선과  Precision-Recall 곡선을 통해 시각적으로 모델 성능을 비교함으로써 각 모델의 강점을 명확하게 파악할 수 있었습니다. 이러한 결과는 인구센서스 데이터를 기반으로 한 소득 예측이 다양한 실생활 응용 분야에서 유용하게 활용될 수 있음을 시사하는 것이라고 생각합니다.
- 향후 기회가 된다면 모델을 더 쉽게 이해할 수 있도록 하고, 추가적인 특징 추출과 하이퍼파라미터 조정을 통해 성능을 더욱 향상시킨 모델을 구축하고 싶습니다.  <br/>
<br/>

