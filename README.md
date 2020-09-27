# bigcon_2020_mmmz

## File Summary

#### 1. data 
> 데이터 일체

- 00 : raw data
- 01 : raw data에 original_c, small_c, middle_c, big_c 작업 마친 상태
- 11 : 외부 변수 생성에 필요한 데이터(날씨, 클릭률 등)
- 12 : dummy 변수 생성에 필요한 데이터
- 13 : 최적화 모델링 및 counterfactual analysis 관련 데이터
- 20 : preprocess 과정을 거친 데이터
- saved_models: 훈련된 모델(bin type)

#### 2. eda 
> eda 파일/클릭률 크롤링/계절성

- eda.ipynb : eda 과정 일체

- naver_shopping_crawling.ipynb : 네이버 쇼핑 트렌드의 클릭률 데이터 crawling

- naver_clickr_crawl.R : 클릭률 crawled data 통합

- seasonal.ipynb : 계절성 확인하기 위한 eda

#### 3. engine 
> feature engineering/train/predict/residual analysis /기타 변수 정의

- features.py : feature engineering 과정
- predict.py : train.py를 통해 훈련된 모델 weight을 불러와서 2020년 6월 매출 예측
- train.py : preprocess + 모델 훈련 + cross validation
- utils.py : preprocess + helper + data split에 필요한 함수 모음
- vars.py : 자주 사용하는 변수 모음

#### 4. opt
> 최적화 모델/counterfactual 관련 코드

- counterfactual.py : counterfactual analysis 관련 코드
- inputs.py : 헝가리안 최적화 알고리즘을 위한 input 생성
- opt_model.py : 헝가리안 최적화 적용 코드

#### 5. submission 
- submission.xlsx : 2020년 6월 편성표 + predicted y



### Variables (2020.08.28 / 37)

name | descript | tyoe 
---- | ---- | ---- 
index | index | index 
방송일시 | raw | object 
노출.분. | raw | object
마더코드 | raw | object
상품코드 | raw | object
상품명 | raw | object
상품군 | raw | object
판매단가 | raw | float
취급액 | raw | float
selling_price | raw(eng) | float
sales | raw(eng) | float
exposed | raw(eng) | float
volume | sales/selling_price | float
month | 월 | object(12)
day | 일 | object(31)
weekday | 요일 | object(7)
holiday | red+weekend | object(2)
red | 공휴일 | object(2)
weekend | 주말 | object(2)
hours | 시간 | object(24)
hours_inweek | 주 단위 시간 | object(168)
**primetime** | 1 = 오전 프라임, 2 = 오후 프라임 | object(3)
**japp** | 30분에서 반올림 | object()
**min_start** | 방송 시작시간(분) | object(8)
**min_range** | 방송 길이(분) | object()
**parttime** | 한 타임 내 방송순서 | object()
**show_id** | 해당 일 방송회차 | object()
**sales_power** | 시간 대비 판매량(?) | float
**men** | 성별 가지는 상품(null) | object(2)
**hottest** | 탑10 포함여부 | object(2)
**pay** | 지불방식 | object(2)
**luxury** | 명품 단가 50만 이상 | object(2)
**brandpower** | 업데이트 예정 | -
**season_item** | 업데이트 예정 | -
**naver_crawl** | 업데이트 예정 | -
**weather** | 업데이트 예정 | -
original_c | naver | object
small_c | naver | object
middle_c | naver(null) | object
big_c | naver(null) | object
brand | 브랜드(null) | object

