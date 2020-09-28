# :star2: bigcon_2020_mmmz :star2:

## :raising_hand: Authors
- **Seungmin, Seo**  
  Master's degree in Statistics & Data Science, Yonsei University, Seoul, South Korea
- **Seungmi, Oh**    
  Master's degree in Statistics & Data Science, Yonsei University, Seoul, South Korea
- **Sojeong, Ahn**   
  Bachelor's degree in Korean Literature, Yonsei University, Seoul, South Korea
- **Kyungmin, Cho**  
  Master's degree in Statistics & Data Science, Yonsei University, Seoul, South Korea
- **Yujin, Jung**   
  Master's degree in Statistics & Data Science, Yonsei University, Seoul, South Korea

## :open_file_folder: File Summary

### 1. data 
   > 데이터 일체

- 00 : raw data
- 01 : raw data에 original_c, small_c, middle_c, big_c 작업 마친 상태
- 11 : 외부 변수 생성에 필요한 데이터(날씨, 클릭률 등)
- 12 : dummy 변수 생성에 필요한 데이터
- 13 : 최적화 모델링 및 counterfactual analysis 관련 데이터
- 20 : preprocess 과정을 거친 데이터
- saved_models: 훈련된 모델(bin type)

### 2. eda 
    > eda 파일/클릭률 크롤링/계절성

- eda.ipynb : eda 과정 일체
- naver_shopping_crawling.ipynb : 네이버 쇼핑 트렌드의 클릭률 데이터 crawling
- naver_clickr_crawl.R : 클릭률 crawled data 통합
- seasonal.ipynb : 계절성 확인하기 위한 eda

### 3. engine 
    > feature engineering/train/predict/residual analysis /기타 변수 정의

- features.py : feature engineering 과정
- predict.py : train.py를 통해 훈련된 모델 weight을 불러와서 2020년 6월 매출 예측
- train.py : preprocess + 모델 훈련 + cross validation
- utils.py : preprocess + helper + data split에 필요한 함수 모음
- vars.py : 자주 사용하는 변수 모음

### 4. opt
    > 최적화 모델/counterfactual 관련 코드

- counterfactual.py : counterfactual analysis 관련 코드
- inputs.py : 헝가리안 최적화 알고리즘을 위한 input 생성
- opt_model.py : 헝가리안 최적화 적용 코드

### 5. submission 
- submission.xlsx : 2020년 6월 편성표 + predicted y


