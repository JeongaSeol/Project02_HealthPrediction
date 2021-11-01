# Project02_HealthPrediction
데이터사이언스 교육 프로젝트2 - 혈압,혈당 예측모델 생성 및 분석

## DataSet
- 국가건강검진.csv 파일 사용 
  - 성별, 연령, 신장, 체중, 허리둘레, bmi(파생변수), absi(파생변수)로 각각 수축기혈압/이완기혈압/혈당 수치를 예측


## Modeling
- Linear Regression / Polynomial Regression / Decision Tree / RandomForest / GBM / LGBM / XGBM / MLPRegressor 등 다양한 모델로 training & validation 수행
- 각 모델에 해당하는 HyperParameter들을 변경하면서 RMSE가 최소인 모델을 찾음
- PJT_모델_분석.hwp 파일에서 확인할 수 있음


## WebProgramming(Django) 
- index.html에서 사용자가 성별, 연령, 신장 등 필요한 x변수들을 입력하면,
- view.py에서 학습시킨 모델에 predict()적용. 
- 예측한 혈압, 혈당 수치를 정상/주의/위험 세 단계를 각각 초록/노랑/빨강 색으로 표현하는 서비스 구현
- 입력받은 연령과 예측한 혈압,혈당 수치를 가지고 맞춤 영양제 추천 서비스 추가 구현 ( Business Insight 도출)
