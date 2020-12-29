# Acoustic-Scene-Classification
  
## 소개
- Acoustic scene classification은 DCASE 2018 Task5 dataset을 사용하여 학습한 모델을 이용하여 음향 환경을 인지하는 프로그램입니다.
  
## 사전작업
- Python3 환경에서 작동합니다.
- Focusrite Control을 사용합니다.
  
## 사용법
1. feature_extraction.py 파일을 실행하여 특징추출하여 hdf5파일로 저장합니다.
1. model.py 파일을 실행하고 특징추출한 hdf5파일을 읽어와서 데이터를 학습시킨 뒤 최적 모델을 h5파일로 저장합니다.
1. ASC_demo 폴더의 main.py 파일을 실행하고 최적 모델을 사용하여 예측 결과를 확인합니다.

