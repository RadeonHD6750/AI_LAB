#######################################################
"""
    서지민에 인공신경망을 이용한 기상예측 프로젝트

"""
########################################################


# 각종 필요한 것들 가져오기
#기본 배열 다루기 Numpy
import numpy as np

#텐서플로우 가져오기
import tensorflow as tf

#데이터 다루기
import pandas as pd

#케라스 가져오기
from keras.models import Sequential # 신경망 모델 생성자함수
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from keras.models import load_model # 신경망 모델 파일 불러오기
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping # 학습조기종료

from sklearn.model_selection import train_test_split # 학습용 따로 검증용 따로


# 일단 데이터부터 가져온다.
# # 기온,강수량,풍속,습도,이슬점,현지기압,시정
#
# # 습도 = 습도 * 0.01
# # 기압 = 기압 - 1000
# # 시정 = 시정 * 0.001

#########################################################################
#  학습 데이터 불러오기
#########################################################################
def load_Data_except():
    answear = input("입력데이터 불러오기 >");

    try:
        Train_Set = pd.read_csv(answear,
                                  names=['기온', '강수량', '풍속', '습도', '이슬점', '기압', '시정']);
    except:
        print("파일열기 실패\n");
        Train_Set = None;

    return Train_Set;


#입력값 불러오기
X_Train_Set_Load = load_Data_except();

while(X_Train_Set_Load is None):
    X_Train_Set_Load = load_Data_except();

print(X_Train_Set_Load.head(5));
print("불러오기 완료\n\n");

#목표값 불러오기
Y_Train_Set_Load = load_Data_except();

while(Y_Train_Set_Load is None):
    Y_Train_Set_Load = load_Data_except();
print(Y_Train_Set_Load.head(5));
print("불러오기 완료\n\n");


X_Train_Set = X_Train_Set_Load;
Y_Train_Set = Y_Train_Set_Load;
#학습용 따로 검증용 따로
"""
X_Train_Set, X_Test_Set, Y_Train_Set, Y_Test_Set = train_test_split(X_Train_Set_Load,
                                                                    Y_Train_Set_Load,
                                                                    test_size=0.25);
"""

# 데이터 가공하기
# 아 그냥 데이터단계에서 해결ㅋㅋㅋㅋㅋㅋ 엑셀 굳


#########################################################################
#  이제 여기서부터는 신경망 만들기
#########################################################################
# 모델 구조 설계
Weather_Forecast_Model = Sequential();

# 최초 입력층


# 은닉층 이름이 달라야 추가된다. 같은 것을 계속 추가할 수는 없다.
# 중간에 ReLU층이 어느 정도 있어야 한다.
Hidden_Layer1 = Dense(14, input_dim=7,activation='linear');
Weather_Forecast_Model.add(Hidden_Layer1);

Hidden_Layer2 = Dense(14, activation='linear');
Weather_Forecast_Model.add(Hidden_Layer2);

Hidden_Layer3 = Dense(7, activation='linear');
Weather_Forecast_Model.add(Hidden_Layer3);

Hidden_Layer4 = Dense(7, activation='linear');
Weather_Forecast_Model.add(Hidden_Layer4);


Hidden_Layer5 = Dense(7, activation='linear');
Weather_Forecast_Model.add(Hidden_Layer5);

Hidden_Layer6 = Dense(7, activation='linear');
Weather_Forecast_Model.add(Hidden_Layer6);

Hidden_Layer7 = Dense(7, activation='linear');
Weather_Forecast_Model.add(Hidden_Layer7);


#########################################################################
#  환경설정 및 학습시작
#########################################################################

# 모델 환경설정
Weather_Forecast_Model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy']);

# 모델 학습하기

# 자동종료 함수
early_stopping = EarlyStopping(monitor='loss', patience=100);
Weather_Forecast_Model.fit(X_Train_Set,Y_Train_Set, epochs=500, batch_size=500, callbacks=[early_stopping]);
#Weather_Forecast_Model.fit(X_Train_Set,Y_Train_Set, epochs=200, batch_size=500);
print("\n 정확도: %.4f" % (Weather_Forecast_Model.evaluate(X_Train_Set,Y_Train_Set)[1]));

Test_Forecast_List = Weather_Forecast_Model.predict(X_Train_Set.head(5));

for i in range(5):
    print(Test_Forecast_List[i]);


#########################################################################
#  신경망 저장하기
#########################################################################

answear = input("\n\n현재 수치예보모델을 저장하기 >");

# 모델 저장하기
Weather_Forecast_Model.save(answear);
Weather_Forecast_Model.summary();  # 이게 있어야 불러올 때 잘 온다.
print("\n저장완료");