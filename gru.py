# 출처: https://welldonecode.tistory.com/97 (수정됨)

import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.losses import mse
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# TensorFlow 로그 메시지 숨기기
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 데이터 로드 및 전처리
agg_10min = pd.read_csv('aggregation_10min.csv', engine='python')
agg_10min = agg_10min.drop(['Unnamed: 0', 'timestamp'], axis=1)

# 데이터 스케일링 (Min-Max 스케일링)
scaler = MinMaxScaler()
agg_10min[['num_gpu', 'gpu_milli']] = scaler.fit_transform(agg_10min[['num_gpu', 'gpu_milli']])

# 슬라이딩 윈도우 함수 정의 (다중 feature 입력 지원)
def create_sliding_window_data(data, lookback_time=5, predict_time=2):
    X, y = [], []
    for i in range(len(data) - (lookback_time - 1) - predict_time):
        x_window = data.iloc[i:i+lookback_time][['num_gpu', 'gpu_milli']].values
        y_value = data.iloc[i+lookback_time+predict_time-1]['gpu_milli']
        X.append(x_window)
        y.append(y_value)
    return np.array(X), np.array(y)

# 입력 데이터 생성
x, t = create_sliding_window_data(agg_10min)

# 데이터 분할 (학습 및 테스트)
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, shuffle=False)

# GRU 모델 생성
cell_size = 256
timesteps = 5
feature = 2  # num_gpu, gpu_milli 두 개

model = Sequential(name="GPU_GRU")
model.add(GRU(cell_size, input_shape=(timesteps, feature), return_sequences=True))
model.add(GRU(cell_size))
model.add(Dense(1))

model.compile(loss=mse, optimizer=Adam(), metrics=['mae'])
model.summary()

# 모델 학습
start = datetime.datetime.now()
history = model.fit(x_train, t_train, epochs=400, batch_size=170, validation_data=(x_test, t_test), verbose=1)
end = datetime.datetime.now()

# 예측 수행
y_pred = model.predict(x_test)

# 역정규화
t_test_reset = scaler.inverse_transform(np.concatenate([np.zeros((len(t_test), 1)), t_test.reshape(-1, 1)], axis=1))[:, 1]
y_pred_reset = scaler.inverse_transform(np.concatenate([np.zeros((len(y_pred), 1)), y_pred], axis=1))[:, 1]

# 평가 지표 계산
mse_obj = MeanSquaredError()
mse_obj.update_state(t_test_reset, y_pred_reset)
rmse_reset = np.sqrt(mse_obj.result())
mae_reset = MeanAbsoluteError()
mae_reset.update_state(t_test_reset, y_pred_reset)
mape_reset = np.mean(np.abs((t_test_reset - y_pred_reset) / t_test_reset)) * 100

print('=== Decode Data (역정규화된 예측값 기준) ===')
print('Test RMSE:', rmse_reset)
print('Test MAPE:', mape_reset)

print('\n학습 시간:', end - start)

# 예측 결과 시각화 (스케일 복원 전/후)
plt.figure(figsize=(12, 6))
plt.plot(t_test, label='Actual (Scaled)', color='red')
plt.plot(y_pred, label='Predicted (Scaled)', color='blue', linestyle='--')
plt.title('GRU Prediction (Scaled)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t_test_reset, label='Actual', color='red')
plt.plot(y_pred_reset, label='Predicted', color='blue', linestyle='--')
plt.title('GRU Prediction (Original Scale)')
plt.legend()
plt.show()

np.save('t_test', t_test)
np.save('y_pred', y_pred)
np.save('t_test_reset', t_test_reset)
np.save('y_pred_reset', y_pred_reset)