import pandas as pd
import chardet
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 한글 폰트 설정 (윈도우에서는 'malgun.ttf'를 기본 폰트로 사용)
font_path = "C:/Windows/Fonts/malgun.ttf"  # 윈도우의 기본 한글 폰트
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# 데이터 로드 및 인코딩 감지
with open('chemical_leakage_data.csv', 'rb') as file:
    result = chardet.detect(file.read())
    encoding = result['encoding']

# 데이터 로드 (자동 인코딩 감지 적용)
data = pd.read_csv('chemical_leakage_data.csv', encoding=encoding)

# '발생일자' 컬럼을 datetime 형식으로 변환
data['발생일자'] = pd.to_datetime(data['발생일자'], errors='coerce')

# 날짜 데이터를 '연도', '월', '일' 등의 숫자형 값으로 변환
data['발생연도'] = data['발생일자'].dt.year
data['발생월'] = data['발생일자'].dt.month
data['발생일'] = data['발생일자'].dt.day

# '발생일자' 컬럼 제거 (불필요한 문자열 컬럼)
data = data.drop(['발생일자'], axis=1)

# '주소'와 관련된 컬럼 제거 (모델에 필요하지 않음)
data = data.drop(['사고장소소재지우편번호', '사고장소소재지지번주소', '사고장소소재지도로명주소'], axis=1)

# 범주형 데이터를 원-핫 인코딩으로 변환
data = pd.get_dummies(data, columns=['사고내용', '사고원인'], drop_first=True)

# '시군명'은 예측할 목표 변수로 설정
X = data.drop(['시군명'], axis=1)  # '시군명' 제외
y = data['시군명']  # 목표 변수

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링 (로지스틱 회귀를 위한 표준화)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 훈련 및 평가
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=2000)  # max_iter을 늘려서 수렴 문제 해결
lr_model.fit(X_train_scaled, y_train)

# 예측
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test_scaled)

# 성능 평가 (Accuracy, Precision, Recall 등)
print("Random Forest Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred, zero_division=1))

print("Logistic Regression Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Classification Report:\n", classification_report(y_test, lr_pred, zero_division=1))

# 오차 행렬
rf_cm = confusion_matrix(y_test, rf_pred)
lr_cm = confusion_matrix(y_test, lr_pred)

# 시각화: 오차 행렬
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
axes[0].set_title('랜덤 포레스트 오차 행렬')
axes[1].set_title('로지스틱 회귀 오차 행렬')

# 그래프를 PNG 파일로 저장
plt.savefig('result/confusion_matrix.png', bbox_inches='tight')  # 'result' 폴더에 저장

plt.show()

# 교차 검증
kf = KFold(n_splits=5, random_state=42, shuffle=True)

rf_cv_scores = []
lr_cv_scores = []

for train_index, test_index in kf.split(X):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

    # 데이터 스케일링 (교차 검증 데이터에 대해 표준화)
    X_train_cv_scaled = scaler.fit_transform(X_train_cv)
    X_test_cv_scaled = scaler.transform(X_test_cv)

    # 모델 훈련
    rf_model.fit(X_train_cv, y_train_cv)
    lr_model.fit(X_train_cv_scaled, y_train_cv)

    # 예측
    rf_pred_cv = rf_model.predict(X_test_cv)
    lr_pred_cv = lr_model.predict(X_test_cv_scaled)

    # 성능 평가
    rf_cv_scores.append(accuracy_score(y_test_cv, rf_pred_cv))
    lr_cv_scores.append(accuracy_score(y_test_cv, lr_pred_cv))

# 평균 성능 평가
print("Random Forest Cross-Validation Accuracy: ", np.mean(rf_cv_scores))
print("Logistic Regression Cross-Validation Accuracy: ", np.mean(lr_cv_scores))

# 결과 해석 및 결론
print("Random Forest Model Accuracy:", accuracy_score(y_test, rf_pred))
print("Logistic Regression Model Accuracy:", accuracy_score(y_test, lr_pred))

if accuracy_score(y_test, rf_pred) > accuracy_score(y_test, lr_pred):
    print("랜덤 포레스트 모델이 더 우수한 성능을 보였습니다.")
else:
    print("로지스틱 회귀 모델이 더 우수한 성능을 보였습니다.")
