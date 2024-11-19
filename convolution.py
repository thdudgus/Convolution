import cv2
import numpy as np
import matplotlib.pyplot as plt

# 기준 도형들을 Matplotlib로 시각화 (확인용)
def displayOriginalIcon():
    plt.figure(figsize=(12, 6))  # 그래프 크기 설정

    # 첫 번째 도형 (사각형)
    plt.subplot(1, 3, 1)  # 한 행에 3개의 그래프, 첫 번째 위치
    plt.title("Icon 1 (Square)")
    plt.imshow(icon1, cmap='grey')
    plt.axis('off')  # 축 제거

    # 두 번째 도형 (삼각형)
    plt.subplot(1, 3, 2)  # 두 번째 위치
    plt.title("Icon 2 (Triangle)")
    plt.imshow(icon2, cmap='grey')
    plt.axis('off')

    # 세 번째 도형 (원)
    plt.subplot(1, 3, 3)  # 세 번째 위치
    plt.title("Icon 3 (Circle)")
    plt.imshow(icon3, cmap='brg')
    plt.axis('off')

    # 그래프 표시
    plt.tight_layout()  # 간격 자동 조정
    plt.show()

# 기준 도형들을 Matplotlib로 시각화 (확인용)
def displayResizedIcon():
    plt.figure(figsize=(12, 4))  # 그래프 크기 설정

    # 첫 번째 도형 (사각형)
    plt.subplot(1, 3, 1)  # 한 행에 3개의 그래프, 첫 번째 위치
    plt.title("icon1_resized (Square)")
    plt.imshow(icon1_resized, cmap='grey')
    plt.axis('off')  # 축 제거

    # 두 번째 도형 (삼각형)
    plt.subplot(1, 3, 2)  # 두 번째 위치
    plt.title("icon2_resized (Triangle)")
    plt.imshow(icon2_resized, cmap='grey')
    plt.axis('off')

    # 세 번째 도형 (원)
    plt.subplot(1, 3, 3)  # 세 번째 위치
    plt.title("icon3_resized (Circle)")
    plt.imshow(icon3_resized, cmap='grey')
    plt.axis('off')

    # 그래프 표시
    plt.tight_layout()  # 간격 자동 조정
    plt.show()


# 1. 원본 이미지 로드
original_img = cv2.imread("images.jpg", cv2.IMREAD_GRAYSCALE)

# 2. 기준 도형 아이콘 생성 (가로/세로 축소)
# 도형 아이콘 영역 설정 (세로 범위, 가로 범위)
icon1 = original_img[75:181, 41:150]  # 첫 번째 도형 (예: 사각형) 116*109
icon2 = original_img[213:319, 41:154] # 두 번째 도형 (예: 삼각형)
icon3 = original_img[668:780, 46:155] # 세 번째 도형 (예: 원)
# icon3 = original_img[520:624, 38:159] # 세 번째 도형 (예: 하트)

# 기준 도형들을 Matplotlib로 시각화 (확인용)
# displayOriginalIcon()

# 축소된 아이콘 생성 (1/2 크기로 축소)
icon1_resized = cv2.resize(icon1, (icon1.shape[1] // 2, icon1.shape[0] // 2))
icon2_resized = cv2.resize(icon2, (icon2.shape[1] // 2, icon2.shape[0] // 2))
icon3_resized = cv2.resize(icon3, (icon3.shape[1] // 2, icon3.shape[0] // 2))

# 1/2 도형들을 Matplotlib로 시각화 (확인용)
#displayResizedIcon()

w = 0.5
# 3. 합성곱 기반 매칭
# 첫 번째 도형 매칭
result1 = cv2.matchTemplate(original_img, icon1_resized, cv2.TM_CCOEFF_NORMED)
locations1 = np.where(result1 >= w)  # 임계값 0.8 이상인 위치

# 두 번째 도형 매칭
result2 = cv2.matchTemplate(original_img, icon2_resized, cv2.TM_CCOEFF_NORMED)
locations2 = np.where(result2 >= w)

# 세 번째 도형 매칭
result3 = cv2.matchTemplate(original_img, icon3_resized, cv2.TM_CCOEFF_NORMED)
locations3 = np.where(result3 >= w-0.1)

# 4. 매칭된 위치에 박스 그리기
# 원본 이미지 복사본 생성
output_img = original_img.copy()
output_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

# # 사각형 매칭 결과 표시 (파란색)
for pt in zip(*locations1[::-1]):
    cv2.rectangle(output_img, pt, (pt[0] + icon1_resized.shape[1], pt[1] + icon1_resized.shape[0]), (0, 0, 255), 2)

# 삼각형 매칭 결과 표시 (빨간색)
for pt in zip(*locations2[::-1]):
    cv2.rectangle(output_img, pt, (pt[0] + icon2_resized.shape[1], pt[1] + icon2_resized.shape[0]), (255, 0, 0), 2)

# 오각형 매칭 결과 표시 (초록색)
for pt in zip(*locations3[::-1]):
    cv2.rectangle(output_img, pt, (pt[0] + icon3_resized.shape[1], pt[1] + icon3_resized.shape[0]), (0, 255, 0), 2)

# 5. 결과 시각화
plt.figure(figsize=(8, 5)) 
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_img, cmap='grey')
plt.subplot(1, 2, 2)
plt.title("Detected Matches")
plt.imshow(output_img, cmap='grey')
plt.show()