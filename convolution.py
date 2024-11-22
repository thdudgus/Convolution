import cv2
import numpy as np
import matplotlib.pyplot as plt

# 기준 도형들을 Matplotlib로 시각화 (확인용)
def displayOriginalIcon():
    plt.figure(figsize=(12, 6))  # 그래프 크기 설정

    # 첫 번째 도형 (사각형)
    plt.subplot(1, 3, 1)  # 한 행에 3개의 그래프, 첫 번째 위치
    plt.title("Icon 1 (Square)")
    plt.imshow(icon1, cmap='brg')
    plt.axis('off')  # 축 제거

    # 두 번째 도형 (삼각형)
    plt.subplot(1, 3, 2)  # 두 번째 위치
    plt.title("Icon 2 (Triangle)")
    plt.imshow(icon2, cmap='grey')
    plt.axis('off')

    # 세 번째 도형 
    plt.subplot(1, 3, 3)  # 세 번째 위치
    plt.title("Icon 3 (Circle)")
    plt.imshow(icon3, cmap='brg')
    plt.axis('off')

    # 그래프 표시
    plt.tight_layout()  # 간격 자동 조정
    plt.show()

# 1/2 아이콘들을 Matplotlib로 시각화 (확인용)
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

# 평균 좌표 계산
def calculate_average_coordinates(locations, radius, min_count):
    # 입력: locations (np.array of [y_coords, x_coords]), 반경 radius, 최소 좌표 개수
    coords = np.column_stack(locations[::-1])  # (x, y) 좌표로 변환
    visited = set()
    clusters = []

    for i, pt in enumerate(coords):
        if i in visited:  # 이미 처리된 점은 스킵
            continue

        # 반경 내의 점 찾기
        cluster = [pt]
        visited.add(i)
        for j, other_pt in enumerate(coords):
            if j in visited:
                continue
            distance = np.sqrt((pt[0] - other_pt[0]) ** 2 + (pt[1] - other_pt[1]) ** 2)
            if distance <= radius:
                cluster.append(other_pt)
                visited.add(j)
        
        # 군집의 좌표 개수가 min_count 이상인 경우만 추가
        if len(cluster) >= min_count:
            clusters.append(cluster)
    
    # 각 클러스터의 평균 좌표 계산
    average_coords = [np.mean(cluster, axis=0) for cluster in clusters]
    return np.array(average_coords)

# 평균 좌표 기준 아이콘을 감싸는 사각형 그리기
def draw_average_locations(image, averages, color=(0, 255, 255)):
    for coord in averages:
        center = (int(coord[0]), int(coord[1]))  # 좌표를 정수로 변환
        size = 120  # 사각형 크기 설정
        top_left = (center[0] - size // 2 +30, center[1] - size // 2 +30)
        bottom_right = (center[0] + size // 2 +30, center[1] + size // 2 +30)
        cv2.rectangle(image, top_left, bottom_right, color, 2)

# 아이콘 매칭
def icon_matching(image, template):
    img_h, img_w = image.shape
    tpl_h, tpl_w = template.shape

    # 결과 배열 초기화
    result = np.zeros((img_h - tpl_h + 1, img_w - tpl_w + 1), dtype=np.float32)

    template = np.flipud(np.fliplr(template))

    # 템플릿 정규화
    template_mean = template.mean()
    template_std = template.std()
    if template_std == 0:
        raise ValueError("Template has zero variance")
    normalized_template = (template - template_mean) / template_std

    # 합성곱 기반 계산
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            # 현재 위치의 이미지 패치 추출
            image_patch = image[y:y + tpl_h, x:x + tpl_w]

            # 이미지 패치 정규화
            patch_mean = image_patch.mean()
            patch_std = image_patch.std()
            if patch_std == 0:
                result[y, x] = 0
                continue
            normalized_patch = (image_patch - patch_mean) / patch_std

            # 정규화된 템플릿과 패치의 합성곱
            numerator = np.sum(normalized_patch * normalized_template)
            denominator = tpl_h * tpl_w  # 템플릿 크기
            result[y, x] = numerator / (denominator + 1e-8)  # 작은 값을 더해 분모 0 방지

    # 결과 값 정규화 (0 ~ 1)
    result_min = result.min()
    result_max = result.max()
    result = (result - result_min) / (result_max - result_min + 1e-8)

    print("Result max (after normalization):", result.max())
    return result

# -----------------------------------------------------------------------------------------------------------------------

# 1. 원본 이미지 로드
original_img = cv2.imread("images.jpg", cv2.IMREAD_GRAYSCALE)


# 2. 기준 도형 아이콘 생성 (가로/세로 축소)
# 도형 아이콘 영역 설정 (세로 범위, 가로 범위)
icon1 = original_img[76:180, 44:147]  # 첫 번째 도형 (예: 사각형) 116*109
icon2 = original_img[213:319, 41:154] # 두 번째 도형 (예: 삼각형)
# icon3 = original_img[668:780, 46:155] # 세 번째 도형 (예: 오각형)
# icon3 = original_img[519:624, 38:159] # 세 번째 도형 (예: 하트)
icon3 = original_img[362:468, 41:148] # 세 번째 도형 (예: 원)

# 기준 도형들을 Matplotlib로 시각화 (확인용)
# displayOriginalIcon()

# 축소된 아이콘 생성 (1/2 크기로 축소)
icon1_resized = cv2.resize(icon1, (icon1.shape[1] // 2, icon1.shape[0] // 2))
icon2_resized = cv2.resize(icon2, (icon2.shape[1] // 2, icon2.shape[0] // 2))
icon3_resized = cv2.resize(icon3, (icon3.shape[1] // 2, icon3.shape[0] // 2))

# 1/2 도형들을 Matplotlib로 시각화 (확인용)
# displayResizedIcon()


# 3. 합성곱 기반 매칭
w = 0.8

flipped_template = np.flipud(np.fliplr(icon1_resized))
result1 = icon_matching(original_img, flipped_template)
locations1 = np.where(result1 >= w)

flipped_template = np.flipud(np.fliplr(icon2_resized))
result2 = icon_matching(original_img, flipped_template)
locations2 = np.where(result2 >= w+0.05)

flipped_template = np.flipud(np.fliplr(icon3_resized))
result3 = icon_matching(original_img, flipped_template)
locations3 = np.where(result3 >= w-0.05)


# 4. 매칭된 위치에 박스 그리기
# 원본 이미지 복사본 생성
output_img = original_img.copy()
output_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

# ㅜㅜㅜㅜㅜㅜㅜ 매칭 결과에 대한 군집 계산 ㅜㅜㅜㅜㅜㅜㅜㅜㅜㅜㅜㅜㅜ
# 반경 및 최소 좌표 개수 설정 : 민감도 조절 가능.
radius = 70 # 
min_count = 15  # 최소 좌표 개수 (최소 클러스터 크기) 

# 군집 계산
average_locations1 = calculate_average_coordinates(locations1, radius, min_count)
average_locations2 = calculate_average_coordinates(locations2, radius, min_count)
average_locations3 = calculate_average_coordinates(locations3, radius, min_count*4)

# 원본 이미지에 평균 좌표 표시
output_img_with_averages = output_img.copy()
draw_average_locations(output_img_with_averages, average_locations1, color=(0, 0, 255))  # 사각형
draw_average_locations(output_img_with_averages, average_locations2, color=(255, 0, 0))  # 삼각형
draw_average_locations(output_img_with_averages, average_locations3, color=(0, 255, 0))  # 원
# ㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗ

# 사각형 매칭 결과 표시 (파란색)
for pt in zip(*locations1[::-1]):
    cv2.rectangle(output_img, pt, (pt[0] + icon1_resized.shape[1], pt[1] + icon1_resized.shape[0]), (0, 0, 255), 2)

# 삼각형 매칭 결과 표시 (빨간색)
for pt in zip(*locations2[::-1]):
    cv2.rectangle(output_img, pt, (pt[0] + icon2_resized.shape[1], pt[1] + icon2_resized.shape[0]), (255, 0, 0), 2)

# 매칭 결과 표시 (초록색)
for pt in zip(*locations3[::-1]):
    cv2.rectangle(output_img, pt, (pt[0] + icon3_resized.shape[1], pt[1] + icon3_resized.shape[0]), (0, 255, 0), 2)


# # 5. 결과 출력
# 중간 결과 출력
"""
plt.figure(figsize=(8, 5)) 
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_img, cmap='grey')
plt.subplot(1, 2, 2)
plt.title("Detected Matches")
plt.imshow(output_img, cmap='grey')
plt.show()
"""

# 최종 결과 출력
plt.figure(figsize=(8, 5))
plt.title("Detected Matches with Clustered Rectangles")
plt.imshow(output_img_with_averages)
plt.show()