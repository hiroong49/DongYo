from collections import Counter
import matplotlib.pyplot as plt
import cv2
import numpy as np


def canny(th):
    rep_edge = cv2.GaussianBlur(rep_gray, (5, 5), 0)  # 에지만 검출하기 위해 명암도 영상에서 잡음 제거
    rep_edge = cv2.Canny(rep_edge, th, th * 3, apertureSize=3) # threshold1 50, threshold2 150, apertureSize 3
    h, w = image.shape[:2]
    cv2.rectangle(rep_edge, (0, 0, w, h), 255, -1)
    color_edge = cv2.bitwise_and(rep_image, rep_image, mask=rep_edge)
    cv2.imshow("color edge", color_edge)

image = cv2.imread("images/handdrawing.jpg", cv2.IMREAD_COLOR)
if image is None: raise Exception("사진 읽기 오류")


# 색상 검출
# RGB를 기준으로 타겟 색상 설정
green = [130, 158, 0]
red = [255, 20, 15]

# 오차 20으로 설정
diff = 20

# opencv는 rgb로 계산하는 것 명심
boundaries = [([green[2], green[1]-diff, green[0]-diff],
           [green[2]+diff, green[1]+diff, green[0]+diff])]

# 스케일 큰 이미지 줄이기
scalePercent = 0.3

# dimensions 계산
width = int(image.shape[1] * scalePercent)
height = int(image.shape[0] * scalePercent)
newSize = (width, height)

# Resize the image
image = cv2.resize(image, newSize, None, None, None, cv2.INTER_AREA)

# 각 범위는 영역 리스트 안에 포함
for (lower, upper) in boundaries:

    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    mask = cv2.inRange(image, lower, upper)

    # 이진화 마스크 처리
    cv2.imshow("binary mask", mask)
    cv2.waitKey(0)

    # 마스크랑 이미지 and 연산 수행하기
    # 하얀 부분은 살아 남고 검은 픽셀은 검게 남음
    output = cv2.bitwise_and(image, image, mask=mask)

    # and연산 마스크
    cv2.imshow("ANDed mask", output)
    cv2.waitKey(0)

    # 하얀 픽셀 개수로 색상 비율 계산
    # 하얀 픽셀 개수를 이미지 사이즈와 나눠 퍼센테이지를 구한다
    ratio_green = cv2.countNonZero(mask)/(image.size/3)

    # 색상 비율 계산
    colorPercent = (ratio_green * 100) / scalePercent

    # 색상 비율 출력
    print('green pixel percentage:', np.round(colorPercent, 2), '%')

    cv2.imshow("images", np.hstack([image, output]))
    cv2.waitKey(0)


th = 50 # threshold
rep_image = cv2.repeat(image, 1, 2)
rep_gray = cv2.cvtColor(rep_image, cv2.COLOR_BGR2GRAY)
canny(th)
cv2.namedWindow("color edge", cv2.WINDOW_AUTOSIZE)
cv2.waitKey(0)

