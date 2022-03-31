import matplotlib.pyplot as plt
import numpy as np
import cv2

# 이거는 가중치로 계산하려고 할 때 쓰려던 그래프
xpos = np.arange(4)
ypos = np.arange(14)
X = ['60', '120', '180', '240']
Y = ['C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5', 'C6', 'D6', 'E6', 'F6', 'G6', 'A6', 'B6']

# 이미지 읽어오기
src = cv2.imread("images/black_handdrawing.png", cv2.IMREAD_COLOR)
# 이진화 시키기
gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

# 가우시안 블러와 캐니 에지 적용
blur = cv2.GaussianBlur(gray, ksize=(3,3), sigmaX=0)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
edged = cv2.Canny(blur, 10, 250)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

# 윤곽선 검출
contours, hierachy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

# 윤곽선 검출 출력
#print("contours = ", contours)

contours_img = cv2.drawContours(closed, [contours[0]], -1, (0, 0, 255), 2)
cv2.imshow('edged', contours_img)

contours_xy = np.array(contours, dtype=object)
#contours_xy.shape

# x의 min과 max 찾기
x_min, x_max = 0, 0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
        x_min = min(value)
        x_max = max(value)
print("x 최솟값: ", x_min)
print("x 최댓값: ", x_max)

x_avg = (x_min + x_max) / 2
print("x 평균값: ", x_avg)

# y의 min과 max 찾기
y_min, y_max = 0, 0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
        y_min = min(value)
        y_max = max(value)
print("y 최소값: ", y_min)
print("y 최댓값: ", y_max)

y_avg = (y_min + y_max) / 2
print("y 평균값: ",   y_avg)

# x_min 계이름 찾기
if x_min < 280:
    # 파장 사용
    if x_min >= 65:
        x_min = "C5"
        #print("C5")
    elif x_min >= 62:
        x_min = "C#5"
        #print("C#5")
    elif x_min >= 58:
        x_min = "D5"
        #print("D5")
    elif x_min >= 55:
        x_min = "D#5"
        #print("D#5")
    elif x_min >= 52:
        x_min = "E5"
        #print("E5")
    elif x_min >= 49:
        x_min = "F5"
        #print("F5")
    elif x_min >= 46:
        x_min = "F#5"
        #print("F#5")
    elif x_min >= 44:
        x_min = "G5"
        #print("G5")
    elif x_min >= 41:
        x_min = "G#5"
        #print("G#5")
    elif x_min >= 39:
        x_min = "A5"
        #print("A5")
    elif x_min >= 37:
        x_min = "A#5"
        #print("A#5")
    elif x_min >= 34:
        x_min = "B5"
        #print("B5")
    elif x_min >= 32:
        x_min = "C6"
        #print("C6")
    elif x_min >= 31:
        x_min = "C#6"
        #print("C#6")
    elif x_min >= 29:
        x_min = "D6"
        #print("D6")
    elif x_min >= 27:
        x_min = "D#6"
        #print("D#6")
    elif x_min >= 26:
        x_min = "E6"
        #print("E6")
    elif x_min >= 23:
        x_min = "F#6"
        #print("F#6")
    elif x_min >= 24:
        x_min = "F6"
        #print("F6")
    elif x_min >= 22:
        x_min = "G6"
        #print("G6")
    elif x_min >= 20:
        x_min = "G#6"
        #print("G#6")
    elif x_min >= 19:
        x_min = "A6"
        #print("A6")
    elif x_min >= 18:
        x_min = "A#6"
        #print("A#6")
    elif x_min >= 17:
        x_min = "B6"
        #print("B6")
else:
    # 주파수 사용
    if x_min <= 523:
        x_min = "C5"
        # print("C5")
    elif x_min <= 554:
        x_min = "C#5"
        # print("C#5")
    elif x_min <= 587:
        x_min = "D5"
        #print("D5")
    elif x_min <= 622:
        x_min = "D#5"
        # print("D#5")
    elif x_min <= 659:
        x_min = "E5"
        # print("E5")
    elif x_min <= 698:
        x_min = "F5"
        #print("F5")
    elif x_min <= 739:
        x_min = "F#5"
        #print("F#5")
    elif x_min <= 783:
        x_min = "G5"
        #print("G5")
    elif x_min <= 830:
        x_min = "G#5"
        #print("G#5")
    elif x_min <= 880:
        x_min = "A5"
        #print("A5")
    elif x_min <= 932:
        x_min = "A#5"
        #print("A#5")
    elif x_min <= 987:
        x_min = "B5"
        #print("B5")
    elif x_min <= 1046:
        x_min = "C6"
        #print("C6")
    elif x_min <= 1108:
        x_min = "C#6"
        #print("C#6")
    elif x_min <= 1174:
        x_min = "D6"
        #print("D6")
    elif x_min <= 1244:
        x_min = "D#6"
        #print("D#6")
    elif x_min <= 1318:
        x_min = "E6"
        #print("E6")
    elif x_min <= 1396:
        x_min = "F6"
        #print("F6")
    elif x_min <= 1479:
        x_min = "F#6"
        #print("F#6")
    elif x_min <= 1567:
        x_min = "G6"
        #print("G6")
    elif x_min <= 1661:
        x_min = "G#6"
        #print("G#6")
    elif x_min <= 1760:
        x_min = "A6"
        #print("A6")
    elif x_min <= 1864:
        x_min = "A#6"
        #print("A#6")
    elif x_min <= 1975:
        x_min = "B6"
        #print("B6")

# x_max 계이름 찾기 (주파수 사용)
if x_max <= 523:
    x_max = "C5"
    #print("C5")
elif x_max <= 554:
    x_max = "C#5"
    #print("C#5")
elif x_max <= 587:
    x_max = "D5"
    #print("D5")
elif x_max <= 622:
    x_max = "D#5"
    #print("D#5")
elif x_max <= 659:
    x_max = "E5"
    #print("E5")
elif x_max <= 698:
    x_max = "F5"
    #print("F5")
elif x_max <= 739:
    x_max = "F#5"
    #print("F#5")
elif x_max <= 783:
    x_max = "G5"
    #print("G5")
elif x_max <= 830:
    x_max = "G#5"
    #print("G#5")
elif x_max <= 880:
    x_max = "A5"
    #print("A5")
elif x_max <= 932:
    x_max = "A#5"
    #print("A#5")
elif x_max <= 987:
    x_max = "B5"
    #print("B5")
elif x_max <= 1046:
    x_max = "C6"
    #print("C6")
elif x_max <= 1108:
    x_max = "C#6"
    #print("C#6")
elif x_max <= 1174:
    x_max = "D6"
    #print("D6")
elif x_max <= 1244:
    x_max = "D#6"
    #print("D#6")
elif x_max <= 1318:
    x_max = "E6"
    #print("E6")
elif x_max <= 1396:
    x_max = "F6"
    #print("F6")
elif x_max <= 1479:
    x_max = "F#6"
    #print("F#6")
elif x_max <= 1567:
    x_max = "G6"
    #print("G6")
elif x_max <= 1661:
    x_max = "G#6"
    #print("G#6")
elif x_max <= 1760:
    x_max = "A6"
    #print("A6")
elif x_max <= 1864:
    x_max = "A#6"
    #print("A#6")
else:
    x_max = "B6"
    #print("B6")

# x_min 계이름 찾기
if x_avg < 280:
    # 파장 사용
    if x_avg >= 65:
        x_avg = "C5"
        #print("C5")
    elif x_avg >= 62:
        x_avg = "C#5"
        #print("C#5")
    elif x_avg >= 58:
        x_avg = "D5"
        #print("D5")
    elif x_avg >= 55:
        x_avg = "D#5"
        #print("D#5")
    elif x_avg >= 52:
        x_avg = "E5"
        #print("E5")
    elif x_avg >= 49:
        x_avg = "F5"
        #print("F5")
    elif x_avg >= 46:
        x_avg = "F#5"
        #print("F#5")
    elif x_avg >= 44:
        x_avg = "G5"
        #print("G5")
    elif x_avg >= 41:
        x_avg = "G#5"
        #print("G#5")
    elif x_avg >= 39:
        x_avg = "A5"
        #print("A5")
    elif x_avg >= 37:
        x_avg = "A#5"
        #print("A#5")
    elif x_avg >= 34:
        x_avg = "B5"
        #print("B5")
    elif x_avg >= 32:
        x_avg = "C6"
        #print("C6")
    elif x_avg >= 31:
        x_avg = "C#6"
        #print("C#6")
    elif x_avg >= 29:
        x_avg = "D6"
        #print("D6")
    elif x_avg >= 27:
        x_avg = "D#6"
        #print("D#6")
    elif x_avg >= 26:
        x_avg = "E6"
        #print("E6")
    elif x_avg >= 23:
        x_avg = "F#6"
        #print("F#6")
    elif x_avg >= 24:
        x_avg = "F6"
        #print("F6")
    elif x_avg >= 22:
        x_avg = "G6"
        #print("G6")
    elif x_avg >= 20:
        x_avg = "G#6"
        #print("G#6")
    elif x_avg >= 19:
        x_avg = "A6"
        #print("A6")
    elif x_avg >= 18:
        x_avg = "A#6"
        #print("A#6")
    elif x_avg >= 17:
        x_avg = "B6"
        #print("B6")
else:
    # 주파수 사용
    if x_avg <= 523:
        x_avg = "C5"
        # print("C5")
    elif x_avg <= 554:
        x_avg = "C#5"
        #print("C#5")
    elif x_avg <= 587:
        x_avg = "D5"
        #print("D5")
    elif x_avg <= 622:
        x_avg = "D#5"
        #print("D#5")
    elif x_avg <= 659:
        x_avg = "E5"
        #print("E5")
    elif x_avg <= 698:
        x_avg = "F5"
        #print("F5")
    elif x_avg <= 739:
        x_avg = "F#5"
        #print("F#5")
    elif x_avg <= 783:
        x_avg = "G5"
        #print("G5")
    elif x_avg <= 830:
        x_avg = "G#5"
        #print("G#5")
    elif x_avg <= 880:
        x_avg = "A5"
        #print("A5")
    elif x_avg <= 932:
        x_avg = "A#5"
        #print("A#5")
    elif x_avg <= 987:
        x_avg = "B5"
        #print("B5")
    elif x_avg <= 1046:
        x_avg = "C6"
        #print("C6")
    elif x_avg <= 1108:
        x_avg = "C#6"
        #print("C#6")
    elif x_avg <= 1174:
        x_avg = "D6"
        #print("D6")
    elif x_avg <= 1244:
        x_avg = "D#6"
        #print("D#6")
    elif x_avg <= 1318:
        x_avg = "E6"
        #print("E6")
    elif x_avg <= 1396:
        x_avg = "F6"
        #print("F6")
    elif x_avg <= 1479:
        x_avg = "F#6"
        #print("F#6")
    elif x_avg <= 1567:
        x_avg = "G6"
        #print("G6")
    elif x_avg <= 1661:
        x_avg = "C#6"
        #print("G#6")
    elif x_avg <= 1760:
        x_avg = "A6"
        #print("A6")
    elif x_avg <= 1864:
        x_avg = "A#6"
        #print("A#6")
    elif x_avg <= 1975:
        x_avg = "B6"
        #print("B6")

# y_min 계이름 찾기 (파장 사용)
if y_min < 280:
    if y_min >= 65:
        y_min = "C5"
        #print("C5")
    elif y_min >= 62:
        y_min = "C#5"
        #print("C#5")
    elif y_min >= 58:
        y_min = "D5"
        #print("D5")
    elif y_min >= 55:
        y_min = "D#5"
        #print("D#5")
    elif y_min >= 52:
        y_min = "E5"
        #print("E5")
    elif y_min >= 49:
        y_min = "F5"
        #print("F5")
    elif y_min >= 46:
        y_min = "F#5"
        #print("F#5")
    elif y_min >= 44:
        y_min = "G5"
        #print("G5")
    elif y_min >= 41:
        y_min = "G#5"
        #print("G#5")
    elif y_min >= 39:
        y_min = "A5"
        #print("A5")
    elif y_min >= 37:
        y_min = "A#5"
        #print("A#5")
    elif y_min >= 34:
        y_min = "B5"
        #print("B5")
    elif y_min >= 32:
        y_min = "C6"
        #print("C6")
    elif y_min >= 31:
        y_min = "C#6"
        #print("C#6")
    elif y_min >= 29:
        y_min = "D6"
        #print("D6")
    elif y_min >= 27:
        y_min = "D#6"
        #print("D#6")
    elif y_min >= 26:
        y_min = "E6"
        #print("E6")
    elif y_min >= 23:
        y_min = "F#6"
        #print("F#6")
    elif y_min >= 24:
        y_min = "F6"
        #print("F6")
    elif y_min >= 22:
        y_min = "G6"
        #print("G6")
    elif y_min >= 20:
        y_min = "G#6"
        #print("G#6")
    elif y_min >= 19:
        y_min = "A6"
        #print("A6")
    elif y_min >= 18:
        y_min = "A#6"
        #print("A#6")
    elif y_min >= 17:
        y_min = "B6"
        #print("B6")
else:
    if y_min <= 523:
        y_min = "C5"
        #print("C5")
    elif y_min <= 554:
        y_min = "C#5"
        #print("C#5")
    elif y_min <= 587:
        y_min = "D5"
        #print("D5")
    elif y_min <= 622:
        y_min = "D#5"
        #print("D#5")
    elif y_min <= 659:
        y_min = "E5"
        #print("E5")
    elif y_min <= 698:
        y_min = "F5"
        #print("F5")
    elif y_min <= 739:
        y_min = "F#5"
        #print("F#5")
    elif y_min <= 783:
        y_min = "G5"
        #print("G5")
    elif y_min <= 830:
        y_min = "G#5"
        #print("G#5")
    elif y_min <= 880:
        y_min = "A5"
        #print("A5")
    elif y_min <= 932:
        y_min = "A#5"
        #print("A#5")
    elif y_min <= 987:
        y_min = "B5"
        #print("B5")
    elif y_min <= 1046:
        y_min = "C6"
        #print("C6")
    elif y_min <= 1108:
        y_min = "C#6"
        #print("C#6")
    elif y_min <= 1174:
        y_min = "D6"
        #print("D6")
    elif y_min <= 1244:
        y_min = "D#6"
        #print("D#6")
    elif y_min <= 1318:
        y_min = "E6"
        #print("E6")
    elif y_min <= 1396:
        y_min = "F6"
        #print("F6")
    elif y_min <= 1479:
        y_min = "F#6"
        #print("F#6")
    elif y_min <= 1567:
        y_min = "G6"
        #print("G6")
    elif y_min <= 1661:
        y_min = "G#6"
        #print("G#6")
    elif y_min <= 1760:
        y_min = "A6"
        #print("A6")
    elif y_min <= 1864:
        y_min = "A#6"
        #print("A#6")
    elif y_min <= 1975:
        y_min = "B6"
        #print("B6")

# y_max 계이름 찾기 (주파수 사용)
if y_max <= 523:
    y_max = "C5"
    #print("C5")
elif y_max <= 554:
    y_max = "C#5"
    #print("C#5")
elif y_max <= 587:
    y_max = "D5"
    #print("D5")
elif y_max <= 622:
    y_max = "D#5"
    #print("D#5")
elif y_max <= 659:
    y_max = "E5"
    #print("E5")
elif y_max <= 698:
    y_max = "F5"
    #print("F5")
elif y_max <= 739:
    y_max = "F#5"
    #print("F#5")
elif y_max <= 783:
    y_max = "G5"
    #print("G5")
elif y_max <= 830:
    y_max = "G#5"
    #print("G#5")
elif y_max <= 880:
    y_max = "A5"
    #print("A5")
elif y_max <= 932:
    y_max = "A#5"
    #print("A#5")
elif y_max <= 987:
    y_max = "B5"
    #print("B5")
elif y_max <= 1046:
    y_max = "C6"
    #print("C6")
elif y_max <= 1108:
    y_max = "C#6"
    #print("C#6")
elif y_max <= 1174:
    y_max = "D6"
    #print("D6")
elif y_max <= 1244:
    y_max = "D#6"
    #print("D#6")
elif y_max <= 1318:
    y_max = "E6"
    #print("E6")
elif y_max <= 1396:
    y_max = "F6"
    #print("F6")
elif y_max <= 1479:
    y_max = "F#6"
    #print("F#6")
elif y_max <= 1567:
    y_max = "G6"
    #print("G6")
elif y_max <= 1661:
    y_max = "G#6"
    #print("G#6")
elif y_max <= 1760:
    y_max = "A6"
    #print("A6")
elif y_max <= 1864:
    y_max = "A#6"
    #print("A#6")
elif y_max <= 1975:
    y_max = "B6"
    #print("B6")

# y 평균값 코드 구하기
if y_avg < 280:
    if y_avg >= 65:
        y_avg = "C5"
        #print("C5")
    elif y_avg >= 62:
        y_avg = "C#5"
        #print("C#5")
    elif y_avg >= 58:
        y_avg = "D5"
        #print("D5")
    elif y_avg >= 55:
        y_avg = "D#5"
        #print("D#5")
    elif y_avg >= 52:
        y_avg = "E5"
        #print("E5")
    elif y_avg >= 49:
        y_avg = "F5"
        #print("F5")
    elif y_avg >= 46:
        y_avg = "F#5"
        #print("F#5")
    elif y_avg >= 44:
        y_avg = "G5"
        #print("G5")
    elif y_avg >= 41:
        y_avg = "G#5"
        #print("G#5")
    elif y_avg >= 39:
        y_avg = "A5"
        #print("A5")
    elif y_avg >= 37:
        y_avg = "A#5"
        #print("A#5")
    elif y_avg >= 34:
        y_avg = "B5"
        #print("B5")
    elif y_avg >= 32:
        y_avg = "C6"
        #print("C6")
    elif y_avg >= 31:
        y_avg = "C#6"
        #print("C#6")
    elif y_avg >= 29:
        y_avg = "D6"
        #print("D6")
    elif y_avg >= 27:
        y_avg = "D#6"
        #print("D#6")
    elif y_avg >= 26:
        y_avg = "E6"
        #print("E6")
    elif y_avg >= 23:
        y_avg = "F#6"
        #print("F#6")
    elif y_avg >= 24:
        y_avg = "F6"
        #print("F6")
    elif y_avg >= 22:
        y_avg = "G6"
        #print("G6")
    elif y_avg >= 20:
        y_avg = "G#6"
        #print("G#6")
    elif y_avg >= 19:
        y_avg = "A6"
        #print("A6")
    elif y_avg >= 18:
        y_avg = "A#6"
        #print("A#6")
    elif y_avg >= 17:
        y_avg = "B6"
        #print("B6")
else:
    if y_avg <= 523:
        y_avg = "C5"
        #print("C5")
    elif y_avg <= 554:
        y_avg = "C#5"
        #print("C#5")
    elif y_avg <= 587:
        y_avg = "D5"
        #print("D5")
    elif y_avg <= 622:
        y_avg = "D#5"
        #print("D#5")
    elif y_avg <= 659:
        y_avg = "E5"
        #print("E5")
    elif y_avg <= 698:
        y_avg = "F5"
        #print("F5")
    elif y_avg <= 739:
        y_avg = "F#5"
        #print("F#5")
    elif y_avg <= 783:
        y_avg = "G5"
        #print("G5")
    elif y_avg <= 830:
        y_avg = "G#5"
        #print("G#5")
    elif y_avg <= 880:
        y_avg = "A5"
        #print("A5")
    elif y_avg <= 932:
        y_avg = "A#5"
        #print("A#5")
    elif y_avg <= 987:
        y_avg = "B5"
        #print("B5")
    elif y_avg <= 1046:
        y_avg = "C6"
        #print("C6")
    elif y_avg <= 1108:
        y_avg = "C#6"
        #print("C#6")
    elif y_avg <= 1174:
        y_avg = "D6"
        #print("D6")
    elif y_avg <= 1244:
        y_avg = "D#6"
        #print("D#6")
    elif y_avg <= 1318:
        y_avg = "E6"
        #print("E6")
    elif y_avg <= 1396:
        y_avg = "F6"
        #print("F6")
    elif y_avg <= 1479:
        y_avg = "F#6"
        #print("F#6")
    elif y_avg <= 1567:
        y_avg = "G6"
        #print("G6")
    elif y_avg <= 1661:
        y_avg = "G#6"
        #print("G#6")
    elif y_avg <= 1760:
        y_avg = "A6"
        #print("A6")
    elif y_avg <= 1864:
        y_avg = "A#6"
        #print("A#6")
    elif y_avg <= 1975:
        y_avg = "B6"
        #print("B6")

print("x, y 최솟값 평균값 최댓값 : ", x_min, ",", y_min, ",", x_avg, ",", y_avg, ",", x_max, ",", y_max)
#cv2.imshow("src", src)
cv2.waitKey(0)

cv2.destroyAllWindows()

plt.xlabel('Time[ticks]')
plt.ylabel('Pitch')
plt.grid(True)
plt.xticks(xpos, X)
plt.yticks(ypos, Y)

# plt.imshow(image)
#plt.show()
