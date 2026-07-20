import cv2
import numpy as np

"""
[주요 변수 설명]
- drawing        : 마우스 드래그 상태 여부
- ix, iy         : ROI 시작 좌표
- rect           : 선택된 ROI 영역 (x, y, width, height)
- processed      : 객체 분리 수행 여부 (중복 연산 방지)
- segmented      : GrabCut으로 분리된 foreground 이미지
- markers        : Watershed 분할 결과 (객체 라벨 정보)
- count          : 검출된 객체 개수

[처리 순서]
1. ROI 선택 
2. GrabCut을 이용한 foreground 추출
3. Threshold 및 Morphological 연산으로 전처리 수행
4. Distance Transform을 통해 객체 중심 계산
5. Watershed 알고리즘으로 객체 분리
6. Contour 검출을 통해 객체 위치 및 개수 계산

[하이퍼 파라미터]
1. threshold : 10
2. Distance Transform : 0.2
3. kernel : 3*3

"""


#전역 설정
drawing = False
ix, iy = -1, -1
rect = None
processed = False

img = cv2.imread('trash.jpg')
if img is None:
    print("이미지 로드 실패")
    exit()

original = img.copy()


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, rect, processed

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = original.copy()
            cv2.rectangle(temp, (ix, iy), (x, y), (255, 0, 0), 2)
            cv2.imshow("Result", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
        processed = False
        print(f"Selected Rect: {rect}")


cv2.namedWindow("Result")
cv2.setMouseCallback("Result", mouse_callback)


segmented = None
count = 0

while True:
    display = original.copy()

    if rect is not None and not processed:


        mask = np.zeros(original.shape[:2], np.uint8)
        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(original, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        segmented = original * mask2[:, :, np.newaxis]


        gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3,3), np.uint8)


        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        _, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        sure_fg = cv2.erode(sure_fg, kernel, iterations=2)

        unknown = cv2.subtract(sure_bg, sure_fg)


        _, markers = cv2.connectedComponents(sure_fg)

        markers = markers + 1
        markers[unknown == 255] = 0


        markers = cv2.watershed(original, markers)

        colored = display.copy()

        for label in np.unique(markers):
            if label <= 1:
                continue

            mask_obj = (markers == label)

            color = np.random.randint(0, 255, size=3)

            colored[mask_obj] = color

        cv2.imshow("Colored", colored)


        count = 0

        for label in np.unique(markers):
            if label <= 1:
                continue

            mask_obj = np.zeros(original.shape[:2], dtype="uint8")
            mask_obj[markers == label] = 255

            contour_result = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contour_result) == 3:
                _, contours, _ = contour_result
            else:
                contours, _ = contour_result

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area < 1000:
                    continue

                count += 1

                x, y, w, h = cv2.boundingRect(cnt)

                cv2.rectangle(display, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(display, f"Obj {count}", (x,y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        processed = True


    if rect is not None:
        cv2.rectangle(display, (rect[0], rect[1]),
                      (rect[0]+rect[2], rect[1]+rect[3]),
                      (255, 0, 0), 2)

        cv2.putText(display, f"Total: {count}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)


    cv2.imshow("Result", display)

    if segmented is not None:
        cv2.imshow("Segmented", segmented)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    elif key == ord('c'):
        rect = None
        segmented = None
        processed = False
        count = 0
        print("초기화")



cv2.destroyAllWindows()