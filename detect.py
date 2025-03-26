# 실행 환경: colab
# %cd '/content/drive/MyDrive/Colab Notebooks/dev/detection/yolov5'
# !pip install ultralytics opencv-python

import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

import cv2
from ultralytics import YOLO
import os

# 모델 로드
model = YOLO("best.pt")

# 이미지 로드
img_path = "../images/sample_box_202503252002 (3).jpg"
results = model(img_path)

# 박스 위치 내부에 바코드가 있는지 확인
def is_barcode_in_box(results):    

    # results: 검출 결과
    for i, img_result in enumerate(results):
        img_array = img_result.orig_img
        bboxes = img_result.boxes.data.cpu().numpy()
        class_ids = img_result.boxes.cls.cpu().numpy().astype(int)
        confs = img_result.boxes.conf.cpu().numpy()

        print(f"[Image {i}] 감지된 객체 수: {len(bboxes)}")

        box_loc_list = []
        barcode_loc_list = []

        for j, (bbox, cls_id, conf) in enumerate(zip(bboxes, class_ids, confs)):
            xmin, ymin, xmax, ymax = map(int, bbox[:4])
            if cls_id == 0:
                barcode_loc_list.append([xmin, ymin, xmax, ymax, conf])
            elif cls_id == 1:
                box_loc_list.append([xmin, ymin, xmax, ymax, conf])

        # 정상 조건: 둘 다 검출 되어야 함
        if not box_loc_list or not barcode_loc_list:
            print("박스 또는 바코드가 검출되지 않음")
            return "barcode_abnormal"

        # 신뢰도 0.5 이상인 바코드, 박스 중 가장 큰 것 기준 사용
        box = max(box_loc_list, key=lambda x: x[4])
        barcode = max(barcode_loc_list, key=lambda x: x[4])

        if box[4] < 0.5 or barcode[4] < 0.5:
            print("신뢰도 0.5 미만")
            return "barcode_abnormal"

        # 바코드가 박스 안에 있는지 확인
        box_x1, box_y1, box_x2, box_y2 = box[:4]
        bc_x1, bc_y1, bc_x2, bc_y2 = barcode[:4]

        if box_x1 <= bc_x1 and box_y1 <= bc_y1 and box_x2 >= bc_x2 and box_y2 >= bc_y2:
            print("(정상)바코드가 박스 안에 있습니다.")
            return "barcode_normal"
        else:
            print("(비정상)바코드가 박스 안에 있지 않습니다.")
            return "barcode_abnormal"

# 결과 판별
result = is_barcode_in_box(results)
print(f"\n최종 결과: {result}")

# 시각화 저장
annotated = results[0].plot()
cv2.imwrite("barcode_box_result.jpg", annotated)