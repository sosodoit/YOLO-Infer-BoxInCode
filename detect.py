# 실행 환경: colab
# %cd '/content/drive/MyDrive/Colab Notebooks/dev/detection/yolov5'
# !pip install ultralytics opencv-python

import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

import cv2
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt

# 모델 로드 (best.pt는 업로드한 맞춤형 모델)
model = YOLO("best.pt")

# 이미지 로드
img_paths = ["../images/sample_box_202503261542 (1).jpg","../images/sample_box_202503261542 (2).jpg","../images/sample_box_202503252002 (1).jpg","../images/sample_box_202503252002 (2).jpg","../images/sample_box_202503252002 (3).jpg","../images/sample_box_202503252002 (4).jpg","../images/sample_box_202503252002 (5).jpg"]
results = model(img_paths)

# 박스 안에 바코드 존재 여부 확인 (class ID: 0 = 바코드, 1 = 박스)
def is_barcode_in_box(results, img_path):

    for i, img_result in enumerate(results):
        img_array = img_result.orig_img
        bboxes = img_result.boxes.data.cpu().numpy()
        class_ids = img_result.boxes.cls.cpu().numpy().astype(int)
        confs = img_result.boxes.conf.cpu().numpy()

        print(f"\n감지된 객체 수: {len(bboxes)}")

        box_loc_list = []
        barcode_loc_list = []

        for j, (bbox, cls_id, conf) in enumerate(zip(bboxes, class_ids, confs)):
            xmin, ymin, xmax, ymax = map(int, bbox[:4])
            
            # 박스 그리기
            label = "Barcode" if cls_id == 0 else "Box"
            color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
            cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), color, 2)

            text = f"{label} {conf:.2f}"
            cv2.putText(img_array, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 위치 저장
            if cls_id == 0:
                barcode_loc_list.append([xmin, ymin, xmax, ymax, conf])
            elif cls_id == 1:
                box_loc_list.append([xmin, ymin, xmax, ymax, conf])

        # 조건: 둘 다 탐지되어야 함
        if not box_loc_list or not barcode_loc_list:
            print("박스 또는 바코드가 감지되지 않음")
            return "barcode_abnormal", img_array

        # box, barcode 각각 신뢰도 0.5 이상 필터링
        box_loc_list = [b for b in box_loc_list if b[4] >= 0.5]
        barcode_loc_list = [b for b in barcode_loc_list if b[4] >= 0.5]

        if not box_loc_list or not barcode_loc_list:
            print("신뢰도 0.5 미만 → 비정상")
            return "barcode_abnormal", img_array

        # 바코드가 박스 안에 있는지 확인
        for box in box_loc_list:
            box_x1, box_y1, box_x2, box_y2 = box[:4]

            for barcode in barcode_loc_list:
                bc_x1, bc_y1, bc_x2, bc_y2 = barcode[:4]

                if box_x1 <= bc_x1 and box_y1 <= bc_y1 and box_x2 >= bc_x2 and box_y2 >= bc_y2:
                    print("(정상)바코드가 박스 안에 있습니다.")
                    return "barcode_normal", img_array

        print("(비정상)바코드가 박스 안에 있지 않습니다.")
        return "barcode_abnormal", img_array

# 전체 이미지 반복 처리
for idx, path in enumerate(img_paths):

    # 결과 판별
    results = model(path)
    result, vis_img = is_barcode_in_box(results, path)
    print(f"[{idx}] {os.path.basename(path)} → 결과: {result}")

    # 이미지 저장
    save_path = f"detect_images/result_{idx}_{result}.jpg"
    cv2.imwrite(save_path, vis_img)

    # 시각화 
    img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(4, 4))
    plt.imshow(img_rgb)
    plt.title(f"{os.path.basename(path)} → {result}")
    plt.axis('off')
    plt.show()