from ultralytics import YOLO
import cv2, time
import numpy as np
from models.pycuda_api import TRTEngine
from tensorrt_infer_det_without_torch import inference

tensorrt_mode = 1
mask_detect_mode = 1
webcam_mode = 1

if mask_detect_mode:
    model = YOLO("model/mask_detect/best.pt")
    engine = TRTEngine("model/mask_detect/best.engine")
    target = "inference/face-mask-video.mp4"
else:
    model = YOLO("model/yolov8l.pt")
    engine = TRTEngine("model/yolov8l.engine")
    #names=model.names
    target = "inference/City.mp4"
    #target = "https://trafficvideo2.tainan.gov.tw/82520774"

# Run batched infernece on a list of images
#results = model.predict(img, stream=True) # return a list of Results objects
if webcam_mode:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(target)
#CountFrame = 0
#dt=0
while True:
    try:
        r, img = cap.read()
        st = time.time()
        #img =cv2.resize(img, (800, 600))
        if not tensorrt_mode:
            results = model(source=img)
            img = results[0].plot() # annotated_frame
        else:
            img = inference(engine, img, mask_detect_mode)

        et = time.time()

        FPS = round(1/(et-st))
        cv2.putText(img, 'FPS=' + str(FPS), (20, 150),
                    cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("YOLOv8", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(e)

cap.release()
cv2.destroyAllWindows()