import cv2
from ultralytics import YOLO
import ultralytics
import random

model = YOLO('best.pt')

class_name_dict = {0: 'Pothole'}
threshold = 0.1

def detect_object(_frame):
    # Convert the frame to grayscale
    grayscale_frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)

    # Edge detection
    canny = cv2.Canny(grayscale_frame, 100, 200)

    # Convert the result back to BGR
    _result = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    return _result

def yolo_detection(_frame):

    # Generate a random number pair
    lat, lng = random.uniform(22.341442, 22.344519), random.uniform(114.106910, 114.111564)

    ## Yolo Detection Model ##
    results = model(_frame)[0]

    score = 0
    bounding_box_areas = 0

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        # print (conf)
        if score > threshold:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            cv2.rectangle(_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 4)
            cv2.putText(_frame, class_name_dict[int(class_id)] + " " + str(round(score, 2)), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            # Calculate the area of the bounding box
            width = int(x2 - x1)
            height = int(y2 - y1)
            bounding_box_areas = width * height

            score = round(score, 3)
            lat = round(lat, 6)
            lng = round(lng, 6)

    return _frame, score, lat, lng, bounding_box_areas

if __name__=="__main__":
    testImg = cv2.imread('yolo/roadDefectProject/data/test/t2.jpg')
    resultImg = yolo_detection(testImg)
    cv2.imshow('My Image', resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
