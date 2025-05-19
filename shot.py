import cv2
import numpy as np
import requests

url = "http://192.168.100.231:8080/shot.jpg?res"

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)

    cv2.imshow("Full-Res Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
