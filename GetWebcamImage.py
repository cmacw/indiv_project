"""
Code: https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
"""
import cv2

cam = cv2.VideoCapture(1)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    temp = frame[240, 320].copy()
    frame[240, 320] = [0, 255, 0]
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "WebCamPhotos/_{}.png".format(img_counter)
        # Restore pixel colour
        frame[240, 320] = temp
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
