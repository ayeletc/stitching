# the script finds the blur value of the image that
# has been saved in Path and named 'detect_blur.png'
# to call this script from ruby you should make sure that
# you are in the right dir and run: blur_val = `python blurVal.py`
# def getBlurVal
# 	output = `python blurVal.py`
# 	length = blur_val.length
# 	blur_val = output[0 .. length - 2].to_f
# end
import cv2
import json


Path = "C:\\Users\\ayele\\Documents\\Ayelet\\Technion\\python\\"
blur_factor = 5


if __name__ == "__main__":
    image = cv2.imread(Path + "detect_blur.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (blur_factor, blur_factor), 0)
    # cv2.imshow("blur", gauss)
    # cv2.waitKey(0)
    print gauss.var()