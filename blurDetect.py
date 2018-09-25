# import numpy as np
import cv2
import numpy as np


Path = "C:\\Users\\ayele\\Documents\\Ayelet\\Technion\\python\\my code\\images\\hrbank\\"


def variance_of_gaussian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_factor = 5
    gauss = cv2.GaussianBlur(gray, (blur_factor, blur_factor), 0)
    cv2.imshow("blur", gauss)
    cv2.waitKey(0)
    return gauss.var()


def variance_of_bilateral(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    cv2.imshow("blur", blur)
    cv2.waitKey(0)
    return blur.var()


def readimages():
    j = 1
    images = []
    image = 1
    while image is not None:
        # print Path + "blur1 (%d).png" % j
        image = cv2.imread(Path + "blur1 (%d).png" % j)
        if image is not None:
            images.append(image)
        j += 1
    return images


if __name__ == "__main__":
    images = readimages()
    # images = [cv2.imread(Path + "detecting_blur_result_009.jpg"), cv2.imread(Path + "detecting_blur_result_010.jpg")]

    i = 1
    for img in images:
        var = variance_of_gaussian(img)
        print "var %d = %f" % (i, var)
        i += 1

        # im1 = cv2.imread(Path + "2017-10-17_17-53-07_camera_full_res_blur3.png")
    # im2 = cv2.imread(Path + "2017-10-17_17-53-16_camera_full_res_blur3.png")
    # im3 = cv2.imread(Path + "2017-10-17_17-53-25_camera_full_res_blur3.png")
    # im4 = cv2.imread(Path + "2017-10-17_17-53-33_camera_full_res_blur3.png")
    # im5 = cv2.imread(Path + "2017-10-17_17-53-46_camera_full_res_blur3.png")
    # im6 = cv2.imread(Path + "20 17-10-17_17-53-53_camera_full_res_blur3.png")
    # im3 = cv2.imread(Path + "2017-10-17_17-53-07_camera_full_res_blur3.png")


    # im1 = cv2.imread(Path + "2017-10-17_17-50-40_camera_full_res_BLUR2.png")
    # im2 = cv2.imread(Path + "2017-10-17_17-51-25_camera_full_res_BLUR2.png")
    # im3 = cv2.imread(Path + "2017-10-17_17-52-37_camera_full_res_BLUR2.png")
    # im1 = cv2.imread(Path + "detecting_blur_result_004.jpg")

