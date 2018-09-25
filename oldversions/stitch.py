# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png 
# import the necessary packages
# from pyimagesearch.panorama import Stitcher
# import argparse
import imutils
import cv2
from translationRowStitch import Stitcher
import numpy as np
from scipy import signal

# import PIL

Path = "C:\\Users\\ayele\\Documents\\Ayelet\\Technion\\python\\my code\\images\\"
tip_length = 250


def calc_mean(sample_num):

    j = 1
    images = []

    print Path + "mean_s%d\\" % sample_num + "im (%d).png" % j
    img = cv2.imread(Path + "mean_s%d\\" % sample_num + "im (%d).png" % j)
    cropped = cut_tip(img)
    h = cropped.shape[0]
    w = cropped.shape[1]
    while img is not None:
        img = cv2.imread(Path + "mean_s%d\\" % sample_num + "im (%d).png" % j)
        if img is not None:
            cropped = cut_tip(img)
            images.append(cropped)
        j += 1

    images = np.array(images)

    mean = np.zeros((h, w, 3))
    num = len(images)
    for img in images:
        mean += img
    mean /= num
    mean_img = np.uint8(mean)
    gray_mean = cv2.cvtColor(mean_img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(Path + "mean_s%d\\mean_img.png" % sample_num, gray_mean)
    return None


def cut_tip(img):
    cropped = img[tip_length:, :]
    return cropped


def read_images(paths):
    i = 0
    images = []
    num = len(paths)
    while i < num:
        img = cv2.imread(paths[i])

        cropped = cut_tip(img)
        # cv2.imshow("cropped image %d" % i, crop)
        # cv2.waitKey(0)

        images += [cropped]
        # cv2.imshow("Image %d" % i, images[i])
        # cv2.waitKey(0)

        i += 1
    return images


if __name__ == "__main__":
    sample_num = 1
    # calc_mean(sample_num)

    # paths = [Path + "i2.png", Path + "i3.png"]
    # paths = [Path + "g5.png", Path + "g6.png"]
    # paths = [Path + "g2.png", Path + "g3.png", Path + "g4.png"]
    paths = [Path+"g2.png", Path+"g3.png", Path+"g4.png",
             Path+"g5.png", Path+"g6.png", Path+"g7.png", Path+"g8.png", Path+"g9.png"]
    # paths = [Path + "img1.png", Path + "img2.png", Path + "img3.png"]

    images = read_images(paths)
    #

    # highpass check
    # img = cv2.imread(Path + "mean_s%d\\" % sample_num + "im (1).png")
    # highpass = highpass(cut_tip(img))
    # cv2.imshow("highpass", highpass)
    # cv2.waitKey(0)
    # stitch the images together to create a panorama
    im1 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)

    # cor = signal.correlate2d(im1, im2)
    # print cor

    stitcher = Stitcher()
    color_result, gray_result = stitcher.stitch(images, showMatches=False)

    # show the images
    # cv2.imshow("Keypoint Matches", vis)
    # cv2.imshow("Result", result)
    # cv2.waitKey(0)

    cv2.imwrite(Path + "result_in_color.png", color_result)
    cv2.imwrite(Path + "result_in_grayscale.png", gray_result)

    (h, w) = color_result.shape[:2]
    size_factor = w / 1550
    resize_color = cv2.resize(color_result, (1550, h * size_factor), interpolation=cv2.INTER_AREA)
    cv2.imshow("Resize Result In Color", resize_color)
    resize_gray = cv2.resize(gray_result, (1550, h * size_factor), interpolation=cv2.INTER_AREA)
    cv2.imshow("Resize Result In Grayscale", resize_gray)
    cv2.waitKey(0)


