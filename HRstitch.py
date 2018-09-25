import imutils
import cv2
from HRtranslationRowStitch import Stitcher
# from moderationWorkinStitch import Stitcher
import numpy as np
from scipy import signal


Path = "C:\\Users\\ayele\\Documents\\Ayelet\\Technion\\python\\my code\\images\\high_resolution\\"
mean_Path = "C:\\Users\\ayele\\Documents\\Ayelet\\Technion\\python\\my code\\images\\HRmean\\"

tip_length = 0


def calc_mean():
    j = 1
    images = []
    img = cv2.imread(mean_Path + "a (%d).png" % j)
    cropped = cut_edges(img)
    h = cropped.shape[0]
    w = cropped.shape[1]
    while img is not None:
        print j
        img = cv2.imread(mean_Path + "a (%d).png" % j)
        if img is not None:
            cropped = cut_edges(img)
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
    cv2.imwrite(mean_Path + "mean_img.png", gray_mean)
    return None


def cut_edges(img):
    (h, w) = img.shape[:2]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = w - 1
    y = h - 1
    black_col = 0
    isblack = True
    while x >= 0 and isblack:
        while y >= 0 and isblack:
            isblack = gray_img[y, x] == 0
            y -= 1
        x -= 1
        y = h - 1
        black_col += 1
    # print "black_col = %d" % black_col
    cropped = img[tip_length:, :(w - black_col)]
    # cv2.imshow("pic", cropped)
    # cv2.waitKey(0)
    return cropped


def read_images(paths):
    i = 0
    images = []
    num = len(paths)
    while i < num:
        img = cv2.imread(paths[i])
        cropped = cut_edges(img)
        # cv2.imshow("cropped image %d" % i, cropped)
        # cv2.waitKey(0)
        images += [cropped]
        # cv2.imshow("Image %d" % i, images[i])
        # cv2.waitKey(0)
        i += 1
    return images


def highpass_check():
    stitcher = Stitcher()
    dark = cut_edges(cv2.imread(Path + "r2 (5).png"))
    gray_dark = cv2.cvtColor(dark, cv2.COLOR_RGB2GRAY)
    HPdark = stitcher.highPass(gray_dark)

    (h, w) = dark.shape[:2]
    darkshow = np.zeros((h, 2 * w), dtype="uint8")
    darkshow[:, 0:w] = gray_dark
    darkshow[:, w:] = HPdark
    resize_dark = cv2.resize(darkshow, (1350, 600), interpolation=cv2.INTER_AREA)

    bright = cut_edges(cv2.imread(Path + "i (0).png"))
    gray_bright = cv2.cvtColor(bright, cv2.COLOR_RGB2GRAY)
    HPbright = stitcher.highPass(gray_bright)

    (y, x) = bright.shape[:2]
    brightshow = np.zeros((y, 2 * x), dtype="uint8")
    brightshow[:, 0:w] = gray_bright
    brightshow[:, w:] = HPbright
    resize_bright = cv2.resize(brightshow, (1350, 600), interpolation=cv2.INTER_AREA)

    cv2.imshow("resize_clear", resize_bright)
    cv2.imshow("resize_dark", resize_dark)
    cv2.waitKey(0)
    cv2.imwrite(Path + "highpass_bright_image.png", resize_bright)
    cv2.imwrite(Path + "highpass_dark_image.png", resize_dark)

    return 0


def histogramModeration(img):

    hist, _ = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max() / cdf.max()

    # show the equalized histogram
    # plt.plot(cdf_normalized, color='b')
    # plt.hist(img.flatten(), 256, [0, 256], color='r')
    # plt.xlim([0, 256])
    # plt.legend(('cdf', 'histogram'), loc='upper left')
    # plt.show()

    # create the equalized image
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    result = cdf[img]

    # show the result
    cv2.imshow("equalized histogram", result)
    cv2.waitKey(0)
    return result


if __name__ == "__main__":
    # calc_mean()
    # highpass_check()

    paths = [Path + "t (1).png", Path + "t (2).png", Path + "t (3).png", Path + "t (4).png"]
    # paths = [Path + "r1 (1).png", Path + "r1 (2).png", Path + "r1 (3).png", Path + "r1 (4).png",
    #          Path + "r1 (5).png", Path + "r1 (6).png"]

    # paths = [Path + "i (0).png", Path + "i (1).png", Path + "i (2).png"]
    # paths = [Path + "i2.png", Path + "i3.png"]
    # paths = [Path + "g5.png", Path + "g6.png"]
    # paths = [Path + "g2.png", Path + "g3.png", Path + "g4.png"]
    # paths = [Path+"g2.png", Path+"g3.png", Path+"g4.png",
    #          Path+"g5.png", Path+"g6.png", Path+"g7.png", Path+"g8.png", Path+"g9.png"]
    # paths = [Path + "img1.png", Path + "img2.png", Path + "img3.png"]

    images = read_images(paths)
    stitcher = Stitcher()
    color_result, gray_result = stitcher.stitch(images, showMatches=False)

    # show the images
    # cv2.imshow("Keypoint Matches", vis)
    # cv2.imshow("Result", result)
    # cv2.waitKey(0)

    cv2.imwrite(Path + "row2.png", color_result)
    cv2.imwrite(Path + "row2_gray.png", gray_result)

    (h, w) = color_result.shape[:2]
    size_factor = w / 1550
    resize_color = cv2.resize(color_result, (1550, h * size_factor), interpolation=cv2.INTER_AREA)
    cv2.imshow("Resize Result In Color", resize_color)
    resize_gray = cv2.resize(gray_result, (1550, h * size_factor), interpolation=cv2.INTER_AREA)
    cv2.imshow("Resize Result In Grayscale", resize_gray)
    cv2.waitKey(0)
