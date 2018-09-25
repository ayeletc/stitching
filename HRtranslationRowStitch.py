import imutils
import cv2
# from PIL import Image
# import math
import numpy as np
import copy
# from scipy import ndimage
import matplotlib.pyplot as plt

moderation_constx = 50
moderation_consty = 600
overlapx_min = 250
overlapy_max = 20
#   num of pixels that we use to calculate the mean difference in the exposure between 2 stitched images
resize_factor = 9 / 10
resolution_factor = 1   #   while debugging it is possible to decrease the image's resolution
blur_factor = 11
i = 0
Path = "C:\\Users\\ayele\\Documents\\Ayelet\\Technion\\python\\my code\\images\\high_resolution\\"


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()
        print self.isv3

    def stitch(self, images, ratio=0.75, reprojThresh=1.0,
               showMatches=False):
        # resize = False
        filtered = self.filterList(images)  # list of filtered images
        j = 0
        num = len(images)
        imageL = images[j]  # left  image
        filtL = filtered[j]
        while num - 1 > j:
            imageR = images[j + 1]  # right image
            filtR = filtered[j + 1]

            # resize_L = cv2.resize(filtL, (650, 500), interpolation=cv2.INTER_AREA)
            # cv2.imshow("Resize imageL", resize_L)
            # resize_R = cv2.resize(filtR, (650, 500), interpolation=cv2.INTER_AREA)
            # cv2.imshow("Resize imageR", resize_R)
            # cv2.waitKey(0)

            # if resize:
            #     imageR = cv2.resize(imageR, (w * resize_factor, h, 3), interpolation=cv2.INTER_AREA)
            #     filtR = cv2.resize(filtR, (w * resize_factor, h), interpolation=cv2.INTER_AREA)
            # cv2.imshow("imgR %d" % j, imageR)
            # cv2.imshow("imgL%d" % j, imageL)
            # cv2.waitKey(0)
            (kpsL, featuresL) = self.detectAndDescribe(filtL)
            (kpsR, featuresR) = self.detectAndDescribe(filtR)

            # match features between the two images
            M = self.matchKeypoints(kpsR, kpsL,
                                    featuresR, featuresL, ratio, reprojThresh)

            # if the match is None, then there aren't enough matched
            # keypoints to create a panorama
            if M is None:
                print "no matching keypoints"
                return None

            # stitch the images according the translation
            (transx, transy) = M    # transy > 0  => L is above
            color_result = self.stickByTranslation(imageL, imageR, transx, transy)
            filt_result = self.stickByTranslation(filtL, filtR, transx, transy)

            # resize_filt = cv2.resize(filt_result, (1550, filt_result.shape[0]), interpolation=cv2.INTER_AREA)
            # cv2.imshow("Resize gray stitch", resize_filt)
            # cv2.waitKey(0)

            imageL = color_result
            filtL = filt_result
            j += 1

        # return the stitched image
        return imageL, filtL


    def filterList(self, images):
        filtered = []
        j = 0
        num = len(images)
        while j < num:
            image = images[j]
            filt = self.filters(image)
            filtered += [filt]
            j += 1
        return filtered


    def filters(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # change resolution
        reso_img = cv2.resize(gray_img, (0, 0), fx=resolution_factor, fy=resolution_factor)

        hp_img = self.highPass(reso_img)
        # resizeHP = cv2.resize(hp_img, (1200, 600), interpolation=cv2.INTER_AREA)
        # cv2.imshow("hp Image %d" % i, resizeHP)
        # cv2.waitKey(0)

        # blur
        gauss = cv2.GaussianBlur(hp_img, (blur_factor, blur_factor), 0)
        # sharpen
        sharp = self.sharpFilter(gauss)
        return sharp


    def sharpFilter(self, image, h=-3.0, l=1.0):
        kernel1 = np.array([[l, l, l], [h, -(3 * h + 5 * l) + 1, h], [l, l, h]])
        sharp1 = cv2.filter2D(image, -1, kernel1)

        kernel2 = np.array([[l, h, h], [l, -(3 * h + 5 * l) + 1, l], [l, h, l]])
        sharp2 = cv2.filter2D(sharp1, -1, kernel2)

        return sharp2


    def highPass(self, img):
        mean = cv2.imread(Path + "mean_img.png")
        # adjust the mean to the img size
        (h, w) = img.shape[:2] m
        crop_mean = mean[0:h, 0:w]
        # remove the mean
        mean_ar = np.array(crop_mean)[:, :, 1]
        img_ar = np.array(img)

        minval = np.amin(img_ar.astype(float) - mean_ar.astype(float))
        diff = img_ar.astype(float) - mean_ar.astype(float) + abs(minval)
        maxval = np.amax(diff)
        ratio = 255 / maxval
        hp_img = np.uint8(diff * ratio)

        # diff = img_ar.astype(float) - mean_ar.astype(float)
        # minval = np.amin(diff)
        # hp_img = np.uint8(diff + minval)
        return hp_img


    def detectAndDescribe(self, image):
        global i
        i += 1
        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        clone = copy.copy(image)
        kp_img = self.drawKeypoints(clone, kps)
        resize_kp_img = cv2.resize(kp_img, (1150, 900), interpolation=cv2.INTER_AREA)
        cv2.imshow("kp %d" % i, resize_kp_img)
        cv2.waitKey(0)

        # return a tuple of keypoints and features
        return kps, features

    def matchKeypoints(self, kpsR, kpsL, featuresR, featuresL,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresR, featuresL, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        transx, transy = self.findTranslation(matches, kpsR, kpsL)
        return transx, transy


    def drawMatches(self, imageR, imageL, kpsR, kpsL, matches, status):
        # initialize the output visualization image
        (hR, wR) = imageR.shape[:2]
        (hL, wL) = imageL.shape[:2]
        vis = np.zeros((max(hR, hL), wR + wL, 3), dtype="uint8")
        vis[0:hR, 0:wR] = imageR
        vis[0:hL, wR:] = imageL

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptR = (int(kpsR[queryIdx][0]), int(kpsR[queryIdx][1]))
                ptL = (int(kpsL[trainIdx][0]) + wR, int(kpsL[trainIdx][1]))
                cv2.line(vis, ptR, ptL, (0, 255, 0), 1)

        # return the visualization
        return vis

    def drawKeypoints(self, img, kps):
        vis = img.copy()
        for kp in kps:
            x = kp[0]
            y = kp[1]
            cv2.circle(vis, (x, y), 10, (0, 255, 0), 1)
        return vis


    def stickByTranslation(self, imageL, imageR, transx, transy):
        dim = imageR.ndim
        if dim == 3:  # RGB image
            #   TODO remove the shading before comparing the values
            for d in range(0, 3):
                imgL = imageL[:, :, d]
                imgR = imageR[:, :, d]
                moderatedR = self.moderateColor(imgL, imgR, transx, transy)
                imageR[:, :, d] = moderatedR
        else:   # grayscale image
            moderatedR = self.moderateColor(imageL, imageR, transx, transy)
            imageR = moderatedR

        (h, w) = imageL.shape[:2]
        (y, x) = imageR.shape[:2]
        if transy < 0:  # R above
            a = h + transy
            b = x + transx
            if dim == 3:    # RGB image
                result = np.zeros((a, b, 3), dtype="uint8")
                result[:, transx:b, :] = imageR[(y - h - transy):y, :, :]
                result[:, 0:w, :] = imageL[0:a, :, :]
            else:   # grayscale image
                result = np.zeros((a, b), dtype="uint8")
                result[:, transx:b] = imageR[(y - h - transy):y, :]
                result[:, 0:w] = imageL[0:a, :]
        else:  # L above
            a = h - transy
            b = x + transx
            if dim == 3:
                result = np.zeros((a, b, 3), dtype="uint8")
                result[:, transx:b, :] = imageR[0:a, :, :]
                # print "w = % d" % w
                # print "transx = % d" % transx
                # print "b = % d" % b
                result[:, 0:w, :] = imageL[transy:h, 0:w, :]
            else:
                result = np.zeros((a, b), dtype="uint8")
                result[:, transx:b] = imageR[0:a, :]
                result[:, 0:w] = imageL[transy:h, 0:w]
        return result


    def moderateColor(self, imageL, imageR, transx, transy):
        #   calculate the ratio between moderation_const-pixels in the 2 images
        ratio = 0.0
        # for
        # visR = np.array(imageR)
        # visL = np.array(imageL)

        for col in range(0, moderation_constx):
            for row in range(0, moderation_consty):
                if transy > 0:
                    if imageL[row + transy, col + transx] is not 0:
                        ratio += (float(imageR[row, col])) / (float(imageL[row + transy, col + transx]))

                    # cv2.circle(visR, (col, row), 10, (0, 255, 0), 1)
                    # cv2.circle(visL, (col + transx, row + transy), 10, (0, 255, 0), 1)
                else:  # R above
                    if imageL[row, col + transx] is not 0:
                        ratio += (float(imageR[row - transy, col]) / (float(imageL[row, col + transx])))
                    # cv2.circle(visR, (col, row-transy), 10, (0, 0, 255), 1)
                    # cv2.circle(visL, (col + transx, row), 10, (0, 0, 255), 1)
        # resize_R = cv2.resize(visR, (650, 500), interpolation=cv2.INTER_AREA)
        # resize_L = cv2.resize(visL, (650, 500), interpolation=cv2.INTER_AREA)
        # cv2.imshow("visR", resize_R)
        # cv2.imshow("visL", resize_L)
        # cv2.waitKey(0)

        ratio /= float(moderation_constx * moderation_consty)
        # print "moderation ratio = %f" % ratio
        # multiple one of the images and the mean ratio

        # to fix the RGB moderation
        # ratioR = (imageR / ratio)
        # diff = imageR - ratioR
        # maxdiff = np.amax(diff)
        # newimageR = np.uint8(diff + maxdiff)

        moderated_array = imageR / ratio
        newimageR = self.stretchto255limit(moderated_array)

        # self.showsidebyside("The two images", imageL, imageR)
        # self.showsidebyside("imageR before and after", imageR, newimageR)

        return newimageR

    def findTranslation(self, matches, kpsR, kpsL):
        transx = []
        transy = []
        for (trainIdx, queryIdx) in matches:
            ptR = (int(kpsR[queryIdx][0]), int(kpsR[queryIdx][1]))
            ptL = (int(kpsL[trainIdx][0]), int(kpsL[trainIdx][1]))
            x = abs(ptR[0] - ptL[0])
            y = (ptL[1] - ptR[1])   #   + => L is above

            transx.append(x)
            transy.append(y)
        k = 1
        stdy = abs(np.std(transy, axis=0))
        while stdy > 5 and k < 4:
            num = len(transx) - 1
            meany = np.mean(transy, axis=0)
            temp_transx = []
            temp_transy = []
            for j in range(0, num):
                # print "transy[j] = %d" % transy[j]
                # print "transx[j] = %d" % transx[j]
                # print "meany = %d" % meany
                # print "stdy = %d" % stdy
                if abs(transy[j] - meany) <= stdy and abs(transy[j]) < overlapy_max and transx[j] > overlapx_min:
                    temp_transx.append(transx[j])
                    temp_transy.append(transy[j])
            transx = temp_transx
            transy = temp_transy
            stdy = abs(np.std(transy, axis=0))
            k += 1
            #   TODO if transx is empty and kpsR&kpsL is not empty =>
            # try  to send the images back to increase the blur and collect kps again
            plt.plot(transx, 'r^')
            plt.plot(transy, 'bs')
            plt.show()
        translationx = int(np.ceil(np.mean(transx, axis=0)))
        translationy = int(np.ceil(np.mean(transy, axis=0)))

        return translationx, translationy


    def showsidebyside(self, title, imL, imR):

        (y, x) = imL.shape[:2]
        (h, w) = imR.shape[:2]
        miny = min(y, h)
        vis = np.zeros((miny, x + w), dtype="uint8")
        vis[:, 0:x] = imL[:miny, :]
        vis[:, x:] = imR[:miny, :]

        resize = cv2.resize(vis, (1350, 600), interpolation=cv2.INTER_AREA)
        cv2.imshow(title, resize)
        cv2.waitKey(0)
        return None


    def stretchto255limit(self, image_array):
        # get a float array of an image and if needed
        # stretch the values that the max value will be 255
        # return uint8 array
        maxval = np.amax(image_array)
        if maxval > 255:
            ratio = 255 / maxval
            stretched = np.uint8(image_array * ratio)
            return stretched

        return np.uint8(image_array)

