import imutils
import cv2
from PIL import Image
import numpy as np
import copy
import math
import matplotlib.pyplot as plt


moderation_const = 200  #   num of pixels that we use to calculate the mean difference in the exposure between 2 stitched images
resize_factor = 9 / 10
resolution_factor = 1   #   while debugging it is possible to decrease the image's resolution
blur_factor = 9
i = 0
Path = "C:\\Users\\ayele\\Documents\\Ayelet\\Technion\\python\\my code\\images\\high_resolution\\"


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

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

            # # resize the pictures
            # (h, w) = color_result.shape[:2]
            # if w > 1200:
            #     color_result = cv2.resize(color_result, (0, 0), fx=resize_factor, fy=1)
            #     filt_result = cv2.resize(filt_result, (0, 0), fx=resize_factor, fy=1)
            #     resize = True
            resize_filt = cv2.resize(filt_result, (1550, filt_result.shape[0]), interpolation=cv2.INTER_AREA)
            cv2.imshow("Resize gray stitch", resize_filt)
            cv2.waitKey(0)

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
        # cv2.imshow("hp Image %d" % i, hp_img)
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
        (h, w) = img.shape[:2]
        crop_mean = mean[0:h, 0:w]
        # remove the mean
        mean_ar = np.array(crop_mean)[:, :, 1]
        img_ar = np.array(img)
        minval = np.amin(img_ar.astype(float) - mean_ar.astype(float))
        hp_img = np.uint8(img_ar.astype(float) - mean_ar.astype(float) + minval)
        return hp_img


    def detectAndDescribe(self, image):
        global i
        i += 1
        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

            # sift = cv2.SIFT()
            # (kps, features) = sift.detectAndCompute(gray, None)


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

        kp_img = self.drawKeypoints(image, kps)
        cv2.imshow("kp %d" % i, kp_img)
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

    # stick the left image on the stitched image
    # for grayscale image only
    def stickLeftImage(self, left, warp):
        (h, w) = warp.shape[:2]
        (x, y) = left.shape[:2]
        result = np.zeros(shape=(h, w), dtype="uint8")
        for j in range(0, h - 1):
            for k in range(0, w - 1):
                if j < x and k < y:
                    if left[j, k] == 0:
                        result[j, k] = warp[j, k]
                    else:
                        result[j, k] = left[j, k]

                else:
                    result[j, k] = warp[j, k]
        return result


    def stickLeftImageInColor(self, left, warp):
        # for colorful image only

        (h, w) = warp.shape[:2]
        (x, y) = left.shape[:2]
        result = np.zeros(shape=(h, w, 3), dtype="uint8")
        # pixR = result.load()
        for j in range(0, h - 1):
            for k in range(0, w - 1):
                if j < x and k < y:
                    if left[j, k, :].all() == 0:
                        result[j, k, :] = warp[j, k, :]
                    else:
                        result[j, k, :] = left[j, k, :]

                else:
                    result[j, k, :] = warp[j, k, :]
        return result


    def stickLeft(self, gr_left, gr_warp, co_left, co_warp):
        (h, w) = gr_warp.shape[:2]
        (x, y) = gr_left.shape[:2]
        gr_result = np.zeros(shape=(h, w), dtype="uint8")
        co_result = np.zeros(shape=(h, w, 3), dtype="uint8")
        for j in range(h):
            for k in range(w):
                if j < x and k < y:
                    if gr_left[j, k] == 0:
                        gr_result[j, k] = gr_warp[j, k]
                        co_result[j, k, :] = co_warp[j, k, :]
                    else:
                        gr_result[j, k] = gr_left[j, k]
                        co_result[j, k, :] = co_left[j, k, :]
                else:
                    gr_result[j, k] = gr_warp[j, k]
                    co_result[j, k, :] = co_warp[j, k, :]
        return gr_result, co_result



    def stickByTranslation(self, imageL, imageR, transx, transy):
        dim = imageR.ndim
        if dim == 3:  # RGB image
            #   TODO remove the shading before comparing the values
            for d in range(0, 3):
                imgL = imageL[:, :, d]
                imgR = imageR[:, :, d]
                self.moderateColor(imgL, imgR, transx, transy)
                imageL[:, :, d] = imgL
                imageR[:, :, d] = imgR
        else:   # grayscale image
            self.moderateColor(imageL, imageR, transx, transy)

        (h, w) = imageL.shape[:2]
        (y, x) = imageR.shape[:2]
        if transy < 0:  # R above
            # print "R above"
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
            # print "L above"
            a = h - transy
            b = x + transx
            if dim == 3:
                result = np.zeros((a, b, 3), dtype="uint8")
                result[:, transx:b, :] = imageR[0:a, :, :]
                result[:, 0:w, :] = imageL[transy:h, 0:w, :]
            else:
                result = np.zeros((a, b), dtype="uint8")
                result[:, transx:b] = imageR[0:a, :]
                result[:, 0:w] = imageL[transy:h, 0:w]

        return result


    def moderateColor(self, imageL, imageR, transx, transy):
        #   calculate the difference between moderation_const-pixels in the 2 images
        row_diff = 0
        if transx > 3:
            if transy > 0:
                for r in range(0, moderation_const):
                    print "imageR[0, r] = %d" % (imageR[0, r])
                    print "imageL[transx, r + transy] = %d" % (imageL[transx, r + transy])
                    row_diff += float(imageR[0, r]) - float(imageL[transx, r + transy])
                    print row_diff
            else:  # R above
                for r in range(0, moderation_const):
                    row_diff += float(imageR[0, r - transy]) - float(imageL[transx, r])
            row_diff = row_diff / moderation_const

            # subtract the difference from one image
            if row_diff > 0.0:
                imageR -= int(row_diff)
            else:   #   imageR is more dark
                imageR += int(abs(row_diff))
        else:
            print "got the same images"


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
                if abs(transy[j] - meany) <= stdy and abs(transy[j]) < 10 and transx[j] > 200:
                    temp_transx.append(transx[j])
                    temp_transy.append(transy[j])
            transx = temp_transx
            transy = temp_transy
            stdy = abs(np.std(transy, axis=0))
            # plt.plot(transx, 'r^')
            # plt.plot(transy, 'bs')
            # plt.show()

            k += 1

        translationx = int(math.ceil(np.mean(transx, axis=0)))
        translationy = int(math.ceil(np.mean(transy, axis=0)))

        return translationx, translationy




