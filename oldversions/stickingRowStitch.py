import imutils
import cv2
from PIL import Image
import numpy as np
import copy
import math
import matplotlib.pyplot as plt



resolution_factor = 1
i = 0
Path = "C:\\Users\\ayele\\Documents\\Ayelet\\Technion\\python\\my code\\images\\"


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojThresh=1.0,
               showMatches=False):
        filtered = self.filterList(images)  # list of filtered images
        # filtered = images
        j = 0
        num = len(images)
        imageL = images[j]  # left  image
        filtL = filtered[j]
        while num - 1 > j:
            imageR = images[j + 1]  # right image
            filtR = filtered[j + 1]

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

            # otherwise, apply a perspective warp to stitch the images together

            (matches, H, status) = M

            color_warp = cv2.warpPerspective(imageR, H,
                                             (imageR.shape[1] + imageL.shape[1], imageR.shape[0]))
            filt_warp = cv2.warpPerspective(filtR, H,
                                            (filtR.shape[1] + filtL.shape[1], filtR.shape[0]))

            # cv2.imshow("color stitch %d" % j, color_result)
            # cv2.waitKey(0)
            # cv2.imshow("filtered stitch %d" % j, filt_result)
            # cv2.waitKey(0)

            #   TODO repair the stitch by if imageL[j][k] != 0 : color_result[j][k] = imageL[j][k] else ----

            (filt_result, color_result) = self.stickLeft(filtL, filt_warp, imageL, color_warp)

            # crop black borders as the right side
            gray = cv2.cvtColor(color_result, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)
            color_crop = color_result[0:y + h, 0:x + w]
            filt_crop = filt_result[0:y + h, 0:x + w]
            # check to see if the keypoint matches should be visualized
            vis = None
            if showMatches:
                vis = self.drawMatches(imageR, imageL, kpsR, kpsL,
                                       matches, status)
            imageL = color_crop
            filtL = filt_crop
            cv2.imshow("gray stitch", filtL)
            cv2.waitKey(0)
            j += 1

        # return the stitched image
        return imageL, vis

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
        # cv2.imshow("gray Image %d" % i, gray_img)
        # cv2.waitKey(0)

        # change resolution
        reso_img = cv2.resize(gray_img, (0, 0), fx=resolution_factor, fy=resolution_factor)
        # cv2.imshow("resolution Image %d" % i, reimg)
        # cv2.waitKey(0)

        # high pass image
        # lp_img = cv2.GaussianBlur(reso_img, (51, 51), 0)
        # lp_img = np.array(lp_img)
        # reso_img = np.array(reso_img)
        # minval = np.amin(reso_img.astype(float) - lp_img.astype(float))
        # hp_img = np.uint8(reso_img.astype(float) - lp_img.astype(float) + minval)
        hp_img = self.highPass(reso_img)
        # cv2.imshow("hp Image %d" % i, hp_img)
        # cv2.waitKey(0)

        # blur
        gauss = cv2.GaussianBlur(hp_img, (7, 7), 0)
        # cv2.imshow("blur", gauss)
        # cv2.waitKey(0)
        # cv2.imshow("gaussian Image %d" % i, gauss)
        # cv2.waitKey(0)

        # # histogram equalization
        # equ = cv2.equalizeHist(gauss)
        # cv2.imshow("equal Image %d" % i, equ)
        # cv2.waitKey(0)

        # p = -2.5
        # kernel = np.array([[p, p, p], [p, -(8*p - 1), p], [p, p, p]])
        # sharp4 = cv2.filter2D(gauss, -1, kernel)
        h = -3.0
        l = 1.0
        kernel1 = np.array([[l, l, l], [h, -(3 * h + 5 * l) + 1, h], [l, l, h]])
        sharp1 = cv2.filter2D(gauss, -1, kernel1)

        kernel2 = np.array([[l, h, h], [l, -(3 * h + 5 * l) + 1, l], [l, h, l]])
        sharp2 = cv2.filter2D(sharp1, -1, kernel2)
        # kernel3 = np.array([[h, l, l], [l, -(2 * h + 6 * l) + 1, l], [l, l, h]])
        # sharp3 = cv2.filter2D(sharp2, -1, kernel3)
        #
        # kernel4 = np.array([[l, l, h], [l, -(2 * h + 6 * l) + 1, l], [h, l, l]])
        # sharp4 = cv2.filter2D(sharp3, -1, kernel4)


        # cv2.imshow("sharp Image %d" % i, sharp2)
        # cv2.waitKey(0)
        # cv2.imwrite(Path + "clean\\im %d.png" % i, sharp)
        return sharp2

    def highPass(self, img):
        sample_num = 1
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        mean = cv2.imread(Path + "mean_s%d\\" % sample_num + "mean_img.png")
        # adjust the mean to the img size
        (h, w) = img.shape[:2]
        crop_mean = mean[0:h, 0:w]

        # cv2.imshow("mean_img", mean)
        # cv2.waitKey(0)
        mean_ar = np.array(crop_mean)[:, :, 1]
        img_ar = np.array(img)
        minval = np.amin(img_ar.astype(float) - mean_ar.astype(float))
        hp_img = np.uint8(img_ar.astype(float) - mean_ar.astype(float) + minval)
        return hp_img

        #
        # hp_img = np.uint8(((255-(img_ar.astype(float) - mean_ar.astype(float))) / 3 + minval))
        # hp_img = np.uint8((255 - (img_ar.astype(float) - mean_ar.astype(float) + minval)) / 2)
        #



        # (h, w) = img_ar.shape
        # hp_img = np.zeros(shape=(h, w))
        # row = 0
        # el = 0
        # iff = 0
        # while row < h:
        #     col = 0
        #     while col < w:
        #         im = img_ar[row, col]
        #         me = mean_ar[row, col]
        #         if (im > (me - 5)) and (im < (me + 5)):
        #             iff += 1
        #             hp_img[row, col] = abs(minval)
        #         else:
        #             el += 1
        #             hp_img[row, col] = im
        #
        #
        #         col += 1
        #     row += 1
        # print "else = %d" % el + " if = %d" % iff
        # hp_img = np.uint8(hp_img)

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

        # kp_img = self.drawKeypoints(image, kps)
        # cv2.imshow("kp %d" % i, kp_img)
        # cv2.waitKey(0)

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

        # transx, transy = self.findTranslation(matches, kpsR, kpsL)

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsR = np.float32([kpsR[j] for (_, j) in matches])
            ptsL = np.float32([kpsL[j] for (j, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsR, ptsL, cv2.RANSAC,
                                             reprojThresh)

            # A = H[0, 0]
            # B = H[0, 1]
            # D = H[1, 0]
            # E = H[1, 1]
            #
            # H[0, 0] = A / math.sqrt(A*A+B*B)
            # H[0, 1] = B / math.sqrt(A*A+B*B)
            # H[1, 0] = D / math.sqrt(D*D+E*E)
            # H[1, 1] = E / math.sqrt(D*D+E*E)
            # H[2, 0] = -0.000000
            # H[2, 1] = -0.000000
            # H[2, 2] = 1.00000

            # teta = 0.00001
            # H[0, 0] = math.cos(teta)
            # H[0, 1] = -math.sin(teta)
            # H[1, 0] = math.sin(teta)
            # H[1, 1] = math.cos(teta)
            # H[2, 0] = -0.000000
            # H[2, 1] = -0.000000
            # H[2, 2] = 1.00000

            # return the matches along with the homography matrix
            # and status of each matched point
            return matches, H, status

        # otherwise, no homography could be computed
        return None

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
        # TODO  check why there is still zeros on the  stitching
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
        # TODO  check why there is still zeros on the  stitching
        # TODO make this def useful for colorful img also
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


    def findTranslation(self, matches, kpsR, kpsL):
        transx = []
        transy = []
        for (trainIdx, queryIdx) in matches:
            ptR = (int(kpsR[queryIdx][0]), int(kpsR[queryIdx][1]))
            ptL = (int(kpsL[trainIdx][0]), int(kpsL[trainIdx][1]))
            x = abs(ptR[0] - ptL[0])
            y = abs(ptR[1] - ptL[1])
            # TODO remove the abs!
            transx.append(x)
            transy.append(y)

        # plt.plot(transx, 'ro')
        # plt.plot(transy, 'ro')
        # plt.show()

        meanx = np.mean(transx, axis=0)

        trnsx = []
        trnsy = []
        num = len(transx) - 1
        for j in range(0, num):
            if abs(transx[j] - meanx) < 10:
                trnsx.append(transx[j])
                trnsy.append(transy[j])

        translationx = math.ceil(np.mean(trnsx, axis=0))
        translationy = math.ceil(np.mean(trnsy, axis=0))


        # plt.plot(trnsx, 'ro')
        # plt.plot(trnsy, 'ro')
        # plt.show()
        #
        return translationx, translationy

