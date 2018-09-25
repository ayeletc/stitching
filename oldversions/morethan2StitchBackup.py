# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png

# import the necessary packages
# from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2
import numpy as np


class Stitcher3:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    def stitch3(self, images, ratio=0.75, reprojThresh=4.0,
               showMatches=False):

        imageA = images[0]     # left image
        del images[0]
        leng = len(images)
        i = 1
        while leng > 0:
            print i
            i += 1
            imageB = images[0]
            del images[0]
            leng -= 1

            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)

            # match features between the two images
            M = self.matchKeypoints(kpsB, kpsA,
                                    featuresB, featuresA, ratio, reprojThresh)

            # if the match is None, then there aren't enough matched
            # keypoints to create a panorama
            if M is None:
                print "no keypoints"
                return None

            # otherwise, apply a perspective warp to stitch the images together
            (matches, H, status) = M
            result = cv2.warpPerspective(imageA, H,
                                         (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
            cv2.imshow("midresult", result)
            cv2.waitKey(0)
            # perspectiveSize = getPerspectiveTransform()
            result[0:imageA.shape[0], 0:imageA.shape[1]] = imageA

            # check to see if the keypoint matches should be visualized
            if showMatches:
                vis = self.drawMatches(imageB, imageA, kpsB, kpsA, matches,
                                       status)

            imageA = result[0:550, :]
            cv2.imshow("stitched", imageA)
            cv2.waitKey(0)

        # return the stitched image

        return result, vis

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsB, kpsA, featuresB, featuresA,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsB = np.float32([kpsB[i] for (_, i) in matches])
            ptsA = np.float32([kpsA[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageB, imageA, kpsB, kpsA, matches, status):
        # initialize the output visualization image
        (hB, wB) = imageB.shape[:2]
        (hA, wA) = imageA.shape[:2]
        vis = np.zeros((max(hB, hA), wB + wA, 3), dtype="uint8")
        vis[0:hB, 0:wB] = imageB
        vis[0:hA, wB:] = imageA

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptB = (int(kpsB[queryIdx][0]), int(kpsB[queryIdx][1]))
                ptC = (int(kpsA[trainIdx][0]) + wB, int(kpsA[trainIdx][1]))
                cv2.line(vis, ptB, ptC, (0, 255, 0), 1)

        # return the visualization
        return vis
