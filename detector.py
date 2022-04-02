# import argparse
import cv2
from collections import namedtuple

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to input image")
# args = vars(ap.parse_args())
args = dict()
args["image"] = "bulbs4.jpeg"

img = cv2.imread(args["image"])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
generalThresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
adaptiveThresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 10
)
dilated = cv2.dilate(generalThresh, None, iterations=2)
# cv2.imshow("Image", dilated)
# cv2.waitKey(0)
output = cv2.connectedComponentsWithStats(dilated, 4, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

BrightDetector = namedtuple("BrightDetector", ["x", "y", "w", "h", "area", "centroid"])
bright_spots: list[BrightDetector] = []
for i, (x1, y1, w, h, area) in enumerate(stats[1:]):
    height, width = img.shape[:2]
    if area > height * width / 2000:
        bright_spots.append(
            BrightDetector(x1, y1, w, h, area, centroids[i+1])
        )

brightest = sorted(bright_spots, key=lambda x: x.area, reverse=True)[0]
bright_spots.remove(brightest)
for i, spot in enumerate(bright_spots):
    r = spot.w if spot.w > spot.h else spot.h
    cv2.circle(img, (int(spot.centroid[0]), int(spot.centroid[1])), r, (0, 0, 255), 2)
    cv2.putText(
        img,
        "Spot #{}".format(i + 1),
        (int(spot.centroid[0] - r), int(spot.centroid[1] - r)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )
cv2.circle(
    img,
    (int(brightest.centroid[0]), int(brightest.centroid[1])),
    brightest.w,
    (0, 255, 0),
    2,
)
cv2.putText(
    img,
    "Brightest Spot",
    (
        int(brightest.centroid[0] - brightest.w),
        int(brightest.centroid[1] - brightest.w),
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)
cv2.imshow("Image", img)
cv2.waitKey(0)
