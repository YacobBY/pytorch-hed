import numpy as np
import scipy.signal
import scipy
import cv2
from skimage import exposure
from skimage import img_as_ubyte

def tresholdBorders(image):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
  morph = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
  morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
  # take morphological gradient
  gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

  # # split the gradient image into channels
  # image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
  # channel_height, channel_width, _ = image_channels[0].shape
  #
  # # apply Otsu threshold to each channel
  # for i in range(0, 3):
  #   _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
  #   image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))
  #
  # # merge the channels
  # filteredImage = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
  filteredImage = cv2.bilateralFilter(gradient_image, 30, 25, 255)
  return filteredImage

if __name__ == '__main__':
  img =  cv2.imread('/home/pc/Documents/BindbandDetection/PhoneCamPics2/out.png')  # Load the image

  #CLAHE Doesn't seem to improve detection of lines, only introduce more noise.
  # hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  # h, s, v = cv2.split(hsv_image)
  # np.clip(v, a_min = 0, a_max = 240)
  # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
  # v = clahe.apply(v)
  # hsv_image = cv2.merge([h, s, v])
  # img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

  img = cv2.bilateralFilter(img, 30, 150, 150)
  img = originalImage = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)

  # img = cv2.resize(img, (0, 0), fx=2, fy=2)       #resize image

  filteredImage = tresholdBorders(img)
  # save the denoised image

  cv2.imshow("Original", originalImage)
  cv2.imshow("bilateralFilter", img)
  cv2.imshow("denoised image", filteredImage)

  tresholdedImageGray = cv2.cvtColor(filteredImage, cv2.COLOR_BGR2GRAY)

  img_blur = cv2.bilateralFilter(originalImage.copy(), d=7,
                                 sigmaSpace=75, sigmaColor=75)

  img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)

  a = img_gray.max()
  _, thresh = cv2.threshold(tresholdedImageGray, a / 2 + 60, a, cv2.THRESH_BINARY_INV)
  contours, hierarchy = cv2.findContours(
    image=thresh,
    mode=cv2.RETR_TREE,
    method=cv2.CHAIN_APPROX_SIMPLE)

  contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Draw the contour
  img_copy = img.copy()
  final = cv2.drawContours(img_copy, contours, contourIdx=-1,
                           color=(255, 0, 0), thickness=2)


  cv2.imshow("tresholding", img_copy)

  cv2.imshow("DrawnLinesResult", tresholdedImageGray)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


"""
  cannyEdges = cv2.Canny(inputImageGray, 0, 255, apertureSize=3)
  cv2.imshow('CannyEdges', cannyEdges)
  lines = cv2.HoughLines(cannyEdges, 1, np.pi / 180, 200)

  for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
    x1 = int(x0 + 1000 * (-b))
    # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
    y1 = int(y0 + 1000 * (a))
    # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
    x2 = int(x0 - 1000 * (-b))
    # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
    y2 = int(y0 - 1000 * (a))
    cv2.line(filteredImage, (x1, y1), (x2, y2), (0, 0, 255), 5)
"""
