import cv2
import numpy as np
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    print(line_img)
    draw_lines(line_img, lines)
    return line_img


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function draws `lines` with `color` and `thickness`.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def houghDefaults(img):
    masked_image = img
    # hough lines
    rho = 5
    theta = np.pi / 180
    threshold = 5
    min_line_len = 30
    max_line_gap = 30

    houged = hough_lines(masked_image, rho, theta,
                         threshold, min_line_len, max_line_gap)
    cv2.imshow("houghed", houged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    inputImagePath = '/home/pc/Documents/pytorch-hed/out.png'
    a = cv2.imread(inputImagePath, 0)
    low_threshold = 0
    high_threshold = 255
    a = cv2.Canny(a, low_threshold, high_threshold)
    cv2.imshow("asd", a)
    houghDefaults(a)