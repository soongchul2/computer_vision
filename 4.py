import cv2 as cv
import sys
img=cv.imread('/Users/win/Desktop/COMPUTER VISION/mooc.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray_small=cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5)

cv.imwrite('mooc_gray.jpg', gray)
cv.imwrite('mooc_gray_small.jpg', gray_small)

cv.imshow('Color image', img)
cv.imshow('Gray image', gray)
cv.imshow('Gray image small', gray_small)

cv.waitKey()
cv.destroyAllWindows()