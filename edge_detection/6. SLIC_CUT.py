import skimage
import numpy as np
import cv2
import time

img = skimage.data.coffee()

start = time.time()

slic = skimage.segmentation.slic(img, compactness = 20, n_segments = 600, start_label = 1)

#skimage의 버전이 높아져 future는 더 이상 사용하지 않음.
g = skimage.graph.rag_mean_color(img, slic, mode = 'similarity')
ncut = skimage.graph.cut_normalized(slic, g)
print(img.shape, 'image를 분할하는 데', time.time()-start, '초 소요')

marking = skimage.segmentation.mark_boundaries(img,ncut)
ncut_coffee = np.uint8(marking*255.0)

#ncut_coffe는 이미 컬러 이미지이기 때문에 변환이 필요 없음
cv2.imshow('Normalized cut', ncut_coffee)

cv2.waitKey()
cv2.destroyAllWindows()
