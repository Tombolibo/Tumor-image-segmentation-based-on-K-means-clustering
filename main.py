import numpy as np
import cv2

img = cv2.imread(r'./tumor.png')
imgSrc = img.copy()
print('img.shape: ', img.shape)

#设定分类停止阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)  #查看官方文档示意

#准备数据
data = img.reshape((-1,3)).astype(np.float32)  #data.shape -> (行*列，通道）
print('data.shape: ', data.shape)
print('data.dtype: ', data.dtype)

#进行k均值聚类
K = 5
dataDistance, dataLabel, dataCenter = cv2.kmeans(data, K, None,
                                                 criteria, 5, cv2.KMEANS_RANDOM_CENTERS)  #返回每个点到中心距离平方和(float)、每个点类别，各个类别的中
print('dataDistance: ', dataDistance)
print('dataLabel.shape: ', dataLabel.shape)
print('dataCenter.shape: ', dataCenter.shape)
print('center:\n', dataCenter)

#聚类结果可视化
imgKmeans = dataCenter[dataLabel].reshape(img.shape).astype(np.uint8)
# imgKmeans = cv2.medianBlur(imgKmeans, 7, None)
print('imgKmeans.shape: ', imgKmeans.shape)

#展示结果
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.namedWindow('imgKmeans', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
cv2.imshow('imgKmeans', imgKmeans)
cv2.waitKey(0)

