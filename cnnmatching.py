import argparse
import cv2
import numpy as np
import imageio
import plotmatch
from lib.cnn_feature import cnn_feature_extract
import matplotlib.pyplot as plt
import time
from skimage import measure
from skimage import transform
import pandas as pd
from PIL import Image

#time count
start = time.perf_counter()

_RESIDUAL_THRESHOLD = 30
#Test1nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8
# imgfile1 = 'df-ms-data/1/df-googleearth-1k-20091227.jpg'
# imgfile2 = 'df-ms-data/1/df-googleearth-500-20181029.jpg'
# imgfile1 = 'df-ms-data/1/df-uav-sar-500.jpg'

imgfile1 = '/media/lab125/HDD11/SAR_dataset/XiangJiang/2600x1500/OPT1.png'
imgfile2 = '/media/lab125/HDD11/SAR_dataset/XiangJiang/2600x1500/SAR1.png'

start = time.perf_counter()

# read left image
image1 = imageio.imread(imgfile1)
image2 = imageio.imread(imgfile2)

print('read image time is %6.3f' % (time.perf_counter() - start))

start0 = time.perf_counter()

kps_left, sco_left, des_left = cnn_feature_extract(image1,  nfeatures = -1)
kps_right, sco_right, des_right = cnn_feature_extract(image2,  nfeatures = -1)

print('Feature_extract time is %6.3f, left: %6.3f,right %6.3f' % ((time.perf_counter() - start), len(kps_left), len(kps_right)))
start = time.perf_counter()

#Flann特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=40)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_left, des_right, k=2)

goodMatch = []
locations_1_to_use = []
locations_2_to_use = []

# 匹配对筛选
min_dist = 1000
max_dist = 0
disdif_avg = 0
# 统计平均距离差
for m, n in matches:
    disdif_avg += n.distance - m.distance
disdif_avg = disdif_avg / len(matches)

for m, n in matches:
    #自适应阈值
    if n.distance > m.distance + disdif_avg:
        goodMatch.append(m)
        p2 = cv2.KeyPoint(kps_right[m.trainIdx][0],  kps_right[m.trainIdx][1],  1)
        p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
        locations_1_to_use.append([p1.pt[0], p1.pt[1]])
        locations_2_to_use.append([p2.pt[0], p2.pt[1]])
#goodMatch = sorted(goodMatch, key=lambda x: x.distance)
print('match num is %d' % len(goodMatch))
locations_1_to_use = np.array(locations_1_to_use)
locations_2_to_use = np.array(locations_2_to_use)

# #######################################################
df = pd.read_csv('./global_matches/converted_result_1.csv')
locations_1_to_use = df[['GlobalLeftX', 'GlobalLeftY']].values
locations_2_to_use = df[['GlobalRightX', 'GlobalRightY']].values
locations_1_to_use = np.array(locations_1_to_use)
locations_2_to_use = np.array(locations_2_to_use)

all_indices = np.arange(locations_1_to_use.shape[0])

# #######################################################

# Perform geometric verification using RANSAC.
model, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                          transform.AffineTransform,
                          min_samples=3,
                          residual_threshold=_RESIDUAL_THRESHOLD,
                          max_trials=1000)

print('Found %d inliers' % sum(inliers))

inlier_idxs = np.nonzero(inliers)[0]
#最终匹配结果
matches = np.column_stack((inlier_idxs, inlier_idxs))
print('whole time is %6.3f' % (time.perf_counter() - start0))

# # 计算 RMSE（Root Mean Square Error）这里的RMSE只是指匹配对应的程度
# if sum(inliers) > 0:
#     transformed_points = model(locations_1_to_use[inliers])
#     error = transformed_points - locations_2_to_use[inliers]
#     # 计算每个误差向量的欧几里得范数（距离）
#     error_norm = np.linalg.norm(error, axis=1)
    
#     # 绘制误差的直方图
#     plt.figure(figsize=(8, 5))
#     plt.hist(error_norm, bins=30, color='skyblue', edgecolor='black')
#     plt.title('Histogram of Matching Errors')
#     plt.xlabel('Euclidean Error Distance (pixels)')
#     plt.ylabel('Number of Matches')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('error_histogram.jpg', dpi=200)

#     rmse = np.sqrt(np.mean(np.sum(error ** 2, axis=1)))
#     print(f'RMSE (Root Mean Square Error): {rmse:.4f}')
# else:
#     print('No inliers found. RMSE cannot be computed.')

# 计算 RMSE（Root Mean Square Error）
transformed_points = model(locations_1_to_use[inliers])
error = transformed_points - locations_2_to_use[inliers]
# 计算每个误差向量的欧几里得范数（距离）
error_norm = np.linalg.norm(error, axis=1)
rmse = np.sqrt(np.mean(np.sum(error ** 2, axis=1)))
print(f'RMSE (Root Mean Square Error): {rmse:.4f}')

plt.figure(figsize=(8, 5))
plt.hist(error_norm, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Matching Errors')
plt.xlabel('Euclidean Error Distance (pixels)')
plt.ylabel('Number of Matches')
plt.grid(True)
plt.tight_layout()
plt.savefig('error_histogram.jpg', dpi=200)
plt.show()

# 计算 CMR（Correct Match Ratio）
cmr = sum(inliers) / len(goodMatch) if len(goodMatch) > 0 else 0
print(f'CMR (Correct Match Ratio): {cmr:.4f}')

# # Visualize correspondences, and save to file.
# #1 绘制匹配连线
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
plt.rcParams['figure.figsize'] = (16.0, 9.0) # 设置figure_size尺寸
_, ax = plt.subplots()
plotmatch.plot_matches(
    ax,
    image1,
    image2,
    locations_1_to_use,
    locations_2_to_use,
    np.column_stack((inlier_idxs, inlier_idxs)),
    plot_matche_points = False,
    matchline = False,
    matchlinewidth = 0.3)
ax.axis('off')
ax.set_title('')
plt.savefig('matches.jpg', dpi=200)

#######################################################
# 添加误差圆
from matplotlib.patches import Circle
# 偏移量：右图相对于左图的x轴偏移
offset_x = image1.shape[1]

# 添加误差圆：左图和右图都画
for pt1, pt2, err in zip(locations_1_to_use, locations_2_to_use, error_norm):
    # 左图的误差圆
    circle1 = Circle((pt1[0], pt1[1]), radius=err,
                     edgecolor='red', facecolor='none', linewidth=0.5, alpha=0.6)
    ax.add_patch(circle1)
    # 右图的误差圆（注意 x 偏移）
    circle2 = Circle((pt2[0] + offset_x, pt2[1]), radius=err,
                     edgecolor='red', facecolor='none', linewidth=0.5, alpha=0.6)
    ax.add_patch(circle2)
ax.axis('off')
plt.tight_layout()
plt.savefig('matches_with_errors.jpg', dpi=300)  # 保存高分辨率图像
plt.show()
#######################################################


h2, w2 = image2.shape[:2]
M_all, _ = cv2.estimateAffine2D(locations_1_to_use[inliers], locations_2_to_use[inliers])
registered_all = cv2.warpAffine(image1, M_all, (w2, h2))


registered_img = registered_all  # 默认使用所有内点的仿射变换

# 创建棋盘格掩码
# 定义棋盘格大小（每个格子的边长）
tile_size = 80

# 创建与图像大小相同的掩码
checkerboard = np.zeros((h2, w2), dtype=np.uint8)

# 填充棋盘格
for i in range(0, h2, tile_size):
    for j in range(0, w2, tile_size):
        # 根据位置决定是否填充
        if ((i // tile_size) + (j // tile_size)) % 2 == 0:
            # 检查边界，避免索引超出范围
            end_i = min(i + tile_size, h2)
            end_j = min(j + tile_size, w2)
            checkerboard[i:end_i, j:end_j] = 1

# 将掩码从2D扩展到与图像相同的通道数
if len(image2.shape) == 3:  # 彩色图像
    checkerboard = np.stack([checkerboard] * 3, axis=2)

# 使用棋盘格掩码合并两幅图像
fused_img = registered_img.copy()
if len(image2.shape) == 3 and len(registered_img.shape) == 3:
    # 彩色图像
    fused_img = registered_img * (1 - checkerboard) + image2 * checkerboard
else:
    # 如果图像是灰度的，转换为相同类型
    if len(image2.shape) == 2:
        img2 = image2
    else:
        img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
    if len(registered_img.shape) == 2:
        reg_img = registered_img
    else:
        reg_img = cv2.cvtColor(registered_img, cv2.COLOR_BGR2GRAY)
        
    fused_img = reg_img * (1 - checkerboard) + img2 * checkerboard

# 确保图像数据类型正确
fused_img = fused_img.astype(np.uint8)
image = Image.fromarray(fused_img, 'RGB')
image.save('fused_img.jpg')  # 也可以保存为PNG等其他格式
