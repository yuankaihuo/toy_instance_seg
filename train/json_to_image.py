import json
import math
import numpy as np
import cv2
from PIL import Image


json_file_path = './'

data = json.load(open(json_file_path + '20190403_CL4_EGFP-EPS8_mCh-Espin_005_Decon_MaxIP_crop1_256x256_frame1.json'))
img_rgb = cv2.imread(json_file_path + '20190403_CL4_EGFP-EPS8_mCh-Espin_005_Decon_MaxIP_crop1_256x256_frame1.png')

def cal_distance(x1, y1, x2, y2):
	dist = math.sqrt((float(x2) - float(x1))**2 + (float(y2) - float(y1))**2)
	return dist



def main():
	count = 0
	img = img_rgb[:, :, 0]
	img = np.zeros_like(img)

	dist_set = []
	distribution = np.zeros(31)
	for shape in data['shapes']:
		cor_set = shape['points']
		cor_set = np.array(cor_set, dtype = np.uint8)
		x1 = cor_set[0,0]
		y1 = cor_set[0,1]
		x2 = cor_set[1,0]
		y2 = cor_set[1,1]


		# dist = cal_distance(cor_set[0,0], cor_set[0,1], cor_set[1, 0], cor_set[1,1])
		# dist = int(dist)
		# dist_set.append(dist)
		# distribution[dist] = distribution[dist]+1

		# print(stick_num)
		# x = np.random.randint(30, 225)
		# y = np.random.randint(30, 225)
		# w = 1
		# # h = np.random.randint(80, 100)
		# h = np.random.randint(10, 30)
		# theta = np.random.randint(-90, 90)
		# rect = ([x, y], [w, h], theta)  # 中心(x,y), (宽,高), 旋转角度
		# box = np.int0(cv2.boxPoints(rect))
		#

		gt = np.zeros_like(img)
		# gt = cv2.fillPoly(gt, [box], 1)

		start_point = (x1, y1)
		end_point = (x2, y2)
		color = (255)
		thickness = 2
		gt = cv2.line(gt, start_point, end_point, color, thickness)
		img = cv2.line(img, start_point, end_point, color, thickness)
		count = count+1

	# img = cv2.resize(img, (128,128))
	img = img[64:168,64:168]

	print('count = %d' % count)
	cv2.imwrite(json_file_path + '20190403_CL4_EGFP-EPS8_mCh-Espin_005_Decon_MaxIP_crop1_256x256_frame1_binary.png', img)


if __name__ == "__main__":
	main()

