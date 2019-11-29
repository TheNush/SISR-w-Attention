import numpy as np
import cv2
import os


def crop(im_path):
	img = cv2.imread(im_path)
	# print(img.shape)
	height, width, channel = img.shape
	crop_img = img[int(height/2)-24:int(height/2)+24, int(width/2)-24:int(width/2)+24]
	# print(crop_img.shape)
	return crop_img

def prep(flag):
	if flag=='train':
		train_path = "D:/Storage/ML Datasets/DIV2K/DIV2K_train_LR_wild/"
		target_path = "D:/Storage/ML Datasets/DIV2K/DIV2K_train_HR"
	elif flag=='test':
		train_path = "D:/Storage/ML Datasets/DIV2K/DIV2K_valid_LR_wild/"
		target_path = "D:/Storage/ML Datasets/DIV2K/DIV2K_valid_HR"
	x = []
	y = []
	i = 0
	for file in os.listdir(train_path):
		path = os.path.join(train_path, file)
		x_img = crop(path)

		y_id = file[:4] + '.png'
		path = os.path.join(target_path, y_id)
		y_img = crop(path)

		x.append(x_img)
		y.append(y_img)

	x = np.asarray(x)
	y = np.asarray(y)
	print(x.shape)
	print(y.shape)
	return (x,y)
	# print(os.listdir(path))

def save_files(x,filename):
	np.savez_compressed(filename, a=x)

def main():
	(x,y) = prep(flag='test')
	save_files(x, filename='C:/Users/TheNush07/Desktop/Work/Projects/XRAY- Image Super Resolution/val/samples.npz')
	save_files(y, filename='C:/Users/TheNush07/Desktop/Work/Projects/XRAY- Image Super Resolution/val/labels.npz')

if __name__ == "__main__":
	main()
