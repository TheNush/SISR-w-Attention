import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision.transforms import Compose, CenterCrop

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim

device = 'cuda'

class res_block(nn.Module):
	def __init__(self, in_channel):
		super(res_block, self).__init__()
		self.bottleneck = nn.Conv2d(in_channel, 10, kernel_size=1, padding=0)
		self.conv1 = nn.Conv2d(10, 8, kernel_size=3, stride=1, padding=1)
		self.batchnorm1 = nn.BatchNorm2d(8)
		self.conv2 = nn.Conv2d(8, 6, kernel_size=3, stride=1, padding=1)
		self.batchnorm2 = nn.BatchNorm2d(6)
		# self.bottleneck = nn.Conv2d()
	def forward(self, x):
		temp = x
		bottleneck = self.bottleneck(temp)
		conv1 = self.conv1(bottleneck)
		batchnorm1 = self.batchnorm1(conv1)
		relu = F.relu(batchnorm1)
		conv2 = self.conv2(batchnorm1)
		batchnorm2 = self.batchnorm2(conv2)
		output = torch.cat([batchnorm2, temp], 1)
		# print(shape)
		return output

class attention_net(nn.Module):
	def __init__(self, initial=16):
		super(attention_net, self).__init__()
		# self.initial = initial
		self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
		self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
		self.one = res_block(initial)
		self.two = res_block(initial+6)
		self.three = res_block(initial+12)
		self.deconv1 = nn.ConvTranspose2d(initial+18, int((initial//2)+9), kernel_size=3, stride=2, padding=1)
		self.fourth = res_block(int(1.5*initial)+21)
		self.deconv2 = nn.ConvTranspose2d(int(1.5*initial)+27, int(0.75*initial)+13, kernel_size=3, stride=2, padding=1)
		self.fifth = res_block(int(1.75*initial)+19)
		self.deconv3 = nn.ConvTranspose2d(int(1.75*initial)+25, int(0.875*initial)+12, kernel_size=3, stride=2, padding=1)
		self.conv3 = nn.Conv2d(int(0.875*initial)+28, 16, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(conv1)
		m_pool = self.max_pool(conv2)
		first = self.one(m_pool)
		a_pool = self.avg_pool(first)
		second = self.two(a_pool)
		a_pool = self.avg_pool(second)
		third = self.three(a_pool)

		upsam = self.deconv1(third, output_size=(second.shape[2], second.shape[3]))
		upsam = torch.cat([upsam, second], axis=1)
		fourth = self.fourth(upsam)
		upsam = self.deconv2(fourth, output_size=first.shape)
		upsam = torch.cat([upsam, first], axis=1)
		fifth = self.fifth(upsam)
		upsam = self.deconv3(fifth, output_size=conv2.shape)
		upsam = torch.cat([upsam, conv2], axis=1)
		conv3 = self.conv3(upsam)
		# print(conv3.shape)
		conv4 = self.conv4(conv3)
		# print(conv4.shape)
		return conv4

class feature_extraction(nn.Module):
	def __init__(self, initial=16, upscale_factor=4):
		super(feature_extraction, self).__init__()
		self.conv1 = nn.Conv2d(3, initial, kernel_size=3, stride=1, padding=1)

		self.denres1 = dense_res_block(initial)
		self.denres2 = dense_res_block(initial+24)
		self.denres3 = dense_res_block(initial+48)
		self.denres4 = dense_res_block(initial+72)
		self.denres5 = dense_res_block(initial+96)
		self.denres6 = dense_res_block(initial+120)

		self.sub_pixel = nn.PixelShuffle(upscale_factor=upscale_factor)
		self.conv2 = nn.Conv2d(int(initial/8)+9, 3, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		temp = x
		conv1 = self.conv1(temp)
			
		first = self.denres1(conv1)
		second = self.denres2(first)
		third = self.denres3(second)
		fourth = self.denres4(third)
		fifth = self.denres5(fourth)
		sixth = self.denres6(fifth)

		concat = torch.cat([sixth, conv1], axis=1)
		sub_pixel = self.sub_pixel(concat)
		output = self.conv2(sub_pixel)

		return output

class combined_model(nn.Module):
	def __init__(self):
		super(combined_model, self).__init__()
		self.attention = attention_net()
		self.feat_model = feature_extraction()

	def forward(self, x_bic, x):
		feat_recon = self.feat_model(x)
		att = self.attention(x_bic)
		result = torch.mul(feat_recon, att) + x_bic
		return result


# def ri(path):
	# # img = cv2.imread(path)
	# img = crop(path)
	# img = img.reshape(1, img.shape[2], img.shape[1], img.shape[0])
	# # print(img.shape)
	# img = torch.tensor(img, device=device).float()
	# return img

def dense_res_block(initial):
	model = res_block(in_channel=initial)
	model = nn.Sequential(model, res_block(initial + 6), res_block(initial + 12), res_block(initial +18))#, res_block(initial+24), res_block(initial+30))
	return model

# def feature_net(initial=16):
# 	# input_layer = nn.Conv2d(3, initial, kernel_size=3, stride=1, padding=1)
# 	model = nn.Sequential(
# 						  dense_res_block(initial=initial),
# 						  dense_res_block(initial=initial+24),
# 						  dense_res_block(initial=initial+48),
# 						  dense_res_block(initial=initial+72),
# 						  dense_res_block(initial=initial+96),
# 						  dense_res_block(initial=initial+120),
# 						  )
# 	# print(model)
# 	return model

# def sub_pixel(result, upscale_factor=4):
# 	pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
# 	return pixel_shuffle(result)

# def convolve(x, in_channel):
# 	model = nn.Conv2d(in_channel, 3, kernel_size=3, stride=1, padding=1)
# 	model = model.to(device)
# 	return model(x)

def show_img(tensor):
	tensor = tensor.to('cpu')
	img = tensor.detach().numpy()
	img = img.reshape(img.shape[2], img.shape[3], img.shape[1])
	cv2.imshow('res', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# plt.imshow(img)
	# plt.show()

# def feature_reconstruction(initial=16, upscale_factor=4):
# 	# feat_model = feature_net().to(device)
# 	# img = ri(im_path)
# 	# input_layer = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1).to(device)
# 	# op1 = input_layer(img)
# 	# result = feat_model(op1)
# 	# print(result.shape)
# 	# result = torch.cat([result, op1], axis=1)
# 	# print(result.shape)
# 	# result = sub_pixel(result)
# 	# print("Result shape", result.shape)
# 	# feat_result = convolve(result, result.shape[1])
# 	feat_model = feature_extraction(initial=initial, upscale_factor=upscale_factor).to(device)
# 	# feat_result = feat_model(img)
# 	# show_img(result)
# 	# print("Image Shape", img.shape)
# 	# print(feat_result.shape)
# 	# model = nn.Sequential(feat_model, sub_pixel(), )
# 	return feat_model

# def attention_output(input):
	# # input = torch.tensor(input, device=device).float()
	# model = attention_net(initial=16).to(device)
	# # print(model)
	# result = model(input)
	# return result

# def bicubic_inter(df):
# 	# img = crop(im_path)
# 	# print("Image pre-inter shaep", img.shape)
# 	for i in range(df.shape[0]):
# 		img = df[i,:,:,:]
# 		df[i,:,:,:] = cv2.resize(img, (img.shape[0]*4, img.shape[1]*4), interpolation=cv2.INTER_CUBIC)
# 		print()
# 	# cv2.imshow('bicubic', bicubic)
# 	# cv2.waitKey(0)
# 	# cv2.destroyAllWindows()
# 	bicubic = bicubic.reshape(1, bicubic.shape[2], bicubic.shape[0], bicubic.shape[1])
# 	bicubic = torch.tensor(bicubic, device=device).float()
# 	# print("Bicubic O/P shape ", bicubic.shape)
# 	return bicubic

# def crop(im_path):
	# img = cv2.imread(im_path)
	# print(img.shape)
	# height, width, channel = img.shape
	# crop_img = img[int(height/2)-24:int(height/2)+24, int(width/2)-24:int(width/2)+24]
	# print(crop_img.shape)
	# return crop_img
	# cv2.imshow('crop', crop_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

# def model():
	# im_path = "D:/Storage/ML Datasets/DIV2K/DIV2K_train_LR_wild/0005x4w1.png"
	# # im_path = "D:/Storage/ML Datasets/DIV2K/DIV2K_train_HR/0004.png"
	
	# recon = feature_reconstruction(im_path)
	# print("Recon O/P shape ", recon.shape)
	# bicubic = bicubic_inter(im_path)
	# att = attention_output(bicubic)
	# print("Attention O/P shape", att.shape)
	# result = torch.mul(recon, att) + bicubic
	# print(result.shape)
	# show_img(result)
	# crop(im_path)

def load_data():
	x_train_bic = np.load("C:/Users/TheNush07/Desktop/Work/Projects/XRAY- Image Super Resolution/train/samples_bic.npz")
	x_train_bic = x_train_bic['a']
	x_train = np.load("C:/Users/TheNush07/Desktop/Work/Projects/XRAY- Image Super Resolution/train/samples.npz")
	x_train = x_train['a']
	y_train = np.load("C:/Users/TheNush07/Desktop/Work/Projects/XRAY- Image Super Resolution/train/labels.npz")
	y_train = y_train['a']

	x_test_bic = np.load("C:/Users/TheNush07/Desktop/Work/Projects/XRAY- Image Super Resolution/test/samples_bic.npz")
	x_test_bic = x_test_bic['a']
	x_test = np.load("C:/Users/TheNush07/Desktop/Work/Projects/XRAY- Image Super Resolution/test/samples.npz")
	x_test = x_test['a']
	y_test = np.load("C:/Users/TheNush07/Desktop/Work/Projects/XRAY- Image Super Resolution/test/labels.npz")
	y_test = y_test['a']

	return x_train_bic, x_train, y_train, x_test_bic, x_test, y_test

# def create_model():
# 	model = combined_model()
# 	criterion = nn.L1Loss()
# 	optimizer = nn.Adam(model.parameters(), lr=0.9)
# 	print(model)
# 	return model

def adjust_shape(df):
	df = df.reshape(df.shape[0], df.shape[3], df.shape[1], df.shape[2])
	return df

def main():
	print("===> Loading Data")
	x_train_bic, x_train, y_train, x_test_bic, x_test, y_test = load_data()
	x_train_bic = adjust_shape(x_train_bic)
	x_train = adjust_shape(x_train)
	y_train = adjust_shape(y_train)
	x_test_bic = adjust_shape(x_test_bic)
	x_test = adjust_shape(x_test)
	y_test = adjust_shape(y_test)
	x_train = torch.tensor(x_train, device=device).float()
	x_train_bic = torch.tensor(x_train_bic, device=device).float()
	y_train = torch.tensor(y_train, device=device).float()
	x_test = torch.tensor(x_test, device=device).float()
	x_test_bic = torch.tensor(x_test_bic, device=device).float()
	y_test = torch.tensor(y_test, device=device).float()
	print("===> Preping Model")
	model = combined_model().to(device)
	print("===> Using MAE(L1) Loss")
	criterion = nn.L1Loss()
	# criterion = -ssim()
	print("===> Adam for optimization")
	optimizer = optim.Adam(model.parameters(), lr=0.2)

	train_batch_size = 8
	test_batch_size = train_batch_size // 2
	num_workers = 0

	train_data = torch.utils.data.TensorDataset(x_train_bic, x_train, y_train)
	test_data = torch.utils.data.TensorDataset(x_test_bic, x_test, y_test)
	# train_data = [x_train_bic, x_train, y_train]
	# test_data = [x_test_bic, x_test, y_test]
	# print(len(train_data))

	train_indices = list(range(len(train_data[0])))
	print(len(train_data[0]))
	print(len(train_indices))
	np.random.shuffle(train_indices)
	# print(len(train_indices))

	test_indices = list(range(len(test_data[0])))
	np.random.shuffle(test_indices)
	# print(len(test_indices))

	train_sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(x_train.shape[0])))
	test_sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(x_test.shape[0])))
	print(x_train.shape[0])

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, sampler=train_sampler, num_workers=num_workers)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, sampler=test_sampler, num_workers=num_workers)
	print(train_loader)

	train(model, optimizer, criterion, train_loader)

def train(model, optimizer, criterion, train_loader, epochs=100, batch_size=8, steps=400):
	for epoch in range(epochs):
		train_loss = 0.0
		for i, data in enumerate(train_loader):
			x_train_bic, x_train, y_train = data[0], data[1], data[2]
			x_train_bic = x_train_bic.requires_grad_(False)
			x_train = x_train.requires_grad_(False)
			y_train = y_train.requires_grad_(False)
			optimizer.zero_grad()

			output = model(data[0], data[1])
			# loss = ssim(output, data[2], data_range=data[2].max()-data[2].min(), multichannel=True)
			loss = criterion(output, data[2])
			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			if i%100 == 0:
				print("{}/{} loss: {}".format(epoch+1, epochs+1, train_loss/100))
				train_loss = 0.0
	print("Hey we made it!")

if __name__ == "__main__":
	main()
