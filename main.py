from net import *
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image
import random
import os

def normalize(img):
	n_arr = np.asarray(img)
	n_arr = n_arr.astype(np.float32)
	n_arr /= 255.0
	img = Image.fromarray(n_arr.astype(np.uint8))
	return img


class SR_Dataset(Dataset):
	def __init__(self, sam_path, lab_path, flag, uf=4, transform=None):
		self.uf = uf
		self.flag = flag
		self.sam_path = sam_path
		self.lab_path = lab_path
		self.transform = transform
		self.ids = {}
		for i, file in enumerate(os.listdir(self.sam_path)):
			self.ids[i] = file[:4]

	def __len__(self):
		return len(self.ids)

	def __getitem__(self, idx):
		if self.flag == 'train':
			filename = self.ids[idx] + 'x4w' + str((idx%4)+1) + '.png'
		elif self.flag == 'test':
			filename = self.ids[idx] + 'x4w' + '.png'
		x = Image.open(os.path.join(self.sam_path, filename))
		x = normalize(x)
		a = random.choice([0,1])
		rot = RotateImage(a)
		x_trsf = transforms.CenterCrop(48)
		x = x_trsf(x)
		x = rot(x)
		x_bic = x.resize((x.size[0]*self.uf, x.size[1]*self.uf), resample=Image.BICUBIC)
		x_bic = rot(x_bic)
		# print(x)

		y = Image.open(os.path.join(self.lab_path, self.ids[idx]+'.png'))
		y = normalize(y)
		y_trsf = transforms.CenterCrop(192)
		y = y_trsf(y)
		y = rot(y)

		if self.transform:
			x = self.transform(x)
			x_bic = self.transform(x_bic)
			y = self.transform(y)

		return (x_bic, x, y)

class RotateImage:
	def __init__(self, a):
		self.a = a

	def __call__(self, x):
		if self.a == 0:
			return TF.rotate(x, 0)
		elif self.a == 1:
			return TF.rotate(x, 90)

def train(model, optimizer, criterion, train_loader, test_loader, epochs=100):
	print("Training Begins!")
	best = np.Inf
	for epoch in range(epochs):
		train_loss = 0.0
		val_loss = 0.0
		for data in (train_loader):
			x_bic, x, y = data[0].to('cuda'), data[1].to('cuda'), data[2].to('cuda')
			optimizer.zero_grad()

			output = model(x_bic, x)
			loss = criterion(output, y)
			loss.backward()
			optimizer.step()

			train_loss += loss.item()*x_bic.shape[0]


		for data in test_loader:
			x_bic, x, y = data[0].to('cuda'), data[1].to('cuda'), data[2].to('cuda')
			output = model(x_bic, x)
			loss = criterion(output, y)
			val_loss += loss.item()*x_bic.shape[0]

		train_loss = train_loss/len(train_loader.dataset)
		val_loss = val_loss/len(test_loader.dataset)

		print('Epoch: {} \tTraining Loss: {} \tValidation Loss: {}'.format(epoch, train_loss, val_loss))

		if val_loss <= best:
			print("Validation loss decreased ({} --> {}). Saving model....".format(best, val_loss))
			torch.save(model.state_dict(), 'sisr_{}.pt'.format(epoch+1))
			best = val_loss
			
	print("EOT")

transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
sam_train = "D:/Storage/ML Datasets/DIV2K/DIV2K_train_LR_wild/"
lab_train = "D:/Storage/ML Datasets/DIV2K/DIV2K_train_HR/"
# sam_path = "DIV2K_train_LR_wild/"
# lab_path = "DIV2K_train_HR/"
sam_test = "D:/Storage/ML Datasets/DIV2K/DIV2K_valid_LR_wild/"
lab_test = "D:/Storage/ML Datasets/DIV2K/DIV2K_valid_HR/"
# sam_test = "DIV2K_valid_LR_wild/"
# lab_test = "DIV2K_valid_HR/"

train_data = SR_Dataset(sam_path=sam_train, lab_path=lab_train, flag='train', transform=transform)
test_data = SR_Dataset(sam_path=sam_test, lab_path=lab_test, flag='test', transform=transform)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

model = sisr().to('cuda')
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train(model, optimizer, criterion, train_loader, test_loader)
# for i, data in enumerate(loader):
# 	x_bic, x, y = data[0], data[1], data[2]
# 	print(x_bic.shape)
# 	print(x.shape)
# 	print(y.shape)
'''
# data = np.load("C:/Users/TheNush07/Desktop/Work/Projects/XRAY- Image Super Resolution/train/samples.npz")
# x_train_bic = data['a']
# print(x_train_bic.shape)
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
sam_path = "D:/Storage/ML Datasets/DIV2K/DIV2K_train_LR_wild/"
lab_path = "D:/Storage/ML Datasets/DIV2K/DIV2K_train_HR/"
train_data = SR_Dataset(sam_path=sam_path, lab_path=lab_path, transform=transform)
loader = DataLoader(train_data, batch_size=16, shuffle=True)

for i, data in enumerate(loader):
	x = data[0]
	x_bic = []
	trsf = transforms.ToTensor()
	for i in range(x.shape[0]):
		n_arr = x[i].numpy()
		n_arr = np.transpose(n_arr)
		n_arr /= 255.0
		img = Image.fromarray(n_arr.astype('uint8'))
		img = img.resize((img.size[0]*4, img.size[1]*4), resample=Image.BICUBIC)
		n_arr = np.asarray(img)
		n_arr = np.transpose(n_arr)
		x_bic.append(n_arr)
	break
x_bic = torch.FloatTensor(x_bic)
print(x_bic)
print(data[0])
print(data[1])


'''