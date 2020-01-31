import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision.transforms import Compose, CenterCrop

import numpy as np
import matplotlib.pyplot as plt
# import cv2

device = 'cuda'

class res_block(nn.Module):
	def __init__(self, in_channel):
		super(res_block, self).__init__()
		self.conv1 = nn.Conv2d(in_channel, int(2*in_channel), kernel_size=3, stride=1, padding=1, bias=False)
		self.batch_norm1 = nn.BatchNorm2d(int(2*in_channel))
		self.conv2 = nn.Conv2d(int(2*in_channel), in_channel, kernel_size=3, stride=1, padding=1, bias=False)
		self.batch_norm2 = nn.BatchNorm2d(in_channel)

	def forward(self, x):
		temp = x

		conv1 = self.conv1(temp)
		print(torch.cuda.memory_allocated())
		
		bn1 = F.relu(self.batch_norm1(conv1))
		print(torch.cuda.memory_allocated())

		conv2 = self.conv2(bn1)
		bn2 = self.batch_norm2(conv2)

		output = bn2 + x
		print(torch.cuda.memory_allocated())

		return output

class dense_res_block(nn.Module):
	def __init__(self, initial):
		super(dense_res_block, self).__init__()
		self.res_block = res_block(initial)
		self.bn = nn.BatchNorm2d(initial)
		self.bottleneck1 = nn.Conv2d(int(2*initial), initial, kernel_size=1, stride=1, padding=0, bias=False)
		self.bottleneck2 = nn.Conv2d(int(4*initial), initial, kernel_size=1, stride=1, padding=0, bias=False)
		self.bottleneck3 = nn.Conv2d(int(8*initial), initial, kernel_size=1, stride=1, padding=0, bias=False)
		self.bottleneck4 = nn.Conv2d(int(16*initial), initial, kernel_size=1, stride=1, padding=0, bias=False)
		self.ps = nn.PixelShuffle(upscale_factor=4)

	def forward(self, x):
		temp = x

		r1 = self.res_block(temp)
		r1 = torch.cat([r1, temp], axis=1)
		# print(r1.shape)
		b1 = F.relu(self.bn(self.bottleneck1(r1)))

		r2 = self.res_block(b1)
		r2 = torch.cat([r2, r1, temp], axis=1)
		b2 = F.relu(self.bn(self.bottleneck2(r2)))

		r3 = self.res_block(b2)
		r3 = torch.cat([r3, r2, r1, temp], axis=1)
		b3 = F.relu(self.bn(self.bottleneck3(r3)))

		r4 = self.res_block(b3)
		r4 = torch.cat([r4, r3, r2, r1, temp], axis=1)
		b4 = F.relu(self.bn(self.bottleneck4(r4)))
		return b4
		# return self.ps(r4)

class feat_reconstruction(nn.Module):
	def __init__(self):
		super(feat_reconstruction, self).__init__()
		self.drblock = dense_res_block(256)
		self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv2 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
		self.ps = nn.PixelShuffle(upscale_factor=4)
	def forward(self, x):
		temp = x

		c1 = self.conv1(temp)

		dr1 = self.drblock(c1)
		dr2 = self.drblock(dr1)
		dr3 = self.drblock(dr2)
		dr4 = self.drblock(dr3)
		dr5 = self.drblock(dr4)
		dr6 = self.drblock(dr5)

		dr6 += c1

		pixel_shuffle = self.ps(dr6)
		c2 = self.conv2(pixel_shuffle)
		# print(c2.shape)
		
		return c2

class dense_layer(nn.Module):
	def __init__(self, in_channel, growth_rate):
		super(dense_layer, self).__init__()
		self.bottleneck = nn.Conv2d(in_channel, int(2*in_channel), kernel_size=1, stride=1, padding=0)
		self.conv1 = nn.Conv2d(int(2*in_channel), growth_rate, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		temp = x

		b1 = self.bottleneck(temp)
		c1 = self.conv1(b1)
		c1 = torch.cat([c1, temp], axis=1)

		return c1

class dense_block(nn.Module):
	def __init__(self, in_channel, growth_rate=16):
		super(dense_block, self).__init__()
		self.layer1 = dense_layer(in_channel, growth_rate)
		self.layer2 = dense_layer(in_channel+growth_rate, growth_rate)
		self.layer3 = dense_layer(in_channel+int(2*growth_rate), growth_rate)
		self.layer4 = dense_layer(in_channel+int(3*growth_rate), growth_rate)
		self.layer5 = dense_layer(in_channel+int(4*growth_rate), growth_rate)
		self.layer6 = dense_layer(in_channel+int(5*growth_rate), growth_rate)
		self.bottleneck = nn.Conv2d(in_channel+int(6*growth_rate), int(in_channel/2) + int(3*growth_rate), kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		temp = x

		l1 = self.layer1(x)
		l2 = self.layer2(l1)
		l3 = self.layer3(l2)
		l4 = self.layer4(l3)
		l5 = self.layer5(l4)
		l6 = self.layer6(l5)

		b1 = self.bottleneck(l6)

		return b1

class attention(nn.Module):
	def __init__(self, initial=32):
		super(attention, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(16, initial, kernel_size=3, stride=1, padding=1)

		self.m_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.a_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

		self.db1 = dense_block(initial)
		self.db2 = dense_block(int(initial/2) + 48)
		self.db3 = dense_block(int(initial/4) + 72)
		# self.db4 = dense_block(int(initial/4) + 120)
		self.db4 = dense_block(int(initial/2) + 144)
		self.db5 = dense_block(initial + 96)

		self.upsam1 = nn.ConvTranspose2d(int(initial/8) + 84, int(initial/4) + 72, kernel_size=2, stride=2, padding=0)
		self.upsam2 = nn.ConvTranspose2d(int(initial/4)+120, int(initial/2)+48, kernel_size=2, stride=2, padding=0)
		self.upsam3 = nn.ConvTranspose2d(int(initial/2)+96, initial, kernel_size=2, stride=2, padding=0)

		self.conv3 = nn.Conv2d(int(2*initial), 16, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		temp = x

		c1 = self.conv1(temp)
		c2 = self.conv2(c1)

		mp = self.m_pool(c2)
		db1 = self.db1(mp)
		ap = self.a_pool(db1)
		db2 = self.db2(ap)
		ap = self.a_pool(db2)
		db3 = self.db3(ap)

		up1 = self.upsam1(db3)
		up1 = torch.cat([up1, db2], axis=1)
		db4 = self.db4(up1)
		up2 = self.upsam2(db4)
		up2 = torch.cat([up2, db1], axis=1)
		db5 = self.db5(up2)
		up3 = self.upsam3(db5)
		up3 = torch.cat([up3, c2], axis=1)

		c3 = self.conv3(up3)
		c4 = self.conv4(c3)

		return torch.sigmoid(c4)

class sisr(nn.Module):
	def __init__(self,):
		super(sisr, self).__init__()
		self.feat = feat_reconstruction()
		self.att = attention()
	def forward(self, x_bic, x):
		f_recon = self.feat(x)
		att_net = self.att(x_bic)
		output = torch.mul(f_recon, att_net)
		output += x_bic
		return output


