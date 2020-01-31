import torch
from torch import nn
from torchvision import transforms
from net import *
import argparse
import cv2
from PIL import Image
import sys

def normalize(img):
	n_arr = np.asarray(img)
	# n_arr = n_arr.astype(np.float32)
	n_arr = n_arr[:,:,:3]
	# n_arr /= 255.0
	print(n_arr.max())
	print(n_arr.min())
	# img = Image.fromarray(n_arr.astype(np.uint8))
	img = Image.fromarray(n_arr)
	return img

# load model
model = sisr().to('cuda')
model.eval()
save_path = 'sisr_100.pt'
model.load_state_dict(torch.load(save_path))

# parse arguments
parser = argparse.ArgumentParser(description="img path")
parser.add_argument('img_path', type=str, help="Give image path")
args = parser.parse_args()

# load image and normalize
path = args.img_path
img = Image.open(args.img_path)
img = normalize(img)

# transforms
trsfms = transforms.Compose([transforms.ToTensor()])
							 # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# pre-process image
crop = transforms.CenterCrop(48)
img = crop(img)
img_bic = img.resize((img.size[0]*4, img.size[1]*4), resample=Image.BICUBIC)

# arr = np.asarray(img)
# cv2.imshow("arr", arr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = trsfms(img)
img_bic = trsfms(img_bic)

print(img.max())
print(img.min())
print(img_bic.max())
print(img_bic.min())

img = torch.unsqueeze(img, dim=0)
img_bic = torch.unsqueeze(img_bic, dim=0)

img = img.to('cuda')
img_bic = img_bic.to('cuda')

res = model(img_bic, img)
print(res.shape)

op = torch.squeeze(res, dim=0)
op = op.detach().cpu().numpy()
op = np.transpose(op)
print(op)
print(op.max())
print(op.min())
cv2.imshow('op', op)
cv2.waitKey(0)
cv2.destroyAllWindows()