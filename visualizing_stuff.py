import torch
import torch.autograd
import torch.nn as nn
import os
import numpy as np
# import matplotlib.pyplot as plt
# from torchviz import make_dot
import torchvision.utils as vutils

DIRECTORY = "filters"

class SaveFeatures():
	def __init__(self, module, device=torch.device("cuda:0")):
		self.hook = module.register_forward_hook(self.hook_fn)
		self.device = device
	def hook_fn(self, module, input, output):
		self.features = output.requires_grad_(True)
	def close(self):
		self.hook.remove()

def repeat(tensor, dims):
	if len(dims) != len(tensor.shape):
		raise ValueError("The length of the second argument must equal the number of dimensions of the first.")
	for index, dim in enumerate(dims):
		repetition_vector = [1]*(len(dims)+1)
		repetition_vector[index+1] = dim
		new_tensor_shape = list(tensor.shape)
		new_tensor_shape[index] *= dim
		tensor = tensor.unsqueeze(index+1).repeat(repetition_vector).reshape(new_tensor_shape)
	return tensor

class FilterVisualizer():

	def get_layers_by_type(self, name):
		name = name.__name__
		# children_list = list([item for item in self.model.children()])
		children = next(self.model.children())
		# [next(self.model.children())[i] for i in range(len(next(self.model.children())))

		good_children = []
		for index, child in enumerate(children):
			# print("index", index, "child", child.__class__.__name__, name)
			if child.__class__.__name__ == name:
				good_children.append((index, child.out_channels))

		return good_children

	def __init__(self, model, size, upscaling_steps, upscaling_factor, additional_vector=None, device=torch.device("cuda:0")):
		self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
		self.model = model.eval()
		self.additional_vector = additional_vector
		self.device = device

		# set_trainable(self.model, False)

	def visualize(self, layer, filter, lr=0.1, opt_steps=1000, blur=None, random_img=None):
		sz = self.size
		# img = random_img if random_img is not None else np.float32(np.random.normal(0.0, 1.0, (1, 3, sz, sz)))  # generate random image
		img = random_img
		city = self.additional_vector[None,:,None,None].repeat(1, 1, sz , sz)

		chosen = next(self.model.children())[layer]
		# print("chosen", chosen)
		activations = SaveFeatures(chosen)  # register hook

		# for _ in range(self.upscaling_steps):  # scale the image up upscaling_steps times
		# 	img_var = torch.FloatTensor(img, requires_grad=True).to(device)  # convert image to Variable that requires grad
		# 	city_var = torch.FloatTensor(img, requires_grad=False).to(device)
		# 	composed_var = torch.cat([img_var, city_var], axis=2)
		# 	optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)
		# 	for n in range(opt_steps):  # optimize pixel values for opt_steps times
		# 		optimizer.zero_grad()
		# 		self.model(composed_var)
		# 		loss = -activations.features[0, filter].mean()
		# 		loss.backward()
		# 		optimizer.step()
		# 	img = img_var.cpu().detach().numpy().transpose(1,2,0)
		# 	self.output = img
		# 	sz = int(self.upscaling_factor * sz)  # calculate new image size
		# 	img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_LINEAR)  # scale image up
		# 	city = np.tile(self.additional_vector[None,None,:], (sz, sz, 1))
		# 	# if blur is not None: img = cv2.blur(img,(blur,blur))  # blur image to reduce high frequency patterns

		img_var = torch.FloatTensor(img).to(self.device)#.requires_grad_()
		img_var.requires_grad_()
		city_var = city.clone().detach().to(self.device)#.requires_grad_()
		city_var.requires_grad_(False)

		composed_var = torch.cat([img_var, city_var], dim=1)
		# # print("composed_var", composed_var)

		# print("composed_var.shape", composed_var.shape)
		# print("composed_var.is_leaf", composed_var.is_leaf, "composed_var.requires_grad", composed_var.requires_grad)
		# print("img_var.is_leaf", img_var.is_leaf, "img_var.requires_grad", img_var.requires_grad)
		# print("composed_var.is_leaf", composed_var.is_leaf)
		# print("Yeah")
		optimizer = torch.optim.SGD([img_var], lr=lr, weight_decay=1e-6)
		for n in range(opt_steps):  # optimize pixel values for opt_steps times
			# composed_var = torch.cat([img_var, city_var], dim=1)
			# print("img_var", img_var, "city_var", city_var, "composed_var", composed_var)
			optimizer.zero_grad()
			self.model(composed_var)
			# print("activations.features", activations.features)
			loss = -activations.features[0, filter].mean()
			# g1 = make_dot(loss)
			# g1.render(filename='g1.dot')
			# print("loss", loss)
			loss.backward()
			optimizer.step()
			# quit()
			# print("img_var.grad", img_var.grad, "city_var.grad", city_var.grad)
		# print("img_var", img_var.squeeze().shape, "city_var", city_var.squeeze().shape, "composed_var", composed_var.squeeze().shape)
		# print("img_var", img_var.squeeze().permute(1,2,0), "city_var", city_var.squeeze().permute(1,2,0), "composed_var", composed_var.squeeze().permute(1,2,0))
		self.output = img_var.cpu().detach()
		# if blur is not None: img = cv2.blur(img,(blur,blur))  # blur image to reduce high frequency patterns

		assert(sz == 128)
		self.save(layer, filter)
		activations.close()

	def save(self, layer, filter):
		# print("yeah, saving")
		vutils.save_image(self.output.squeeze(), os.path.join(DIRECTORY, "layer_"+str(layer)+"_filter_"+str(filter)+"_city_"+str(self.additional_vector.tolist())+".jpg"))
		# plt.imsave(os.path.join(DIRECTORY, "layer_"+str(layer)+"_filter_"+str(filter)+"_city_"+str(self.additional_vector.tolist())+".jpg"), np.clip(self.output.squeeze().transpose(1,2,0), 0, 1))

# (0): Conv2d(7, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
# (1): LeakyReLU(negative_slope=0.2, inplace)
# (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
# (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (4): LeakyReLU(negative_slope=0.2, inplace)
# (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
# (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (7): LeakyReLU(negative_slope=0.2, inplace)
# (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
# (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (10): LeakyReLU(negative_slope=0.2, inplace)
# (11): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
# (12): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (13): LeakyReLU(negative_slope=0.2, inplace)
# (14): Conv2d(1024, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
# (15): Sigmoid()


# layer = 40
# filter = 265

# img = PIL.Image.open("layer_"+str(layer)+"_filter_"+str(filter)+".jpg")
# plt.figure(figsize=(7,7))
# plt.imshow(img)

def visualize_stuff(model, additional_vectors):
	# size = 2
	# upscaling_factor = 2
	# upscaling_steps = int(math.log2(image_size)-math.log2(size))+1)

	if not os.path.exists(DIRECTORY):
			os.mkdir(DIRECTORY)

	size = 128
	upscaling_factor = 0
	upscaling_steps = 0

	# for vector in np.split(additional_vectors, additional_vectors.shape[0]):
	random_img_array = {}
	for vector_index in range(additional_vectors.shape[0]):
	# for vector_index in range(1):
		vector = additional_vectors[vector_index,:]
		print("vector", vector)
		# vector = vector.tolist()
		FV = FilterVisualizer(model, size, upscaling_steps, upscaling_factor, additional_vector=vector)
		good_layers = FV.get_layers_by_type(nn.Conv2d)

		print("good_layers", good_layers)
		# os.remove("g1.dot")
		# os.remove("g1.dot.pdf")

		for layer, n_filters in good_layers:
			# for f in range(min(10, n_filters)):
			for f in range(1):
				index_tuple = (layer, f)
				if index_tuple not in random_img_array:
					random_img_array[index_tuple] = np.float32(np.random.normal(0.0, 1.0, (1, 3, size, size)))
				FV.visualize(layer, f, random_img=random_img_array[index_tuple])
