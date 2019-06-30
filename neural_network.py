import torch
import torch.nn as nn
import math

nc, nz, ngf, ndf, device, hack, crop_size, num_classes = None, None, None, None, None, None, None, None

def get_generator_and_discriminator(nc_arg, nz_arg, ngf_arg, ndf_arg, device_arg, hack_arg, crop_size_arg, num_classes_arg):
	global nc
	global nz
	global ngf
	global ndf
	global device
	global hack
	global crop_size
	global num_classes
	nc, nz, ngf, ndf, device, hack, crop_size, num_classes = nc_arg, nz_arg, ngf_arg, ndf_arg, device_arg, hack_arg, crop_size_arg, num_classes_arg

	netG = Generator().to(device)
	netD = Discriminator().to(device)
	netG.apply(weights_init)
	netD.apply(weights_init)
	return netG, netD

# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		m.weight.data.normal_(1.0, 0.02)

def generate_generator(start, end):
	layers = []
	layers_necessary = int(round(math.log2(end) - math.log2(start)))
	for i in range(layers_necessary):
		multiplier = int(round(math.pow(2, layers_necessary-i-2)))
		if i == layers_necessary-1:
			layers += [
				nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
				nn.Tanh()
			]
		elif i == 0:
			layers += [
				nn.ConvTranspose2d(nz+num_classes, ngf * multiplier, 4, 1, 0, bias=False),
				nn.BatchNorm2d(ngf * multiplier),
				nn.ReLU(True)
			]
		else:
			layers += [
				nn.ConvTranspose2d(ngf * multiplier * 2, ngf * multiplier, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ngf * multiplier),
				nn.ReLU(True),
			]
	return layers

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.layers = generate_generator(2, crop_size)
		self.main = nn.Sequential(*self.layers)

	def forward(self, input):
		output = self.main(input)
		return output

class Squeeze(nn.Module):
    def forward(self, input):
        return input.squeeze()

def generate_discriminator(start, end):
	layers = []
	layers_necessary = int(round(math.log2(start) - math.log2(end)))
	final_layers = []
	for i in range(layers_necessary):
		multiplier = int(round(math.pow(2, i-1)))
		if i == layers_necessary-1:
			if hack:
				layers += [
					nn.Conv2d(multiplier * ndf, 1, 4, 1, 0, bias=False),
					nn.Sigmoid()
				]
			else:
				layers += [
					nn.Conv2d(multiplier * ndf, multiplier * ndf, 4, 1, 0, bias=False),
					nn.BatchNorm2d(multiplier * ndf),
					nn.LeakyReLU(0.2, inplace=True),
					Squeeze()]
				final_layers += [
					torch.nn.Linear(multiplier*ndf+num_classes, nz, bias=False),
					nn.BatchNorm1d(nz),
					nn.LeakyReLU(0.2, inplace=True),
					torch.nn.Linear(nz, 1, bias=False),
					nn.Sigmoid()
				]
		elif i==0:
			if hack:
				layers += [
					nn.Conv2d(nc + num_classes, ndf, 4, 2, 1, bias=False),
					nn.LeakyReLU(0.2, inplace=True)
				]
			else:
				layers += [
					nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
					nn.LeakyReLU(0.2, inplace=True)
				]
		else:
			layers += [
				nn.Conv2d(multiplier * ndf, 2 * multiplier * ndf, 4, 2, 1, bias=False),
				nn.BatchNorm2d(2 * multiplier * ndf),
				nn.LeakyReLU(0.2, inplace=True)
			]
	return layers, final_layers


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		layers, final_layers = generate_discriminator(crop_size, 2)
		self.main = nn.Sequential(*layers)
		self.end = nn.Sequential(*final_layers)

	def forward(self, input, labels=None):
		if labels is None:
			return self.forward_non_conditional(input)
		else:
			return self.forward_conditional(input, labels)

	def forward_conditional(self, input, labels):
		if hack:
			class_one_hot_repeated = labels.repeat(1, 1, crop_size, crop_size)
			output = self.main(torch.cat([input, class_one_hot_repeated], dim=1))
		else:
			output = self.main(input)
			output = self.end(torch.cat([output, labels.squeeze()], dim=1))
		return output.view(-1, 1).squeeze(1)

	def forward_non_conditional(self, input):
		output = self.main(input)
		return output.view(-1, 1).squeeze(1)

