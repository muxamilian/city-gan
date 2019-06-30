#!/usr/bin/env python3

import argparse
import sys
print("sys.version", sys.version)
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import math
import neural_network
from inspect import signature
import numpy as np
from visualizing_stuff import visualize_stuff
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=150, help='the height / width of the input image to network')
parser.add_argument('--hack', action='store_true', help='if `hack\' is on, the city labels are added as an extra channel to the input of the discriminator. Otherwise they are added in the end as the input to a linear layer')
parser.add_argument('--classifyGenerated', action='store_true', help='Classify generated images and not real ones.')
parser.add_argument('--nonConditional', action='store_true', help='makes the GAN non-conditional')
parser.add_argument('--cropSize', type=int, default=128, help='the height / width of the image after cropping')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='output.png', help='output file when using an analysis technique')
parser.add_argument('--function', default='train', help='the function that is going to be called')
parser.add_argument('--arg', default='None', help='arguments that "function" is going to be called with')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--numSamples', type=int, default=8, help='how many random samples are going to be used for visualization')
parser.add_argument('--numClasses', type=int, default=None, help='number of classes in the dataset; only necessary when using a function other than train')
parser.add_argument('--maxSamples', type=int, default=sys.maxsize, help='maximum number of samples')

opt = parser.parse_args()
print(opt)

FAKE_IMAGE_INTERVAL = 10000
DUMP_PARAMETER_INTERVAL = 100000

if opt.manualSeed is None:
	opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

nc=3
if opt.function == "train" or opt.function == "classify":
	if opt.dataset == "gan-city":
		# folder dataset
		dataset = dset.ImageFolder(root=opt.dataroot,
									transform=transforms.Compose([
										transforms.Resize(opt.imageSize),
										transforms.RandomHorizontalFlip(p=0.5),
										transforms.RandomCrop(opt.cropSize),
										transforms.ToTensor(),
										transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
									]))
		nc=3
	assert dataset

	classes = dataset.classes

if opt.nonConditional:
	num_classes = 0

if opt.function == "train" or opt.function == "classify":
	num_classes = len(classes)
	print("Classes:", dataset.class_to_idx)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
											shuffle=True, num_workers=int(opt.workers), pin_memory=True)
else:
	assert(opt.numClasses is not None)
	num_classes = opt.numClasses
print("Hack:", opt.hack)
print("Non-conditional:", opt.nonConditional)

assert(not opt.nonConditional or opt.hack)

device = torch.device("cuda:0" if opt.cuda else "cpu")
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

netG, netD = neural_network.get_generator_and_discriminator(nc, nz, ngf, ndf, device, opt.hack, opt.cropSize, num_classes)

if opt.netG != '':
	netG.load_state_dict(torch.load(opt.netG, map_location=device))
if opt.netD != '':
	netD.load_state_dict(torch.load(opt.netD, map_location=device))

def combine_fixed_noise_with_classes(num_samples, labels_one_hot):
	size = labels_one_hot.size()
	fixed_noise = torch.randn(num_samples, nz, 1, 1, device=device).repeat(size[0], 1, 1, 1)
	labels_one_hot = labels_one_hot.repeat(1,num_samples).view(size[0]*num_samples, size[1])
	return torch.cat([fixed_noise, labels_one_hot.unsqueeze(2).unsqueeze(3)], dim=1)

def get_one_hot_vector(class_indices, num_classes, batch_size):
	y_onehot = torch.FloatTensor(batch_size, num_classes)
	y_onehot.zero_()
	return y_onehot.scatter_(1, class_indices.unsqueeze(1), 1)

def train():
	writer = SummaryWriter()
	criterion = nn.BCELoss()

	if not opt.nonConditional:
		fixed_noise = get_fixed_noise()
	else:
		fixed_noise = torch.randn(opt.numSamples, nz, 1, 1, device=device)

	real_label = 1
	fake_label = 0

	# setup optimizer
	optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

	# processed_samples = 0
	if opt.netG != '' or opt.netD != '':
		processed_samples = int(opt.netG.split("_")[-1].split(".")[0])
	else:
		processed_samples = 0
	while True:
		for data in dataloader:

			############################
			# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
			###########################
			# train with real
			netD.zero_grad()
			real_cpu = data[0].to(device, non_blocking=True)
			batch_size = real_cpu.size(0)

			if not opt.nonConditional:
				class_one_hot = get_one_hot_vector(data[1], num_classes, batch_size).unsqueeze(2).unsqueeze(3).to(device, non_blocking=True)

			label = torch.full((batch_size,), real_label, device=device)

			netD.train()
			if not opt.nonConditional:
				output = netD(real_cpu, class_one_hot)
			else:
				output = netD(real_cpu)
			errD_real = criterion(output, label)
			errD_real.backward()
			D_x = output.mean().item()

			# train with fake
			noise = torch.randn(batch_size, nz, 1, 1, device=device)

			netG.train()
			if not opt.nonConditional:
				fake = netG(torch.cat([noise, class_one_hot], dim=1))
			else:
				fake = netG(noise)
			label.fill_(fake_label)
			if not opt.nonConditional:
				output = netD(fake.detach(), class_one_hot)
			else:
				output = netD(fake.detach())
			errD_fake = criterion(output, label)
			errD_fake.backward()
			D_G_z1 = output.mean().item()
			errD = errD_real + errD_fake
			optimizerD.step()

			############################
			# (2) Update G network: maximize log(D(G(z)))
			############################
			netG.zero_grad()
			label.fill_(real_label)  # fake labels are real for generator cost
			netD.eval()
			if not opt.nonConditional:
				output = netD(fake, class_one_hot)
			else:
				output = netD(fake)
			errG = criterion(output, label)
			errG.backward()
			D_G_z2 = output.mean().item()
			optimizerG.step()

			processed_samples += batch_size
			writer.add_scalar("loss_D", errD.item(), processed_samples)
			writer.add_scalar("loss_G", errG.item(), processed_samples)

			if math.floor((processed_samples-batch_size)/FAKE_IMAGE_INTERVAL) < math.floor(processed_samples/FAKE_IMAGE_INTERVAL):
				netG.eval()
				fake = netG(fixed_noise)
				image_for_saving = vutils.make_grid(fake, normalize=True, scale_each=True, nrow=opt.numSamples)
				writer.add_image('fake_image', image_for_saving, processed_samples)

			# do checkpointing
			if math.floor((processed_samples-batch_size)/DUMP_PARAMETER_INTERVAL) < math.floor(processed_samples/DUMP_PARAMETER_INTERVAL):
				torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (writer.log_dir, processed_samples))
				torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (writer.log_dir, processed_samples))

def get_fixed_noise():
	avg_and_classes_vec = get_classes_and_avg_vec()
	fixed_noise = combine_fixed_noise_with_classes(opt.numSamples, avg_and_classes_vec)
	return fixed_noise

def get_classes_and_avg_vec():
	avg_vec = torch.tensor([1/num_classes], device=device).repeat(1, num_classes)
	classes_vec = torch.zeros(1, device=device).repeat(num_classes, num_classes)
	for i in range(num_classes):
		classes_vec[i, i] = 1
	avg_and_classes_vec = torch.cat([avg_vec, classes_vec], dim=0)
	return avg_and_classes_vec

def make_grid(a, b, steps):
	a, b = np.array(a), np.array(b)
	if steps > 1:
		difference = (b-a)/(steps-1)
		full_matrix = np.tile(a[np.newaxis,:], (steps, 1)) + np.tile(difference[np.newaxis,:], (steps, 1))*np.tile(np.arange(steps)[:, np.newaxis], (1, a.shape[0]))
	else:
		full_matrix = np.tile(a[np.newaxis,:], (steps, 1))
	return full_matrix

def get_samples():
	netG.eval()
	fixed_noise = get_fixed_noise()

	samples = netG(fixed_noise)
	print("samples.shape", samples.shape)
	path = opt.netG.replace("/", "_")
	number_of_rows_to_plot = opt.numSamples
	vutils.save_image(samples, f"generated_samples/samples_{path}_{opt.numSamples}_{opt.manualSeed}.png",normalize=True, nrow=number_of_rows_to_plot)

def plot_along_grid(a, b, steps):
	netG.eval()
	grid = torch.FloatTensor(make_grid(a, b, steps)).to(device)
	input_for_generator = combine_fixed_noise_with_classes(opt.numSamples, grid)

	samples = netG(input_for_generator)
	path = opt.netG.replace("/", "_")
	number_of_rows_to_plot = opt.numSamples if opt.numSamples > 1 else num_classes
	vutils.save_image(samples, f"generated_samples/samples_{path}_{a}_{b}_{opt.numSamples}_{opt.manualSeed}.png",normalize=True, nrow=number_of_rows_to_plot)

def print_net():
	print(netG)
	# print(list(netG.children()))
	print(netD)
	# print(list(netD.children()))

def visualize():
	visualize_stuff(netD, get_classes_and_avg_vec())

def argmax(vals):
    return max(
        range(len(vals)),
        key = lambda ii: vals[ii],
    )
def argmin(vals):
    return min(
        range(len(vals)),
        key = lambda ii: vals[ii],
    )


def classify():
	global dataloader
	global dataset
	from statistics import mean, stdev

	if opt.nonConditional:
		raise ValueError("Cannot classify if the discriminator is not conditional.")

	# directory = "real_samples"
	# if not os.path.exists(directory):
	# 		os.mkdir(directory)

	global_outputs = [list() for _ in range(num_classes)]
	global_labels = []

	# distribution = [0.5319990576390119, 0.46236549538647526, 0.5464009061221632, 0.5255237457947151]
	# distribution = [1,1,1,1]

	if opt.classifyGenerated:
		netG.eval()
		generated_data = []
		for i in range(num_classes):
			for j in range(min(int(len(dataset)*opt.niter/num_classes), int(opt.maxSamples/num_classes))):
				noise = torch.randn(1, nz, 1, 1)
				class_one_hot = get_one_hot_vector(torch.LongTensor([i]), num_classes, 1).unsqueeze(2).unsqueeze(3)
				# class_one_hot = torch.FloatTensor(num_classes)
				# class_one_hot.zero_()
				# class_one_hot[i] = 1
				# class_one_hot = class_one_hot.unsqueeze(0).unsqueeze(2).unsqueeze(3)
				fake = netG(torch.cat([noise, class_one_hot], dim=1).to(device)).squeeze(0).cpu().detach()
				generated_data.append((fake, i))

		class FakeLoader(torch.utils.data.Dataset):
			def __init__(self, data):
				self.data = data

			def __len__(self):
				return len(self.data)

			def __getitem__(self, index):
				return self.data[index]

		dataset = FakeLoader(generated_data)
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
										shuffle=True, num_workers=int(opt.workers), pin_memory=True)
	else:
		dataset = torch.utils.data.ConcatDataset([dataset for _ in range(opt.niter)])

	print("dataset length:", len(dataset))

	samples = 0
	for index, data in enumerate(dataloader):
		print("index", index)

		netD.eval()
		real_cpu = data[0].to(device)
		batch_size = real_cpu.size(0)

		# print("index", index)
		# if index==0:
		# 	vutils.save_image(real_cpu, f"{directory}/{index}_{opt.manualSeed}.png",normalize=True, nrow=int(math.sqrt(batch_size)))
		# 	print(data[1].tolist())
		# 	fake_cpu = torch.FloatTensor(np.float32(np.random.normal(0.0, 1.0, (batch_size, 3, opt.cropSize, opt.cropSize)))).to(device, non_blocking=True)

		outputs = []
		for ith_class in range(num_classes):
			class_one_hot = get_one_hot_vector(torch.LongTensor([ith_class]*batch_size), num_classes, batch_size).unsqueeze(2).unsqueeze(3).to(device)
			# print("class_one_hot", class_one_hot)
			output = netD(real_cpu, class_one_hot)
			# print("output", output)
			outputs.append(output.detach().tolist())

		global_outputs = [global_item+item for global_item, item in zip(global_outputs, outputs)]
		global_labels = global_labels + data[1].tolist()

		samples += real_cpu.shape[0]
		if samples >= opt.maxSamples:
			break
		# classification_scores = list(zip(range(1,batch_size+1), list(zip(*outputs)), [argmax(item) for item in zip(*outputs)], data[1].tolist()))
		# print("classification_scores", "\n".join(map(str,classification_scores)))
		# break
	# print("global_outputs", global_outputs, "global_labels", global_labels)
	# global_outputs = [[subitem/divisor for subitem in item] for item, divisor in zip(global_outputs, distribution)]
	# print("global_outputs", global_outputs)
	# print([mean(item) for item in global_outputs], mean([a==b for a, b in zip([argmax(item) for item in zip(*global_outputs)], global_labels)]))
	# print("global_outputs", global_outputs)
	# print(mean([a==b for a, b in zip([argmax(item) for item in zip(*global_outputs)], global_labels)]))
	correct_outputs = [(label, item[label]) for item, label in zip(zip(*global_outputs), global_labels)]
	outputs_per_class = [list() for _ in range(num_classes)]
	for i, score in correct_outputs:
		outputs_per_class[i].append(score)
	print("means", [mean(item) for item in outputs_per_class])
	print("stds", [stdev(item) for item in outputs_per_class])

function = globals()[opt.function]
number_of_arguments = len(signature(function).parameters)
if number_of_arguments == 0:
	function()
else:
	function(*eval(opt.arg))

