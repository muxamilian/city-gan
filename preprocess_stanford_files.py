#!/usr/bin/env python3
# FIXME: This file is broken. It is copying files around while actually it would be a lot faster to create hardlinks and not copies of each file.

import asyncio
import sys
import os
import re
import math
import time
from pathlib import Path
# import shutil
import aiofiles
import aiofiles.os

dataset_regex = re.compile("^\d+$")

ALLOWED_ANGLE = (0, 40)
path = sys.argv[1]
directory_at_which_to_start = None
if len(sys.argv) > 2:
	directory_at_which_to_start = sys.argv[2]

amsterdam = (52.366667, 4.9)
paris = (48.8567, 2.3508)
san_francisco = (37.783333, -122.416667)
chicago = (41.836944, -87.684722)
new_york = (40.7127, -74.0059)
florence = (43.783333, 11.25)
dc = (38.904722, -77.016389)
las_vegas = (36.175, -115.136389)

def minimum_distance(l):
	minimum = float("inf")
	for i in range(len(l)):
		for j in range(i+1, len(l)):
			minimum = min(minimum, math.sqrt( (l[i][0]-l[j][0])**2 + (l[i][1]-l[j][1])**2 ))
	return minimum

city_coordinates = (amsterdam, paris, san_francisco, chicago, new_york, florence, dc, las_vegas)

inclusion_radius = minimum_distance(city_coordinates)/2

print("inclusion_radius", inclusion_radius)

def get_closest(cities, latlng):
	minimum = float("inf")
	index = 0
	for i, item in enumerate(cities):
		new_min = math.sqrt( (item[0]-latlng[0])**2 + (item[1]-latlng[1])**2 )
		if new_min <= minimum:
			minimum = new_min
			index = i
	assert(minimum < inclusion_radius)
	return index

for i in range(len(city_coordinates)):
	city_name = "city_{}".format(i)
	if not os.path.exists(path+"/"+city_name):
		os.makedirs(path+"/"+city_name)

extracted_folders = sorted([f for f in os.listdir(path) if dataset_regex.match(f)])
index = 0
if directory_at_which_to_start is not None:
	index = extracted_folders.index(directory_at_which_to_start)

async def coroutine(actual_file):
	try:
		async with aiofiles.open(actual_file, "r") as content_f:
			content = await content_f.read()
	except UnicodeDecodeError:
		return 0
	content = content.split()

	pitch = float(content[16])
	if pitch < ALLOWED_ANGLE[0] or pitch > ALLOWED_ANGLE[1]:
		return 0
	roll = float(content[17])
	latlng = (float(content[11]), float(content[12]))
	closest_city = get_closest(city_coordinates, latlng)
	assert (roll == 0)

	actual_file_str = str(actual_file)
	last_part = actual_file_str.split("/")[-1]
	closest_city = "city_{}".format(closest_city)

	async with aiofiles.open(actual_file_str, "rb") as src_meta_file:
		statinfo = await aiofiles.os.stat(actual_file_str)
		async with aiofiles.open(path + "/" + closest_city + "/" + last_part, "wb") as dst_meta_file:
			copied_bytes = await aiofiles.os.sendfile(dst_meta_file.fileno(), src_meta_file.fileno(), None, statinfo.st_size)
			assert(copied_bytes == statinfo.st_size)
	try:
		async with aiofiles.open(actual_file_str[:-4] + ".jpg", "rb") as src_jpg_file:
			statinfo2 = await aiofiles.os.stat(actual_file_str[:-4] + ".jpg")
			async with aiofiles.open(path + "/" + closest_city + "/" + last_part[:-4] + ".jpg", "wb") as dst_jpg_file:
				copied_bytes2 = await aiofiles.os.sendfile(dst_jpg_file.fileno(), src_jpg_file.fileno(), None, statinfo2.st_size)
				assert(copied_bytes2 == statinfo2.st_size)

		return 1

	except FileNotFoundError as e:
		return 0

LIMIT = 500

if len(sys.argv) > 3:
	extracted_folders = sys.argv[2:]
	print("Using list of tar files")

sys.exit()

for i in range(index, len(extracted_folders)):
	f = extracted_folders[i]
	print("Entering folder", f)
	loop = asyncio.new_event_loop()
	filenames = []
	start_time = time.time()

	for actual_file in Path(path+"/"+f, mode='r').iterdir():
		actual_file_str = str(actual_file)
		if actual_file_str[-4:] != ".txt":
			continue
		filenames.append(actual_file)

	nr_extracted = 0
	number_of_iterations = int(math.ceil(len(filenames)/LIMIT))
	for iteration in range(number_of_iterations):
		current_filenames = filenames[iteration * LIMIT:(iteration + 1) * LIMIT]
		tasks = []
		for actual_file in current_filenames:
			tasks.append(loop.create_task(coroutine(actual_file)))
		loop.run_until_complete(asyncio.wait(tasks))
		nr_extracted += sum([x.result() for x in tasks])
		print(
			f'Extracted {nr_extracted} files from the {extracted_folders[i]}-th folder. '
			f'Took {time.time() - start_time}s.')
