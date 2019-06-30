#!/usr/bin/env python3

import sys
from pathlib import Path
import json
import os
from PIL import Image

MINIMUM = 1000
OUTPUT_DIR = "dataset"

cities = sys.argv[1:]
print("Got these cities:", cities)

if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

for city in cities:
	valid_images = 0
	if not os.path.exists(OUTPUT_DIR+"/"+city):
		os.makedirs(OUTPUT_DIR+"/"+city)
	file_list = json.loads(Path(city+"/dataset_info.json").read_text())

	print(city+":", len(file_list))
	assert(len(file_list) >= MINIMUM)
	for file_name in file_list:
		if valid_images >= 1000:
			break

		file_name = file_name.replace("/", "→")
		# img = Image.open(city+"/"+file_name+".jpg")
		# exif_data = img._getexif()
		d = Path(city+"/"+file_name+".jpg").read_bytes()
		xmp_start = d.find(b'<x:xmpmeta')
		xmp_end = d.find(b'</x:xmpmeta')
		xmp_str = str(d[xmp_start:xmp_end+12])
		# print("file_name", file_name, "xmp_str", xmp_str)

		if xmp_str.find("bad") == -1:
			existing_contents = Path(city+"/"+file_name+".jpg").read_bytes()
			Path(OUTPUT_DIR+"/"+city+"/"+file_name+".jpg").write_bytes(existing_contents)
			valid_images += 1
	print("Valid images:", valid_images)
