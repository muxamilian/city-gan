#!/usr/bin/env python3

import sys
import os
from pathlib import Path
from tqdm import tqdm

NUMBER_OF_FILES = 500_000

base_dir = sys.argv[1]

# There are 8 cities in the dataset
for i in range(8):
	city = f"city_{i}"
	city_limited = f"city_{i}_{NUMBER_OF_FILES}"
	dir = base_dir+"/"+city

	print(f"Looking at {city}")
	number_of_valid_files = 0
	inspected_files = []
	for actual_file in Path(dir, mode='r').iterdir():
		actual_file_str = str(actual_file)
		if actual_file_str[-4:] == ".jpg":
			inspected_files.append(actual_file_str)
			number_of_valid_files += 1

	if number_of_valid_files >= NUMBER_OF_FILES:
		print(f"Hard linking files from {city}")
		if not os.path.exists(base_dir+"/"+city_limited):
			os.makedirs(base_dir+"/"+city_limited)
		with tqdm(total=NUMBER_OF_FILES) as pbar:
			for index, actual_file in enumerate(inspected_files):
				if index >= NUMBER_OF_FILES:
					break
				actual_file_str = actual_file
				if actual_file_str[-4:] == ".jpg":
					actual_file_list = actual_file_str.split("/")
					os.link(actual_file_str, "/".join(actual_file_list[:-2])+"/"+city_limited+"/"+actual_file_list[-1])
				pbar.update(1)
	else:
		print(f"Not hard linking files from {city}")

	print(f"Done with {city}")


