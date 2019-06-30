#!/usr/bin/env python3

import sys
import json
from pathlib import Path
import numpy as np
import urllib.request
import urllib.parse
import os
import random
from tqdm import tqdm
import subprocess
from functools import reduce

SEED = 1
LIMIT = 1500

file_name = sys.argv[1]
city_name = file_name.split(".")[0]

if not os.path.exists(city_name):
	os.makedirs(city_name)

key = Path("api-key.txt").read_text()

UPWARDS_ANGLE = 20

request_string = "https://maps.googleapis.com/maps/api/streetview?size=640x640&location={}&pitch="+str(UPWARDS_ANGLE)+"&key={}&source=outdoor"
metadata_request_string = "https://maps.googleapis.com/maps/api/streetview/metadata?location={}&key={}&source=outdoor"

contents = Path(file_name).read_text()
parsed = json.loads(contents)
features = parsed["features"]
addresses = []
for feature in features:
	node = feature["properties"]
	if "addr:postcode" in node and "addr:street" in node and "addr:housenumber" in node:
		addresses.append({"postcode": node["addr:postcode"], "street": node["addr:street"], "housenumber": node["addr:housenumber"]})

def get_full_address(address):
	return "{} {}, {}".format(address["housenumber"], address["street"], address["postcode"])

addresses = list({*list(map(lambda x: get_full_address(x), addresses))})

print("Got {} addresses".format(len(addresses)))
addresses = sorted(addresses)
assert(len(addresses)>=LIMIT)

random.Random(SEED).shuffle(addresses)

index = 0
num_requests = 0

done_addresses = set()
done_addresses_len = len(done_addresses)
addresses_iterator = iter(addresses)

all_files = os.listdir(city_name)

def matches_any_jpg(sample, list_of_data):
	ret = list(filter(lambda x: x[1], zip(list_of_data, map(lambda x: sample in x and ".jpg" in x, list_of_data))))
	return ret[0][0] if ret else None

iterations = 0
with tqdm(total=LIMIT) as pbar:
	while len(done_addresses) < LIMIT:
		iterations += 1
		address = next(addresses_iterator)
		full_address_string = address
		match = matches_any_jpg(full_address_string.replace("/", "→"), all_files)

		if not match:
			request = metadata_request_string.format(urllib.parse.quote(full_address_string.replace(" ", "+")), key)
			response = urllib.request.urlopen(request)
			data = response.read().decode("utf-8")
			Path(city_name+"/"+full_address_string.replace("/", "→")+".json").write_text(data)
			json_metadata = json.loads(data)

			if not "copyright" in json_metadata or json_metadata["copyright"] != "© Google" or json_metadata["status"] != "OK":
				continue

			request = request_string.format(urllib.parse.quote(full_address_string.replace(" ", "+")), key)
			response = urllib.request.urlopen(request)
			num_requests += 1
			data = response.read()
			Path(city_name+"/"+full_address_string.replace("/", "→")+".jpg").write_bytes(data)
		else:
			full_address_string = match[:-4]

		done_addresses.add(full_address_string)
		new_len = len(done_addresses)
		if new_len > done_addresses_len:
			assert(new_len-1 == done_addresses_len)
			pbar.update(1)
		done_addresses_len = new_len

jsonified_dataset = json.dumps(list(done_addresses))
Path(city_name+"/dataset_info.json").write_text(jsonified_dataset)
print("Made {} requests".format(num_requests))
