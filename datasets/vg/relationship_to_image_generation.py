### relationship json data generator

import json
import h5py

from config.path_catalog import path_get

image_data = json.load(open(path_get('image_data')))
vg_sgg = h5py.File(path_get('vg_sgg'))
vg_sgg_dicts = json.load(open(path_get('vg_sgg_dicts')))

corrupted_ims = [1592, 1722, 4616, 4617]

# todo : make (relationiship-image index) pair dictionary and save it into json file format

print("Loading relationship data..")
rel_num = vg_sgg['relationships'].shape[0] 
img_num = len(image_data) - len(corrupted_ims)
rel_to_img = {}

for img_idx in range(img_num) : 
    rth_s = vg_sgg['img_to_first_rel'][img_idx]
    rth_e = vg_sgg['img_to_last_rel'][img_idx]
    if(rth_s == -1) : # when there's no relationship
        continue
    for rth in range(rth_s, rth_e+1) :
        if rth % 100000 == 0 :
            print(f"{rth} relationships are loaded.")
        rel_to_img[rth] = img_idx
print("Loading relationship data done.")

# save that file into json format
with open('relationship_to_image.json', 'w') as json_file:
    json.dump(rel_to_img, json_file)