import os

VG_PATH = "/home/csjihwanh/Desktop/Projects/GCN_Image_Captioning/datasets/vg/"

paths = {
    "image_data" :  os.path.join(VG_PATH, 'image_data.json'),
    "vg_sgg" : os.path.join(VG_PATH, 'VG-SGG-with-attri.h5'),
    "vg_sgg_original" : os.path.join(VG_PATH, 'VG-SGG.h5'),
    "vg_sgg_dicts" : os.path.join(VG_PATH, 'VG-SGG-dicts-with-attri.json'),
    "scene_graph" : os.path.join(VG_PATH, 'scene_graph.json'),
    "rel_to_img" : os.path.join(VG_PATH, 'relationship_to_image.json'),
}

def path_get(key) :
    return paths[key]
