import os
import json
import shutil

# load json and save directory for labels train/val/test
coco_file = 'ILSVRC_FGFA_COCO.json'
save_folder = '../../../ILSVRC2015_yolo/labels/train'



#source of all the images and destination folder for train/test/val
source_path = "../../.."
destination_path = "../../../ILSVRC2015_yolo/images/train"


# Use os.listdir() to get a list of filenames in the folder
file_names = os.listdir(source_path)

with open(coco_file) as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations'] 
categories = {cat['id']: cat['name'] for cat in coco['categories']}

os.makedirs(save_folder, exist_ok=True)
os.makedirs(destination_path, exist_ok=True)

for ann in annotations:
    image = next(img for img in images if (img['id'] == ann['image_id']))
    # if (image["file_name"] not in file_names):
    #     continue
    #print(f"image in annotations =   {type(image['id'])}")
    width, height = image['width'], image['height']
    x_center = (ann['bbox'][0] + ann['bbox'][2] / 2) / width
    y_center = (ann['bbox'][1] + ann['bbox'][3] / 2) / height
    bbox_width = ann['bbox'][2] / width
    bbox_height = ann['bbox'][3] / height
    category_id = ann['category_id']
    image_id = ann['image_id']

    if image['file_name'].startswith("/media/tuf/ssd/"):
        image['file_name'] = image['file_name'][15:] #TODO: edit
    filename = image['file_name']
    filename = '_'.join(os.path.normpath(filename).split(os.sep))
    label_path = os.path.join(save_folder, f'{filename[:-5]}.txt')
    with open(label_path, 'a') as f:
        line = f'{category_id} {x_center} {y_center} {bbox_width} {bbox_height}\n'
        f.write(line)
          
    image_source = source_path + f'/{image["file_name"]}'
    
    shutil.copy2(image_source, os.path.join(destination_path, filename))