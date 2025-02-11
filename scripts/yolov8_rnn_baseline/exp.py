import json
from tqdm import tqdm 
import shutil

# with open('vid_train_coco.json', 'r') as file:
#     data = json.load(file)
#     print(len(data['images']))
#     breakpoint()
#     count = 0
#     for img in tqdm(data['images']):
#         if 'DET' in img['file_name']: # and 'VID' in img['file_name']:
#             print(img['file_name'])
#             count += 1

#     print(count)

for i in range(1, 556):
    shutil.move(f'test{i}', 'test_imagenet_vid_1vid_old_ckpt')

