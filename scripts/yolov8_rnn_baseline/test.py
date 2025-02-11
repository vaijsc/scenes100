import os
import torch
import json
from tqdm import tqdm
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_AP(video_index, file_path):
    with open(file_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Check if the line starts with the specified video index
            if line.startswith(f"Video {video_index}:"):
                # Split the line to extract the mAP and AP50 values
                parts = line.split(',')
                # Extract the mAP and AP50 from the corresponding parts
                map_value = parts[0].split('mAP ')[1]
                ap50_value = parts[1].split('AP50 ')[1]
                if map_value == 'nan' or ap50_value == 'nan':
                    return 0.0, 0.0
                return float(map_value), float(ap50_value)

    # If the video index is not found
    return None, None

# num_vids = 555
# APG = []
# APG50 = []
# base_APs = []
# base_AP50s = []
# ids = []
# with open('result_fgfa_base.txt', 'r') as base:
#     for i in tqdm(range(num_vids)):
#         data = []
#         if not os.path.exists(f'test_imagenet_vid_1vid_fgfa_ckpt_shuffled/test{i+1}/metrics.json'):
#             continue
#         with open(f'test_imagenet_vid_1vid_fgfa_ckpt_shuffled/test{i+1}/metrics.json', 'r') as f:
#             # breakpoint()        
#             AP = 0
#             AP50 = 0
#             baseAP, baseAP50 = get_AP(i, 'result_fgfa_base.txt')
#             for line in f:
#                 result = json.loads(line)
                
#                 if 'iteration' in result and result['iteration'] == 39 and 'bbox/AP' in result:
#                     # breakpoint()
#                     AP = result['bbox/AP']
#                     AP50 = result['bbox/AP50']
#                     if math.isnan(AP):
#                         AP = 0.0
#                     if math.isnan(AP50):
#                         AP50 = 0.0
#                     break


#                 # if 'bbox/AP' in result:
#                 #     if not math.isnan(result['bbox/AP']) and result['bbox/AP'] > AP:
#                 #         AP = result['bbox/AP']
#                 #         AP50 = result['bbox/AP50']
            
#             if baseAP is not None and baseAP50 is not None and not math.isnan(AP) and not math.isnan(AP50):
#                 APG.append(AP-baseAP)
#                 APG50.append(AP50-baseAP50)
#                 base_APs.append(baseAP)
#                 base_AP50s.append(baseAP50)
#                 ids.append(i)


# bad_id = [i for i in range(len(base_APs)) if base_APs[i] <= 96]
# ids=bad_id
# base_APs = [base_APs[i] for i in bad_id]
# base_AP50s = [base_AP50s[i] for i in bad_id]
# APG = [APG[i] for i in bad_id]
# APG50 = [APG50[i] for i in bad_id]

# print(f"Mean base AP: {sum(base_APs)/len(base_APs)}")
# print(f"Mean base AP50: {sum(base_AP50s)/len(base_AP50s)}")
# print(f'Average AP gain: {sum(APG)/len(APG)}')       
# print(f'Average AP50 gain: {sum(APG50)/len(APG50)}')        
# plt.figure(figsize=(15, 6))  # Adjust the width (10) as needed to make the columns wider

# sns.barplot(x=ids, y=APG)
# plt.xticks(rotation=90, fontsize=1)
# plt.xlabel("Video IDs in val set")  # Label for x-axis
# plt.ylabel("AP gain")  # Label for y-axis
# plt.title(f"AP gain for the first {num_vids} videos in ImageNet VID val set")
# plt.savefig("APG_ImageNetVID_val.pdf")



num_vids = 555
APG = []
APG50 = []
base_APs = []
base_AP50s = []
ids = []
with open('result_fgfa_base.txt', 'r') as base:
    with open('result_fgfa_trained.txt', 'r') as trained:
        for i in tqdm(range(num_vids)):
            base_AP, base_AP50 = get_AP(i, 'result_fgfa_base.txt')
            trained_AP, trained_AP50 = get_AP(i, 'result_fgfa_trained.txt')
            if base_AP is not None and trained_AP is not None:
                APG.append(trained_AP-base_AP)
                APG50.append(trained_AP50-base_AP50)
                base_APs.append(base_AP)
                base_AP50s.append(base_AP50)
                ids.append(i)

print(len([x for x in base_APs if x < 10]))
print(f"Mean base AP: {sum(base_APs)/len(base_APs)}")
print(f"Mean base AP50: {sum(base_AP50s)/len(base_AP50s)}")
print(f'Average AP gain: {sum(APG)/len(APG)}')       
print(f'Average AP50 gain: {sum(APG50)/len(APG50)}')        
plt.figure(figsize=(15, 6))  # Adjust the width (10) as needed to make the columns wider

sns.barplot(x=ids, y=APG)
plt.xticks(rotation=90, fontsize=1)
plt.xlabel("Video IDs in val set")  # Label for x-axis
plt.ylabel("AP gain")  # Label for y-axis
plt.title(f"AP gain for the first {num_vids} videos in ImageNet VID val set (valided)")
plt.savefig("APG_ImageNetVID_val_valided.pdf")

# Define the AP bins and labels
bins = np.arange(0, 110, 10)
bin_labels = [f"{i}-{i+10}" for i in range(0, 100, 10)]

# Create a DataFrame to handle the data
df = pd.DataFrame({'AP': base_APs, 'APG': APG})

# Bin the AP values and calculate the mean APG and count the occurrences in each bin
df['AP_range'] = pd.cut(df['AP'], bins=bins, labels=bin_labels, include_lowest=True)
mean_apg_per_bin = df.groupby('AP_range')['APG'].mean()
count_per_bin = df.groupby('AP_range')['AP'].size()

# Combine the mean APG and count data into a single DataFrame
result = pd.DataFrame({'Mean_APG': mean_apg_per_bin, 'Count': count_per_bin})

# Plotting the side-by-side bar chart for Mean APG and Count
fig, ax = plt.subplots(figsize=(10, 6))

# Define the width for the bars and positions for each group
bar_width = 0.35
index = np.arange(len(bin_labels))

# Plot Mean APG and Count side by side
bars1 = ax.bar(index, result['Mean_APG'], bar_width, label='Mean APG', color='skyblue')
bars2 = ax.bar(index + bar_width, result['Count'], bar_width, label='Count', color='lightgreen')

# Adding labels, title, and custom x-axis tick labels
ax.set_xlabel('AP Range')
ax.set_ylabel('Values')
ax.set_title('Mean APG and Count per AP Range')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(bin_labels)
ax.legend()
# Add more y-ticks
ax.set_yticks(np.arange(-10, 100 + 1, 10))

# Add grid
ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
# Display the plot
plt.tight_layout()
plt.savefig("Mean_APG_per_AP_range.pdf")