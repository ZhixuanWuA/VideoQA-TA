"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import re
from PIL import Image

import pandas as pd
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset
from lavis.datasets.data_utils import load_video
import pdb

def generate_labelinput(video_name):
    
    json_file_path = os.path.join('/MSRVTT_Zero_Shot_QA/ori/iouframerevised', f"{video_name}.json")

    # Check if the JSON file exists
    if not os.path.exists(json_file_path):
        print(f"JSON file for video '{video_name}' not found in jsonmerge_dir")
        return ""

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Get all final labels and their corresponding trajectories
    final_labels = []
    trajectories_by_label = {}

    for key, value in data.items():
        final_label = value["final_label"]
        final_labels.append(final_label)
        trajectories = value["trajectory"]
        trajectories_by_label.setdefault(final_label, []).extend(trajectories)

    # Initialize the labelinput with the main description
    labelinput = ""

    # Dictionary to keep track of which objects appear in which frames
    included_frames = {}

    # Iterate over all final labels
    for final_label in final_labels:
        # Sort trajectories for the current label by frame length and representativeness
        representative_trajectories = sorted(
            trajectories_by_label[final_label],
            key=lambda x: (len(x["frame"]), x.get("representativeness", 0)),
            reverse=True
        )

        # Iterate over the selected representative trajectories
        for trajectory in representative_trajectories:
            frame = trajectory['frame']
            if frame not in included_frames:
                included_frames[frame] = set()  # Use a set to avoid duplicate labels
            included_frames[frame].add(final_label)

    frame_count = 0

    # Iterate over the frames in sorted order to build the labelinput
    for frame, objects in sorted(included_frames.items()):
        frame_count += 1
        # Create the description for the current frame
        frame_str = f"In the {frame_count} frame, it includes {', '.join(objects)}.\n"
        # Append the frame description to the labelinput
        labelinput += frame_str

        # Check if labelinput length is nearing the limit
        if len(labelinput) >= 2000:  # Adjust this threshold as needed to ensure the labelinput stays under 2048 characters
            break

    return labelinput

class MSRVTTVQADataset(VideoQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt='', split='train'):
        self.vis_root = vis_root

        self.annotation = {}
        for ann_path in ann_paths:
            self.annotation.update(json.load(open(ann_path)))
        self.question_id_list = list(self.annotation.keys())
        self.question_id_list.sort()
        self.fps = 10

        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt = prompt
        # self._add_instance_ids()
        # pdb.set_trace()

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."
        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]

        # Divide the range into num_frames segments and select a random index from each segment
        segment_list = np.linspace(0, ann['frame_length'], self.num_frames + 1, dtype=int)
        segment_start_list = segment_list[:-1]
        segment_end_list = segment_list[1:]
        selected_frame_index = []
        for start, end in zip(segment_start_list, segment_end_list):
            if start == end:
                selected_frame_index.append(start)
            else:
                selected_frame_index.append(np.random.randint(start, end))

        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join(self.vis_root, ann['video'], "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)
        # print(selected_frame_index, video.shape)
        
        # Generate labelinput description for the video
        # labelinput = self.generate_labelinput(ann['video'])
        labelinput = generate_labelinput(ann['video'])

        question = self.text_processor(ann["question"])
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
        answer = self.text_processor(ann["answer"])

        return {
            "image": video,
            "text_input": question,
            "text_output": answer,
            "question_id": ann["question_id"],
            "labelinput": labelinput,
            # "instance_id": ann["instance_id"],
        }
        
    def __len__(self):
        return len(self.question_id_list)

class MSRVTTVQAEvalDataset(MSRVTTVQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='test'):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='test')

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."
        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]

        selected_frame_index = np.rint(np.linspace(0, ann['frame_length']-1, self.num_frames)).astype(int).tolist()
        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join(self.vis_root, ann['video'], "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)
        
        # Generate labelinput description for the video
        # labelinput = generate_labelinput(ann['video'])

        question = self.text_processor(ann["question"])
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
        answer = self.text_processor(ann["answer"])

        return {
            "image": video,
            "text_input": question,
            "text_output": answer,
            "question_id": ann["question_id"],
            # "labelinput": labelinput,
            # "instance_id": ann["instance_id"],
        }