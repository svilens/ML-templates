from pytube import YouTube
import os

youtube_url = "some_url"
yt = YouTube(youtube_url)
streams = yt.streams.filter(file_extension="mp4")

output_dir = os.path.join(".", "video")
file_path = streams[0].download(output_path=output_dir)

from decord import VideoReader, cpu
import torch
import numpy as np
from huggingface_hub import hf_hub_download

np.random.seed(42)

# this does in-memory decoding of the video 
videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
print("Length of video frames: ", len(videoreader))

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    # Since each frame length is 4 seconds, we need to find the total frame length if we want `clip_len` frames 
    converted_len = int(clip_len * frame_sample_rate)

    # Get a random frame to end on 
    end_idx = np.random.randint(converted_len, seg_len)
    # Find the starting frame, if the frame has length of clip_len
    start_idx = end_idx - converted_len

    # np.linspace returns evenly spaced numbers over a specified interval 
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# video frames retrieval
indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=len(videoreader))
video = videoreader.get_batch(indices).asnumpy()


from transformers import XCLIPProcessor, XCLIPModel

model_name = 'microsoft/xclip-base-patch16-zero-shot'
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

# classify
import torch

text_description_list = ["play piano", "eat sandwich", "play football"]

inputs = processor(text=text_description_list, 
                   videos=list(video), 
                   return_tensors="pt", 
                   padding=True)

with torch.no_grad():
    outputs = model(**inputs)

video_probs = outputs.logits_per_video.softmax(dim=1)
print(dict(zip(text_description_list, video_probs[0])))
