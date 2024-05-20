import av
import numpy as np
from transformers import AutoImageProcessor, VideoMAEModel

np.random.seed(0)

# See: https://huggingface.co/docs/transformers/model_doc/videomae#videomae


def _read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def _sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    """
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def get_videmae_feature(subject_video_file_path):
    # video clip consists of 300 frames (10 seconds at 30 FPS)
    container = av.open(subject_video_file_path)
    print(container)

    # TODO: 全フレームに変える
    # sample 16 frames
    indices = _sample_frame_indices(
        clip_len=3, frame_sample_rate=1, seg_len=container.streams.video[0].frames
    )
    video = _read_video_pyav(container, indices)

    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

    # prepare video for the model
    inputs = image_processor(list(video), return_tensors="pt")

    # forward pass
    outputs = model(**inputs)
    return outputs.last_hidden_state
