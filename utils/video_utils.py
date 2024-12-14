import cv2
from tqdm import tqdm

def read_video(video_path):
    print(f"Reading video from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    print('Reading Done...')
    print(f'Total Frames: {len(frames)}')
    return frames


def write_video(video_path, frames, fps=24):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in tqdm(frames, desc='Writing Video'):
        out.write(frame)
    out.release()
