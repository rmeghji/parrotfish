import cv2
import numpy as np
from identify_faces import FaceDetector
from preprocess import process_video, cut_video
from track_movement import detect_mouth_movement
import argparse

def create_annotated_video(video_path, output_path, N=1):
    """Create a video with colored rectangles indicating speaking status.
    
    Args:
        video_path: Path to input video
        output_path: Path for output video
        N: Process every Nth frame
    """
    frames_data, processed_frames, frame_indices = process_video(video_path, N)
    if frames_data is None:
        return
    
    movement_data = detect_mouth_movement(frames_data)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    target_width = processed_frames[0].shape[1]
    target_height = processed_frames[0].shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    GRAY = (128, 128, 128)
    GREEN = (0, 255, 0)
    
    frame_idx = 0
    processed_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        ratio = target_width / frame.shape[1]
        target_height = int(frame.shape[0] * ratio)
        frame = cv2.resize(frame, (target_width, target_height), 
                         interpolation=cv2.INTER_AREA)
            
        if frame_idx in frame_indices:
            speaking_status = {}
            if processed_idx < len(movement_data):
                for movement in movement_data[processed_idx]['movements']:
                    face_idx = movement['face_index']
                    speaking_status[face_idx] = movement['is_speaking']
            
            for face_idx, face_dict in enumerate(frames_data[processed_idx]):
                x, y, w, h = face_dict['rect']
                color = GREEN if speaking_status.get(face_idx, False) else GRAY
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            processed_idx += 1
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process video for speech detection.")
    parser.add_argument("-i", "--input_video", required=True, type=str, help="Path to the input video file.")
    parser.add_argument("-o", "--output_video", required=True, type=str, help="Path to the output video file.")
    parser.add_argument("-s", "--start_time", type=int, default=None, help="Time in seconds to start processing from")
    parser.add_argument("-l", "--length", type=float, default=None, help="Length of the video in seconds (beginning from start_time if provided)")
    args = parser.parse_args()

    N = 1

    input_video = args.input_video
    if args.start_time is not None and args.length is not None:
        import os
        base, ext = os.path.splitext(args.output_video)
        cut_video_path = f"{base}_cut{ext}"
        cut_video(input_video=args.input_video, output_video=cut_video_path, length=args.length, start_time=args.start_time)
        input_video = cut_video_path
        
    create_annotated_video(input_video, args.output_video, N)
    
    if args.start_time is not None and args.length is not None:
        if os.path.exists(cut_video_path):
            os.remove(cut_video_path)