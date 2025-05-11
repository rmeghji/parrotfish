import cv2
from identify_faces import FaceDetector

def process_video(video_path, N=3):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}, FPS: {fps:.2f}, Size: {frame_width}x{frame_height}, Total Frames: {total_frames}")

    frame_counter = 0
    processed_frame_count = 0

    detector = FaceDetector()

    frames_data = []
    original_frames = []
    frame_indices = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_counter % N == 0:
            processed_frame_count += 1

            target_width_resize = 640
            ratio = target_width_resize / frame.shape[1]
            target_height_resize = int(frame.shape[0] * ratio)
            resized_frame = cv2.resize(frame, (target_width_resize, target_height_resize), interpolation=cv2.INTER_AREA)

            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            face_img, face_rect = detector.detect_faces(gray_frame)

            faces = []
            for face in face_rect:
                face_img = resized_frame[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
                faces.append({"face": face_img, "rect": face})

            for face in faces:
                mouth_img, mouth_rect = detector.detect_mouth(face["face"])
                face["mouth"] = mouth_img

            frames_data.append(faces)
            original_frames.append(resized_frame)
            frame_indices.append(frame_counter)

        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

    return frames_data, original_frames, frame_indices

def cut_video(input_video, output_video, length, start_time=0):
    """Cut a video from start_time for specified length in seconds.
    
    Args:
        input_video (str): Path to input video file
        output_video (str): Path to output video file
        length (float): Length of output video in seconds
        start_time (float, optional): Start time in seconds. Defaults to 0.
    
    Returns:
        bool: True if successful, False otherwise
    """
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_video}")
        return False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    start_frame = 0
    if fps > 0:
        start_frame = int(start_time * fps)
    else:
        print(f"Warning: Could not determine FPS. Starting from the beginning.")
        start_time = 0

    video_duration = total_frames / fps if fps > 0 else total_frames / 30
    if start_time >= video_duration and total_frames > 0:
        print(f"Error: Start time ({start_time}s) is beyond the video duration ({video_duration:.2f}s).")
        cap.release()
        return False

    if start_time + length > video_duration and total_frames > 0:
        print(f"Warning: Requested duration ({start_time}s + {length}s = {start_time + length}s) exceeds video duration ({video_duration:.2f}s).")
        length = video_duration - start_time
        print(f"Adjusting length to {length:.2f}s to fit within video duration.")

    frames_to_keep = int(length * fps) if fps > 0 else int(length * 30)

    if start_frame > 0:
        print(f"Attempting to start from frame: {start_frame} (Time: {start_time}s)")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if abs(current_pos_frame - start_frame) > 1:
            print(f"Warning: Could not accurately seek to frame {start_frame}. Current frame: {int(current_pos_frame)}")

    out = cv2.VideoWriter(output_video, fourcc, fps if fps > 0 else 30, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for: {output_video}")
        cap.release()
        return False

    frame_count_session = 0

    print(f"Cutting video: {input_video}, starting from {start_time} seconds to {length} seconds ({frames_to_keep} frames).")

    while cap.isOpened() and frame_count_session < frames_to_keep:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_count_session += 1
        else:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= total_frames and total_frames > 0:
                print(f"Reached end of video after processing {frame_count_session} frames from start time.")
            else:
                print("Error reading frame or unexpected end of video stream.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if frame_count_session == frames_to_keep:
        print(f"\nSuccessfully cut video to {frame_count_session} frames ({length} seconds) starting from {start_time}s: {output_video}")
        return True
    elif frame_count_session > 0:
        actual_length = frame_count_session / fps if fps > 0 else frame_count_session / 30
        print(f"\nVideo cut to {frame_count_session} frames ({actual_length:.2f} seconds) starting from {start_time}s: {output_video}")
        return True
    else:
        print("\nNo frames were processed for this session.")
        return False

