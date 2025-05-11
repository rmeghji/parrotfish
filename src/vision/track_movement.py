import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

def detect_mouth_movement(frames: List[List[Dict]], window_size: int = 3) -> List[Dict]:
    """
    Analyze mouth movement using a sliding window of frames to detect speech
    
    Args:
        frames: List of frames, where each frame contains a list of detected faces
              with their corresponding mouth regions
        window_size: Number of frames to consider in the sliding window for movement detection
    
    Returns:
        List of dictionaries containing movement metrics for each frame
    """
    movement_data = []
    
    for i in range(len(frames)):
        frame_movement = []
        
        window_start = max(0, i - window_size)
        window_end = min(len(frames), i + window_size + 1)
        window_frames = frames[window_start:window_end]
        
        if len(window_frames) < 2:
            continue
        
        current_frame = frames[i]
        
        for face_idx, current_face in enumerate(current_frame):
            max_movement = {
                'area_change': 0,
                'centroid_movement': 0,
                'shape_difference': float('inf'),
                'face_index': face_idx,
                'is_speaking': False
            }
            
            current_mouth = cv2.cvtColor(current_face['mouth'], cv2.COLOR_BGR2GRAY) \
                if len(current_face['mouth'].shape) == 3 else current_face['mouth']
            
            for compare_frame in window_frames:
                if len(compare_frame) <= face_idx:
                    continue
                    
                compare_face = compare_frame[face_idx]
                compare_mouth = cv2.cvtColor(compare_face['mouth'], cv2.COLOR_BGR2GRAY) \
                    if len(compare_face['mouth'].shape) == 3 else compare_face['mouth']
                
                metrics = calculate_movement_metrics(current_mouth, compare_mouth)
                
                if metrics is not None:
                    max_movement['area_change'] = max(max_movement['area_change'], 
                                                    metrics['area_change'])
                    max_movement['centroid_movement'] = max(max_movement['centroid_movement'], 
                                                          metrics['centroid_movement'])
                    max_movement['shape_difference'] = min(max_movement['shape_difference'], 
                                                         metrics['shape_difference'])
            
            max_movement['is_speaking'] = is_speaking(
                max_movement['area_change'],
                max_movement['centroid_movement'],
                max_movement['shape_difference']
            )
            frame_movement.append(max_movement)
        
        movement_data.append({
            'frame_index': i,
            'movements': frame_movement
        })
    
    return movement_data

def calculate_movement_metrics(current_mouth: np.ndarray, next_mouth: np.ndarray) \
        -> Optional[Dict]:
    """
    Calculate various metrics to quantify mouth movement between two frames.
    
    Args:
        current_mouth: Grayscale image of mouth region in current frame
        next_mouth: Grayscale image of mouth region in next frame
    
    Returns:
        Dictionary containing movement metrics, or None if calculation fails
    """
    try:
        if current_mouth.shape != next_mouth.shape:
            current_mouth = cv2.resize(current_mouth, 
                                     (next_mouth.shape[1], next_mouth.shape[0]))
        
        _, current_thresh = cv2.threshold(current_mouth, 100, 255, 
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, next_thresh = cv2.threshold(next_mouth, 100, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        current_contours, _ = cv2.findContours(current_thresh, cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
        next_contours, _ = cv2.findContours(next_thresh, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
        
        current_contour = max(current_contours, key=cv2.contourArea, 
                            default=None)
        next_contour = max(next_contours, key=cv2.contourArea, 
                         default=None)
        
        if current_contour is None or next_contour is None:
            return None
        
        current_area = cv2.contourArea(current_contour)
        next_area = cv2.contourArea(next_contour)
        area_change = abs(next_area - current_area)
        
        current_M = cv2.moments(current_contour)
        next_M = cv2.moments(next_contour)
        
        if current_M['m00'] == 0 or next_M['m00'] == 0:
            return None
            
        current_cx = int(current_M['m10'] / current_M['m00'])
        current_cy = int(current_M['m01'] / current_M['m00'])
        next_cx = int(next_M['m10'] / next_M['m00'])
        next_cy = int(next_M['m01'] / next_M['m00'])
        
        centroid_movement = np.sqrt((next_cx - current_cx)**2 + 
                                   (next_cy - current_cy)**2)
        
        current_hu = cv2.HuMoments(current_M).flatten()
        next_hu = cv2.HuMoments(next_M).flatten()
        shape_diff = np.sum(np.abs(current_hu - next_hu))
        
        return {
            'area_change': area_change,
            'centroid_movement': centroid_movement,
            'shape_difference': shape_diff,
            'is_speaking': is_speaking(area_change, centroid_movement, shape_diff)
        }
        
    except Exception as e:
        print(f"Error calculating movement metrics: {str(e)}")
        return None

def is_speaking(area_change: float, centroid_movement: float, 
                shape_difference: float) -> bool:
    """
    Determine if the mouth movement indicates speech based on the calculated metrics.
    
    These thresholds are set to be more sensitive to subtle mouth movements.
    
    Args:
        area_change: Change in contour area between frames
        centroid_movement: Movement of the contour centroid between frames
        shape_difference: Difference in shape between frames using Hu Moments
    
    Returns:
        Boolean indicating whether the movement likely indicates speech
    """
    AREA_THRESHOLD = 50
    CENTROID_THRESHOLD = 5
    SHAPE_THRESHOLD = 0.15
    
    conditions_met = 0
    
    if area_change > AREA_THRESHOLD:
        conditions_met += 1
    if centroid_movement > CENTROID_THRESHOLD:
        conditions_met += 1
    if shape_difference < SHAPE_THRESHOLD:
        conditions_met += 1
    
    return conditions_met >= 3