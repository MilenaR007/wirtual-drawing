import cv2
import mediapipe as mp
import numpy as np
import gradio as gr
import time

# Initialize MediaPipe Hand Landmarker model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global variables for managing the drawing state
drawings = []
prev_x, prev_y = 0, 0
hue_value = 30 

def draw_on_video(frame):
    global drawings, prev_x, prev_y, hue_value
    
    if frame is None:
        return frame
        
    # FIX: Create a writable copy of the incoming read-only frame
    frame = frame.copy()
        
    current_time = time.time()
    
    # Keep only the drawing segments that are younger than 5.0 seconds
    drawings = [segment for segment in drawings if current_time - segment['timestamp'] <= 5.0]
        
    # Process the frame to find hand landmarks
    results = hands.process(frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Landmark #8 is the tip of the index finger
            index_finger = hand_landmarks.landmark[8]
            h, w, c = frame.shape
            
            # Convert relative coordinates to pixel coordinates
            cx, cy = int(index_finger.x * w), int(index_finger.y * h)
            
            # Initialize starting point if the hand just entered the frame
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy
                
            # If the finger moved, create a new brush stroke segment
            if (prev_x, prev_y) != (cx, cy):
                # Generate a smooth color transition using the HSV color space
                color_hsv = np.uint8([[[hue_value, 255, 255]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                color_tuple = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
                
                # Append the new line segment with its current timestamp
                drawings.append({
                    'start': (prev_x, prev_y),
                    'end': (cx, cy),
                    'color': color_tuple,
                    'timestamp': current_time
                })
                
                # Rapid color change (shifting hue by 8 degrees)
                hue_value = (hue_value - 8) % 180
            
            prev_x, prev_y = cx, cy
    else:
        # Reset tracker when the hand leaves the frame
        prev_x, prev_y = 0, 0 
        
    # Render the brush strokes
    for segment in drawings:
        start_point = segment['start']
        end_point = segment['end']
        color = segment['color']
        thickness = 20
        radius = 10
        
        # Use cv2.LINE_AA for high-quality anti-aliased (smooth) lines
        cv2.line(frame, start_point, end_point, color, thickness, cv2.LINE_AA)
        cv2.circle(frame, start_point, radius, color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, end_point, radius, color, cv2.FILLED, cv2.LINE_AA)

    return frame

# Build the Web UI
with gr.Blocks() as demo:
    gr.Markdown("# Wirtual Brush")
    gr.Markdown("Fixed UI layout and disabled mirror reflection. Colors shift rapidly, and strokes vanish after 5 seconds.")
    
    with gr.Row():
        video_in = gr.Image(sources=["webcam"], streaming=True, label="Webcam Input", height=480)
        video_out = gr.Image(label="Live Canvas", height=480)
        
    video_in.stream(draw_on_video, inputs=[video_in], outputs=[video_out])

if __name__ == "__main__":
    demo.launch()