import cv2
import mediapipe as mp
import numpy as np
import gradio as gr
import time

# Initialize MediaPipe Hand Landmarker model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Global variables for managing the drawing state
drawings = []      # List to store history of drawn line segments
prev_x, prev_y = 0, 0
hue_value = 30     # Start at 30 (corresponds to a vibrant yellow in OpenCV HSV)

def draw_on_video(frame):
    global drawings, prev_x, prev_y, hue_value
    
    if frame is None:
        return frame
        
    current_time = time.time()
    
    # 1. TIMED ERASING (5-SECOND LIFESPAN)
    # Keep only the drawing segments that are younger than 5.0 seconds
    drawings = [segment for segment in drawings if current_time - segment['timestamp'] <= 5.0]
        
    # Mirror the image horizontally for natural user interaction
    frame = cv2.flip(frame, 1)
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
                # 2. RAINBOW EFFECT (HSV TO BGR GRADIENT)
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
                
                # Pushing the hue value forward to create the ombre transition
                # Moving yellow -> orange -> red -> purple -> blue -> green
                hue_value = (hue_value - 1) % 180
            
            prev_x, prev_y = cx, cy
    else:
        # Reset tracker when the hand leaves the frame to avoid unwanted bridging lines
        prev_x, prev_y = 0, 0 
        
    # 3. RENDERING THE SMOOTH BRUSH LINES (Fully Opaque)
    # Draw all valid segments stored in memory directly onto the webcam frame
    for segment in drawings:
        start_point = segment['start']
        end_point = segment['end']
        color = segment['color']
        thickness = 20  # Thicker line for a richer brush feel
        radius = 10     # Cap radius (exactly half of the line thickness)
        
        # Use cv2.LINE_AA for high-quality anti-aliased (smooth) lines
        cv2.line(frame, start_point, end_point, color, thickness, cv2.LINE_AA)
        
        # Overlay round caps at the joints to create a fluid, painted stroke effect
        cv2.circle(frame, start_point, radius, color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, end_point, radius, color, cv2.FILLED, cv2.LINE_AA)

    return frame

# Build the Gradio web UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Rainbow Air Brush (Gradio & MediaPipe)")
    gr.Markdown("Point your index finger at the screen to paint! The brush strokes feature a smooth ombre effect and fade away automatically after 5 seconds.")
    
    with gr.Row():
        video_in = gr.Image(sources=["webcam"], streaming=True, label="Your Webcam")
        video_out = gr.Image(label="Processed Canvas Output")
        
    video_in.stream(draw_on_video, inputs=[video_in], outputs=[video_out])

if __name__ == "__main__":
    demo.launch()