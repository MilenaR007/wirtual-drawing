import cv2
import mediapipe as mp
import numpy as np
import gradio as gr

# 1. Initialize the MediaPipe Hand Landmarker model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 2. Global variables to store the drawing canvas and finger position
canvas = None
prev_x, prev_y = 0, 0

def draw_on_video(frame):
    global canvas, prev_x, prev_y
    
    # Safety check for empty frames
    if frame is None:
        return frame
        
    # Initialize a blank virtual canvas if it's the first frame
    if canvas is None or canvas.shape != frame.shape:
        canvas = np.zeros_like(frame)
        
    # Mirror the image horizontally for a natural webcam interaction
    frame = cv2.flip(frame, 1)
    
    # Process the frame to find hand landmarks
    results = hands.process(frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Landmark #8 is the tip of the index finger
            index_finger = hand_landmarks.landmark[8]
            h, w, c = frame.shape
            
            # Convert relative coordinates (0.0 - 1.0) to pixel coordinates
            cx, cy = int(index_finger.x * w), int(index_finger.y * h)
            
            # If the hand was just raised, initialize the starting point
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy
                
            # Draw a thick magenta line from the previous position to the current position
            cv2.line(canvas, (prev_x, prev_y), (cx, cy), (255, 0, 255), 8)
            
            # Update the coordinates for the next frame
            prev_x, prev_y = cx, cy
            
            # Optional: Draw the full hand skeleton overlay
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # Reset the starting point when the hand leaves the frame 
        # to prevent connecting lines when the hand reappears
        prev_x, prev_y = 0, 0 
        
    # Overlay the canvas onto the original webcam frame
    output = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
    return output

# 3. Build the Gradio web interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Virtual Air Canvas (Gradio & MediaPipe)")
    gr.Markdown("Show your index finger to the camera to start drawing in the air!")
    
    with gr.Row():
        # Input component: Live webcam streaming
        video_in = gr.Image(sources=["webcam"], streaming=True, label="Your Camera")
        # Output component: Processed frame with the drawing
        video_out = gr.Image(label="Drawing Output")
        
    # Connect the camera stream to the processing function
    video_in.stream(draw_on_video, inputs=[video_in], outputs=[video_out])

# 4. Launch the application
if __name__ == "__main__":
    demo.launch()