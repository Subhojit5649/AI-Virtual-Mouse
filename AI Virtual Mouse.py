import cv2
import numpy as np
import pyautogui

# Function to detect hand landmarks
def detect_hand_landmarks(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Your code for hand landmark detection goes here
    # This could involve using a pre-trained model or hand detection algorithm
    # For simplicity, we'll use a basic hand detection using thresholding as an example
    
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assumed to be the hand)
        hand_contour = max(contours, key=cv2.contourArea)
        
        # Find the centroid of the hand contour
        M = cv2.moments(hand_contour)
        if M["m00"] != 0:
            hand_x = int(M["m10"] / M["m00"])
            hand_y = int(M["m01"] / M["m00"])
            return hand_x, hand_y, hand_contour
    
    return None, None, None

# Main function for controlling the virtual mouse
def control_virtual_mouse():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally

        # Detect hand landmarks
        hand_x, hand_y, hand_contour = detect_hand_landmarks(frame)

        if hand_x is not None and hand_y is not None:
            # Move the mouse cursor
            screen_width, screen_height = pyautogui.size()
            mouse_x = hand_x * screen_width / frame.shape[1]
            mouse_y = hand_y * screen_height / frame.shape[0]
            pyautogui.moveTo(mouse_x, mouse_y)

            # Check for left-click gesture (e.g., spreading fingers)
            if cv2.contourArea(hand_contour) > 10000:
                pyautogui.click()
            # Check for right-click gesture (e.g., closing fingers)
            elif cv2.contourArea(hand_contour) < 5000:
                pyautogui.rightClick()

            # Check for scroll gesture (e.g., moving hand up or down)
            if 5000 < cv2.contourArea(hand_contour) < 10000:
                pyautogui.scroll(1)  # Scroll up
            elif cv2.contourArea(hand_contour) > 20000:
                pyautogui.scroll(-1)  # Scroll down

            # Display the hand landmark
            cv2.circle(frame, (hand_x, hand_y), 10, (0, 255, 0), -1)  # Display hand landmark as a circle

        # Display the frame with the virtual mouse cursor
        cv2.imshow('Virtual Mouse', frame)

        # Exit the program when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the main function
if __name__ == "__main__":
    control_virtual_mouse()
