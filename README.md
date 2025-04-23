# GHOSTLOOM
 # Created a computer vision application that generates an ”Invisible Cloak” effect.
 import cv2
import time
import numpy as np

def adjust_brightness_contrast(image, alpha, beta):
    """
    Adjust the brightness and contrast of the image.
    Alpha controls contrast, beta controls brightness.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def filter_mask(mask, open_kernel, close_kernel, dilation_kernel):
    """
    Apply morphological operations to clean up the mask.
    """
    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_OPEN, open_kernel)
    dilation = cv2.dilate(open_mask, dilation_kernel, iterations=1)
    return dilation

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Capture background frame
print("Capturing background. Please stay out of the frame...")
time.sleep(3)
background_frames = []
for _ in range(30):  # Capture multiple frames for a stable background
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture background frame.")
        exit()
    background_frames.append(frame)
    time.sleep(0.1)
background = np.median(background_frames, axis=0).astype(np.uint8)
background = cv2.flip(background, 1)  # Flip for mirror effect

# Define morphological kernels
open_kernel = np.ones((5, 5), np.uint8)
close_kernel = np.ones((7, 7), np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)

# Define HSV range for green cloak
lower_cloak = np.array([35, 50, 50])  # Lower bound for green
upper_cloak = np.array([85, 255, 255])  # Upper bound for green

# Skin tone HSV range
lower_skin1 = np.array([0, 20, 70])
upper_skin1 = np.array([20, 255, 255])
lower_skin2 = np.array([160, 20, 70])
upper_skin2 = np.array([180, 255, 255])

# Adjust brightness and contrast parameters
alpha = 1.5  # Contrast control (1.0-3.0)
beta = 50    # Brightness control (0-100)

print("Invisible cloak effect starting...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Flip the frame for consistency
    frame = cv2.flip(frame, 1)

    # Adjust brightness and contrast
    adjusted_frame = adjust_brightness_contrast(frame, alpha, beta)

    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)

    # Create cloak mask
    mask_cloak = cv2.inRange(hsv_frame, lower_cloak, upper_cloak)

    # Create skin tone mask
    mask_skin1 = cv2.inRange(hsv_frame, lower_skin1, upper_skin1)
    mask_skin2 = cv2.inRange(hsv_frame, lower_skin2, upper_skin2)
    mask_skin = cv2.bitwise_or(mask_skin1, mask_skin2)

    # Combine the masks
    mask = cv2.bitwise_or(mask_cloak, mask_skin)

    # Filter the mask
    mask = filter_mask(mask, open_kernel, close_kernel, dilation_kernel)

    # Create an inverse mask to retain non-cloak regions
    inverse_mask = cv2.bitwise_not(mask)

    # Apply masks to get the invisibility effect
    invisibility = cv2.bitwise_and(background, background, mask=mask)
    current_background = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    final_output = cv2.add(invisibility, current_background)

    # Display the result
    cv2.imshow("Invisible Cloak", final_output)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
