import numpy as np
import cv2
import torch
from PIL import Image
from models import UNET, transform
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNET().to(device)
model.load_state_dict(torch.load(r"C:\Users\Mayank\Desktop\Lane_det\segmentation_torch_unet.pth", map_location=device, weights_only=True))
model.eval()

def ensure_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image

def get_segmentation_mask(frame):
    try:
        frame_rgb = ensure_rgb(frame)
        img = Image.fromarray(frame_rgb)
        final_img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mask = model(final_img)
        
        output = (mask > 0.5).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return output
    except Exception as e:
        print(f"Error in segmentation mask: {e}")
        return np.ones((frame.shape[0], frame.shape[1]))

# Adaptive Canny Edge Detection
def adaptive_canny(frame):
    threshold = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    edges = cv2.Canny(frame, threshold * 0.75, threshold)
    print(edges.shape)
    plt.imshow(edges,cmap="gray")
    plt.show()
    return edges

# Dynamic Hough Line Transform
def detect_hough_lines(edges):
    try:
        # max(50, edges.shape[0] // 10)
        minLineLength = max(50, edges.shape[0] // 10)
        maxLineGap = max(20, edges.shape[1] // 20)
        # plt.imshow(cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=minLineLength, maxLineGap=maxLineGap),cmap="gray")
        # plt.show()
        return cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=minLineLength, maxLineGap=maxLineGap)
    except Exception as e:
        print(f"Error in Hough Line detection: {e}")
        return None

def draw_lines(frame, lines):

    """Draws Hough lines on the given frame"""
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Extract points
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)  # Draw line (Green)
    return frame

def split_lines(lines,frame):
    filtered_lines = []  # Store only valid lines

    if lines is None or len(lines) == 0:
        print("Warning: No lines detected!")
        return [], []

    # **Step 1: Remove lines with slope in |0| to |10|**
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # **Calculate the slope**
        if x2 - x1 == 0:  
            slope = float('inf')  # Vertical line (Keep it)
        else:
            slope = (y2 - y1) / (x2 - x1)

        # **Skip lines with slope in |0| to |10|**
        if 0 <= abs(slope) <= 10:
            continue  

        filtered_lines.append(line)

    draw_lines(frame,filtered_lines)
    # **Step 2: Split the remaining lines into left and right**
    left_lines, right_lines = [], []

    if not filtered_lines:  # No valid lines left after filtering
        print("Warning: No valid lines remaining after filtering!")
        return left_lines, right_lines

    # Extract x1 values and find the middle x-coordinate
    x_coords = [line[0][0] for line in filtered_lines]
    x_coords.sort()
    xmid = x_coords[len(x_coords) // 2]  # Middle x-value

    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        if x1 <= xmid:
            left_lines.append(line)
        else:
            right_lines.append(line)

    return left_lines, right_lines
# Fit a single line for each lane
def cluster_lines(lines, k_val=1):
    if lines is None or len(lines) == 0:
        print(" Warning: No lines found for clustering!")
        return []  # Return an empty list instead of causing an error

    line_features = np.array([
        [(x1 + x2) / 2, (y1 + y2) / 2, np.arctan2(y2 - y1, x2 - x1)] 
        for [[x1, y1, x2, y2]] in lines
    ])
        
    # Ensure at least k clusters exist
    k_val = min(k_val, len(lines))  # Prevent k > number of data points

    kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10).fit(line_features)
    labels = kmeans.labels_

    clustered_lines = [
        [[int(np.mean([lines[i][0][j] for i in np.where(labels == idx)[0]])) for j in range(4)]]
        for idx in range(k_val)
    ]
    return clustered_lines

# Lane detection pipeline
def lane_detection_pipeline(frame):
    try:
        frame_rgb = ensure_rgb(frame)
        blurred_frame = cv2.GaussianBlur(frame_rgb, (5,5), 1)
        mask = get_segmentation_mask(frame)
        edges = adaptive_canny(blurred_frame)
        mask_resized = cv2.resize(mask.astype(np.uint8), (edges.shape[1], edges.shape[0]), interpolation=cv2.INTER_NEAREST)
        inverted_mask = 1 - mask_resized
        edges = edges * inverted_mask
        lines = detect_hough_lines(edges)
        left_lines, right_lines = split_lines(lines, frame.shape[1])
        left_lane = cluster_lines(left_lines)
        right_lane = cluster_lines(right_lines)
        return [left_lane, right_lane]
    except Exception as e:
        print(f"Error in lane detection pipeline: {e}")
        return [None, None]

# Draw lanes
def draw_lanes_and_arrow(frame, lines):
    try:
        if lines is None or len(lines) < 2:
            print("Not enough lines to draw midline arrow!")
            return frame

        # Draw individual lane lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Compute midline coordinates
        x_start = (lines[0][0][0] + lines[1][0][0]) // 2  # (x1 of first + x1 of second) / 2
        y_start = (lines[0][0][1] + lines[1][0][1]) // 2  # (y1 of first + y1 of second) / 2
        x_end = (lines[0][0][2] + lines[1][0][2]) // 2  # (x2 of first + x2 of second) / 2
        y_end = (lines[0][0][3] + lines[1][0][3]) // 2  # (y2 of first + y2 of second) / 2

        # Draw the midline arrow
        cv2.arrowedLine(frame,  (x_start, y_start),(x_end, y_end),(0, 0, 255), 3, tipLength=0.3)

        return frame  # Return modified frame

    except Exception as e:
        print(f"Error in drawing lanes: {e}")
        return frame  # Return original frame if error

def process_frame(frame):

    frame_rgb = ensure_rgb(frame)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
# Apply bilateral filter
    d = 11              # Diameter of the pixel neighborhood
    sigma_color = 100    # Intensity variance (range kernel)
    sigma_space = 75    # Spatial variance (spatial kernel)
    blurred_frame = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)

    mask = get_segmentation_mask(frame)
    edges = adaptive_canny(blurred_frame)
    mask_resized = cv2.resize(mask.astype(np.uint8), (edges.shape[1], edges.shape[0]), interpolation=cv2.INTER_NEAREST)

    inverted_mask = 1 - mask_resized
    edges = edges * inverted_mask
    # plt.imshow(edges,cmap="gray")
    # plt.show()
    lines = detect_hough_lines(edges)
    draw_lines(frame,lines)
    if lines is None or len(lines) == 0:  
        print("No lines detected. Returning unprocessed frame.")
        return frame_rgb  # Return the original frame

    left_lines, right_lines = split_lines(lines,frame)

    # Handle missing lane lines
    if not left_lines and right_lines:
        print("No left lines found. Using right lines for both lanes.")
        left_lines = right_lines
    elif not right_lines and left_lines:
        print("No right lines found. Using left lines for both lanes.")
        right_lines = left_lines
    elif not left_lines and not right_lines:
        print("No lane lines detected. Returning unprocessed frame.")
        return frame_rgb  # Return the original frame if no lanes detected

    left_lane = cluster_lines(left_lines)
    right_lane = cluster_lines(right_lines)

    if not left_lane or not right_lane:  
        print("No lane detected after clustering. Returning unprocessed frame.")
        return frame_rgb  # Return original frame if clustering fails

    clustered_lines = [left_lane[0], right_lane[0]]
    return draw_lanes_and_arrow(frame_rgb, clustered_lines)

def trial(frame):
    plt.imshow(process_frame(frame))
    plt.show()

def process_video(input_path, output_path):
    """Reads a video, processes each frame, and saves the output video in MP4 format."""
    
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)  

    # Ensure FPS is valid
    if fps <= 0 or fps is None:
        fps = 30  # Default to 30 FPS if not detected

    # Define the codec and create VideoWriter object (H.264 encoding for MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try 'mp4v' instead of 'avc1'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


    print(f"Processing video: {input_path}")
    print(f"Output will be saved as: {output_path} (FPS: {fps}, Resolution: {frame_width}x{frame_height})")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Processed {frame_count} frames. End of video reached.")
            break  # Stop when video ends

        # 🔹 Try processing the frame
        try:
            processed_frame = process_frame(frame)  # Apply processing if possible
            if processed_frame is None:
                print(f"Warning: Processed frame is None at frame {frame_count}. Writing original frame instead.")
                processed_frame = frame  # Use the original frame if processing fails

        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}. Writing original frame instead.")
            processed_frame = frame  # Use the original frame if processing crashes

        out.write(processed_frame)  # Write frame to output
        frame_count += 1

    cap.release()
    out.release()

    print("Processing complete. Video saved.")

    # Handle GUI-related OpenCV issues
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        print("Skipping cv2.destroyAllWindows() due to missing GUI support.")

    # Process video frame by frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached or error reading frame.")
            break

        processed_frame = process_frame(frame)  # Apply processing
        out.write(processed_frame)  # Save processed frame
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("✅ Video processing complete!")
