import numpy as np
import cv2
import torch
from PIL import Image
from models import UNET, transform
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load model once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNET().to(device)
model.load_state_dict(torch.load(r"C:\Users\yagni\PythonProjects\Pathpal\Path_navigation\segmentation_torch_unet.pth", map_location=device, weights_only=True))
model.eval()

# Function to ensure image is RGB
def ensure_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image

def bil_filter(frame) :
    d = 13
    sigma_color = 100
    sigma_space =90
    return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
# Get segmentation mask from model
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
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        edges = cv2.Canny(gray, threshold * 0.5, threshold)
        # edges = cv2.Canny(gray, 150, 300)
        return edges
    except Exception as e:
        print(f"Error in Canny edge detection: {e}")
        return np.zeros_like(frame)

# Dynamic Hough Line Transform
def detect_hough_lines(edges):
    try:
        minLineLength = max(50, edges.shape[0] // 10)
        maxLineGap = max(20, edges.shape[1] // 20)
        return cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=minLineLength, maxLineGap=maxLineGap)
    except Exception as e:
        print(f"Error in Hough Line detection: {e}")
        return None

def split_lines(lines, xmid):
    left_lines, right_lines = [], []
    
    # if lines is None:  # Handle None or empty input safely
    #     print("Warning: No lines detected!")
    #     return left_lines, right_lines

    # x_coords = [line[0][0] for line in lines]  # Extract x1 values

    # if x_coords is None:  # Ensure x_coords is not empty
    #     print("Warning: No valid x-coordinates found!")
    #     return left_lines, right_lines
    
    # x_coords.sort()  # Sort in place
    # xmid = x_coords[len(x_coords) // 2]  # Find the middle x-value
    for line in lines:
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
# def lane_detection_pipeline(frame):
#     try:
#         frame_rgb = ensure_rgb(frame)
#         blu  rred_frame = cv2.bilateralFilter(frame_rgb)
#         mask = get_segmentation_mask(frame)
#         edges = adaptive_canny(blurred_frame)
#         mask_resized = cv2.resize(mask.astype(np.uint8), (edges.shape[1], edges.shape[0]), interpolation=cv2.INTER_NEAREST)
#         inverted_mask = 1 - mask_resized
#         edges = edges * inverted_mask
#         lines = detect_hough_lines(edges)
#         left_lines, right_lines = split_lines(lines, frame.shape[1])
#         left_lane = cluster_lines(left_lines)
#         right_lane = cluster_lines(right_lines)
#         return [left_lane, right_lane]
#     except Exception as e:
#         print(f"Error in lane detection pipeline: {e}")
#         return [None, None]

# Draw lines
def draw_lanes_and_arrow(frame, lines):
    try:
        if lines is None or len(lines) < 2:
            print("Not enough lines to draw midline arrow!")
            return frame

        # Draw individual lane lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Compute midline starting point (centroid of first two lines)
        x_start = (lines[0][0][0] + lines[0][0][2] + lines[1][0][0] + lines[1][0][2]) // 4
        y_start = (lines[0][0][1] + lines[0][0][3] + lines[1][0][1] + lines[1][0][3]) // 4

        # Compute slopes of both lines
        def slope(x1, y1, x2, y2):
            if y2>y1 :
                return (y2 - y1) / (x2 - x1 + 1e-6)
            else :
                return -(y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero

        slope1 = slope(*lines[0][0])
        slope2 = slope(*lines[1][0])
        avg_slope = (slope1 + slope2) / 2

        # Compute lengths of both lines
        def line_length(x1, y1, x2, y2):
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        len1 = line_length(*lines[0][0])
        len2 = line_length(*lines[1][0])
        arrow_length = max(len1, len2)  # Use the longer lane length

        # Compute end point using the average slope and length
        dx = int(250 / np.sqrt(1 + avg_slope ** 2))
        dy = int(avg_slope * dx)

        x_end = x_start + dx
        y_end = y_start + dy

        # Adjust direction based on y_start and y_end
        if y_start < y_end:
            cv2.arrowedLine(frame, (x_end, y_end), (x_start, y_start), (0, 0, 255), 3, tipLength=0.3)
        else:
            cv2.arrowedLine(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 3, tipLength=0.3)

        return frame  # Return modified frame

    except Exception as e:
        print(f"Error in drawing lanes: {e}")
        return frame  # Return original frame if error

def process_frame(frame):
    frame_rgb = ensure_rgb(frame)
    blurred_frame = bil_filter(frame_rgb)
    mask = get_segmentation_mask(frame)
    edges = adaptive_canny(blurred_frame)
    mask_resized = cv2.resize(mask.astype(np.uint8), (edges.shape[1], edges.shape[0]), interpolation=cv2.INTER_NEAREST)
    inverted_mask = 1 - mask_resized
    edges = edges * inverted_mask
    # return edges
    lines = detect_hough_lines(edges)

    if lines is None or len(lines) == 0:  
        print("No lines detected. Returning unprocessed frame.")
        return frame_rgb  # Return the original frame

    left_lines, right_lines = split_lines(lines, (edges.shape[1])//2)

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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is more reliable than 'avc1'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video: {input_path}")
    print(f"Output will be saved as: {output_path} (FPS: {fps}, Resolution: {frame_width}x{frame_height})")

    frame_count = 0
    frame_id = 0  # To process every Nth frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Processed {frame_count} frames. End of video reached.")
            break  # Stop when video ends

        # Process every 2nd frame (adjustable)
        if frame_id % 10 == 0:
            try:
                processed_frame = process_frame(frame)  # Apply processing if possible
                if processed_frame is None:
                    print(f"Warning: No processed output at frame {frame_count}. Writing original frame.")
                    processed_frame = frame  # Use the original frame if processing fails
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}. Writing original frame instead.")
                processed_frame = frame  # Use original frame on error

            # cv2.imshow(f"{frame_count}", processed_frame)
            # print(type(processed_frame))
            out.write(processed_frame)  # Write frame to output
            # plt.imshow(processed_frame)
            # plt.show()
        frame_count += 1
        frame_id += 1  # Increment frame counter

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    # Release resources
    cap.release()
    out.release()

    # Handle OpenCV GUI issues
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        print("Skipping cv2.destroyAllWindows() due to missing GUI support.")

    print(" Video processing complete!")

# Run the script
if __name__ == "__main__":
    input_video = r"C:\Users\yagni\PythonProjects\Pathpal\Path_navigation\lanedet_input_video.mp4"  # Update with the correct path
    output_video = r"C:\Users\yagni\PythonProjects\Pathpal\Path_navigation\output1.mp4"
    process_video(input_video, output_video)
    # frame = cv2.imread(r"C:\Users\yagni\PythonProjects\Pathpal\Path_navigation\test_images\testimg_insti.jpg")
    # e = process_frame(frame)
    # # # Show both images side by side
    # # plt.figure(figsize=(10, 5))

    # # # First image (e)
    # # plt.subplot(1, 2, 1)
    # plt.imshow(e, cmap="gray")
    # # plt.title("Processed Image 1")
    # # plt.axis("off")

    # # # Second image (v)
    # # plt.subplot(1, 2, 2)
    # # plt.imshow(v, cmap="gray")
    # # plt.title("Processed Image 2")
    # # plt.axis("off")

    # plt.show()