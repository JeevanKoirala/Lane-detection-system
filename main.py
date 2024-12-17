import cv2
import numpy as np

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    return blurred

def detect_edges(blurred_image):
    median_intensity = np.median(blurred_image)
    lower_threshold = int(max(0, 0.5 * median_intensity))
    upper_threshold = int(min(255, 1.5 * median_intensity))
    return cv2.Canny(blurred_image, lower_threshold, upper_threshold)

def define_region_of_interest(image):
    height, width = image.shape[:2]
    region_vertices = np.array([
        [
            (width * 0.1, height),
            (width * 0.9, height),
            (width * 0.8, height * 0.6),
            (width * 0.2, height * 0.6)
        ]
    ], dtype=np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, region_vertices, 255)
    return cv2.bitwise_and(image, mask)

def detect_lane_lines(edge_image):
    return cv2.HoughLinesP(
        edge_image, 
        rho=1, 
        theta=np.pi / 180, 
        threshold=30,
        minLineLength=40,
        maxLineGap=10
    )

def classify_lane_lines(frame, lines):
    if lines is None:
        return None

    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1 + np.finfo(float).eps)
            
            if abs(slope) < 0.3:
                continue
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    def average_line(lines_group):
        if not lines_group:
            return None
        x_coords = []
        y_coords = []
        for line in lines_group:
            x_coords.extend([line[0][0], line[0][2]])
            y_coords.extend([line[0][1], line[0][3]])
        
        coefficients = np.polyfit(x_coords, y_coords, 1)
        return coefficients

    left_lane = average_line(left_lines)
    right_lane = average_line(right_lines)

    return left_lane, right_lane

def draw_lane_lines(frame, lane_parameters):
    line_image = np.zeros_like(frame)
    height, width = frame.shape[:2]

    if lane_parameters[0] is not None:
        left_slope, left_intercept = lane_parameters[0]
        y1, y2 = height, int(height * 0.75)
        x1 = int((y1 - left_intercept) / left_slope)
        x2 = int((y2 - left_intercept) / left_slope)
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    if lane_parameters[1] is not None:
        right_slope, right_intercept = lane_parameters[1]
        y1, y2 = height, int(height * 0.75)
        x1 = int((y1 - right_intercept) / right_slope)
        x2 = int((y2 - right_intercept) / right_slope)
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)

    return line_image

def process_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    frame_width = 800
    frame_height = 800

    previous_left_line = None
    previous_right_line = None

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (frame_width, frame_height))
        
        blurred_frame = preprocess_frame(resized_frame)
        edge_image = detect_edges(blurred_frame)
        region_edges = define_region_of_interest(edge_image)
        
        lane_lines = detect_lane_lines(region_edges)
        lane_parameters = classify_lane_lines(resized_frame, lane_lines)
        
        if lane_parameters:
            left_line, right_line = lane_parameters
            if previous_left_line is not None:
                left_line = (0.8 * np.array(previous_left_line) + 0.2 * np.array(left_line)).tolist()
            if previous_right_line is not None:
                right_line = (0.8 * np.array(previous_right_line) + 0.2 * np.array(right_line)).tolist()
            
            previous_left_line = left_line
            previous_right_line = right_line
        else:
            left_line = previous_left_line
            right_line = previous_right_line

        lane_overlay = draw_lane_lines(resized_frame, [left_line, right_line])
        result_frame = cv2.addWeighted(resized_frame, 0.8, lane_overlay, 1, 0)
        cv2.imshow("Lane Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

process_video("test_video.mp4")
