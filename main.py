import argparse
import cv2
import numpy as np


def yellow_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 70, 70]), np.array([45, 255, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def apply_roi(mask, roi_top_ratio):
    h, w = mask.shape
    top = int(np.clip(roi_top_ratio, 0.0, 0.95) * h)
    if top <= 0:
        return mask

    roi = np.zeros_like(mask)
    roi[top:h, 0:w] = 255
    return cv2.bitwise_and(mask, roi)


def keep_main_component(mask, min_component_area=1200, bottom_margin=12):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    h, _ = mask.shape
    candidates = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        y = int(stats[label, cv2.CC_STAT_TOP])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_bottom = (y + comp_h) >= (h - max(bottom_margin, 0))
        candidates.append((label, area, touches_bottom))

    if not candidates:
        return np.zeros_like(mask)

    bottom_candidates = [entry for entry in candidates if entry[2]]
    if bottom_candidates:
        best_label = max(bottom_candidates, key=lambda x: x[1])[0]
    else:
        best_label = max(candidates, key=lambda x: x[1])[0]

    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def detect_line_center(mask, min_contour_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_contour_area:
        return None, None, None

    m = cv2.moments(contour)
    if m["m00"] == 0:
        return contour, None, None

    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return contour, cx, cy


def annotate(image_bgr, contour, cx, cy):
    out = image_bgr.copy()
    h, w = out.shape[:2]
    center_x = w // 2

    cv2.line(out, (center_x, 0), (center_x, h - 1), (255, 0, 0), 2)
    cv2.putText(out, "Frame center", (max(center_x - 70, 5), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if contour is None or cx is None or cy is None:
        cv2.putText(out, "No line detected", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out, None

    cv2.drawContours(out, [contour], -1, (0, 255, 0), 2)
    cv2.circle(out, (cx, cy), 6, (255, 255, 255), -1)
    cv2.line(out, (cx, 0), (cx, h - 1), (0, 255, 255), 2)

    offset = cx - center_x
    cv2.putText(out, f"Line center x={cx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(out, f"Offset from center={offset}px", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out, offset


def run_video(input_source, output_video, roi_top_ratio, min_component_area, min_contour_area):
    # Determine if input is a camera index (int) or file path (string)
    try:
        source = int(input_source)
        is_live_stream = True
    except ValueError:
        source = input_source
        is_live_stream = False

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {input_source}")

    # Get video properties to set the correct playback and save speed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Fallback if FPS cannot be read properly
    if fps == 0 or fps is None or np.isnan(fps):
        fps = 30.0

    # Calculate the dynamic wait delay (in milliseconds)
    # If it's a live camera feed, process as fast as possible (1ms delay)
    # If it's a saved video, wait the correct amount of time between frames
    playback_delay = 1 if is_live_stream else max(1, int(1000 / fps))

    out_writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video, fourcc, int(fps), (width, height))
        print(f"Saving output video to: {output_video}")

    print(f"Processing video at {fps} FPS... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        mask = yellow_mask(frame)
        mask_roi = apply_roi(mask, roi_top_ratio=roi_top_ratio)
        mask_cleaned = keep_main_component(mask_roi, min_component_area=min_component_area)

        contour, cx, cy = detect_line_center(mask_cleaned, min_contour_area=min_contour_area)
        annotated, offset = annotate(frame, contour, cx, cy)

        # Print the center information to the console for your flight controller
        if cx is None:
            print("No line detected")
        else:
            print(f"Line Center: ({cx}, {cy}) | Offset: {offset}px")

        cv2.imshow("Annotated Video", annotated)
        cv2.imshow("Mask", mask_cleaned)

        if out_writer:
            out_writer.write(annotated)

        # Use the calculated delay to play back at normal speed
        if cv2.waitKey(playback_delay) & 0xFF == ord('q'):
            print("Video processing interrupted by user.")
            break

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Video-based yellow line detector.")
    parser.add_argument("input_video", help="Input video path or camera index (e.g., 0 for webcam)")
    parser.add_argument("--output", default="",
                        help="Annotated output video path (e.g., output.mp4). Leave empty to just view.")
    parser.add_argument("--roi-top-ratio", type=float, default=0.05, help="Ignore top image area (0.0-0.95)")
    parser.add_argument("--component-area", type=int, default=1200, help="Minimum connected component area")
    parser.add_argument("--contour-area", type=int, default=100, help="Minimum contour area")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_video(
        input_source=args.input_video,
        output_video=args.output,
        roi_top_ratio=args.roi_top_ratio,
        min_component_area=args.component_area,
        min_contour_area=args.contour_area,
    )

