import argparse
import cv2
import numpy as np
import time
import math

# Use the official Pi 5 camera library (bypasses GStreamer/libcamera bugs)
from picamera2 import Picamera2


def process_frame(frame, args):
    """
    Processes a BGR frame to find the yellow line and calculate drone navigation telemetry.
    Returns the annotated image, the mask, the Cross-Track Error (Offset), and the Angle.
    """
    h, w = frame.shape[:2]
    center_x = w // 2
    out = frame.copy()

    # Draw the vertical center target line for visual reference
    cv2.line(out, (center_x, 0), (center_x, h - 1), (255, 0, 0), 1)

    # 1. Color Masking (Yellow bounds tailored for varying indoor/outdoor light)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 60, 60])
    upper_yellow = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 2. ROI Cropping (Look forward, ignore the horizon/ceiling)
    top_crop = int(np.clip(args.roi_top_ratio, 0.0, 0.95) * h)
    if top_crop > 0:
        mask[:top_crop, :] = 0  # Instantaneous NumPy array slicing

    # 3. Clean up noise (Prevent the drone from chasing random yellow specks)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Find the Line
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx, offset, angle = None, None, None

    if contours:
        # Assume the largest yellow object is our path
        best_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(best_contour) > args.contour_area:
            # --- METRIC 1: Cross-Track Error (Offset for ROLL) ---
            m = cv2.moments(best_contour)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                offset = cx - center_x

                # Draw Center of Mass
                cv2.circle(out, (cx, cy), 5, (255, 255, 255), -1)

            # --- METRIC 2: Heading Angle (for YAW) ---
            # Fits a mathematical straight line through the contour
            [vx, vy, x, y] = cv2.fitLine(best_contour, cv2.DIST_L2, 0, 0.01, 0.01)

            # Calculate angle in degrees relative to vertical (0 = straight ahead)
            if vy != 0:
                angle_rad = np.arctan2(vx, vy)
                angle = float(math.degrees(angle_rad[0]))

                # Draw the calculated trajectory line (Red Line)
                lefty = int((-x[0] * vy[0] / vx[0]) + y[0])
                righty = int(((w - x[0]) * vy[0] / vx[0]) + y[0])
                cv2.line(out, (w - 1, righty), (0, lefty), (0, 0, 255), 2)

            # Draw the contour outline and text
            cv2.drawContours(out, [best_contour], -1, (0, 255, 0), 2)
            cv2.putText(out, f"Offset: {offset}px", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(out, f"Angle: {angle:.1f} deg", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if cx is None:
        cv2.putText(out, "LINE LOST", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return out, mask, offset, angle


def run_vision(args):
    print("Initializing Pi 5 Hardware Camera (Picamera2)...")

    # Calculate height to maintain a 4:3 aspect ratio
    calc_height = int(args.width * 0.75)

    picam2 = Picamera2()

    try:
        # Request native BGR format directly from the ISP to save CPU cycles
        config = picam2.create_video_configuration(
            main={"size": (args.width, calc_height), "format": "BGR888"}
        )
        picam2.configure(config)
        picam2.start()
    except Exception as e:
        print(
            f"FATAL: Failed to start camera hardware. Ensure ribbon cable is seated and OS is configured.\nError: {e}")
        return

    print(f"Vision Active at {args.width}x{calc_height}. Outputting telemetry...")

    try:
        while True:
            start_time = time.time()

            # Grab frame directly from Pi 5 hardware buffer
            frame = picam2.capture_array()

            # FIX: Convert Picamera2's RGB array into OpenCV's native BGR format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Run CV algorithm
            annotated, mask, offset, angle = process_frame(frame, args)

            fps = 1.0 / (time.time() - start_time)

            # Console output (This is what you will parse and send to the flight controller)
            if offset is not None and angle is not None:
                print(f"[FPS: {fps:.1f}] Offset: {offset:4} | Angle: {angle:5.1f}")
            else:
                print(f"[FPS: {fps:.1f}] --- PATH LOST ---")

            # Show video windows (Turn off with --no-gui for massive FPS boost during real flight)
            if not args.no_gui:
                cv2.imshow("Drone Target View", annotated)
                cv2.imshow("Yellow Mask", mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nTerminating vision system...")
    finally:
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Line Follower Vision (Pi 5 Optimized)")
    parser.add_argument("--width", type=int, default=320, help="Hardware capture width (Keep small for high FPS)")
    parser.add_argument("--roi-top-ratio", type=float, default=0.3, help="Ignore top % of image (Horizon)")
    parser.add_argument("--contour-area", type=int, default=150,
                        help="Minimum pixel area to be considered a valid line")
    parser.add_argument("--no-gui", action="store_true", help="Disable video windows for maximum headless performance")

    args = parser.parse_args()
    run_vision(args)