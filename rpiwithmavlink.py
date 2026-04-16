import argparse
import cv2
import numpy as np
import time
import math
from picamera2 import Picamera2
from pymavlink import mavutil

# --- Proportional Controller Tuning Gains ---
# You will need to tune these values in the field or in a simulator
KP_OFFSET = 0.005  # Converts pixel offset to lateral speed (m/s)
KP_ANGLE = 0.015  # Converts degree angle to yaw rate (rad/s)
FORWARD_SPEED = 0.5  # Constant forward speed in m/s


def connect_to_drone(connection_string, baud):
    """Establishes MAVLink connection to the flight controller."""
    print(f"Connecting to flight controller on {connection_string} at {baud} baud...")
    vehicle = mavutil.mavlink_connection(connection_string, baud=baud)
    vehicle.wait_heartbeat()
    print(f"Connected! Target system: {vehicle.target_system}, component: {vehicle.target_component}")
    return vehicle


def send_body_velocity_command(vehicle, vx, vy, vz, yaw_rate):
    """
    Sends a velocity command relative to the drone's body frame.
    vx: Forward (+), Backward (-) [m/s]
    vy: Right (+), Left (-) [m/s]
    vz: Down (+), Up (-) [m/s]  <-- Note: Z is inverted in NED
    yaw_rate: Right (+), Left (-) [rad/s]
    """
    msg = vehicle.mav.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        vehicle.target_system, vehicle.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # Body frame: X is forward, Y is right
        0b0101111111000111,  # Bitmask to enable velocity and yaw_rate ONLY
        0, 0, 0,  # Position x, y, z (ignored by bitmask)
        vx, vy, vz,  # Velocity x, y, z in m/s
        0, 0, 0,  # Acceleration (ignored)
        0, yaw_rate  # Yaw (ignored), Yaw Rate in rad/s
    )
    vehicle.send(msg)


def process_frame(frame, args):
    # [Your exact process_frame function remains unchanged here]
    h, w = frame.shape[:2]
    center_x = w // 2
    out = frame.copy()

    cv2.line(out, (center_x, 0), (center_x, h - 1), (255, 0, 0), 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 60, 60])
    upper_yellow = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    top_crop = int(np.clip(args.roi_top_ratio, 0.0, 0.95) * h)
    if top_crop > 0:
        mask[:top_crop, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx, offset, angle = None, None, None

    if contours:
        best_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(best_contour) > args.contour_area:
            m = cv2.moments(best_contour)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                offset = cx - center_x
                cv2.circle(out, (cx, cy), 5, (255, 255, 255), -1)

            [vx_line, vy_line, x, y] = cv2.fitLine(best_contour, cv2.DIST_L2, 0, 0.01, 0.01)

            if vy_line != 0:
                angle_rad = np.arctan2(vx_line, vy_line)
                angle = float(math.degrees(angle_rad[0]))

                lefty = int((-x[0] * vy_line[0] / vx_line[0]) + y[0])
                righty = int(((w - x[0]) * vy_line[0] / vx_line[0]) + y[0])
                cv2.line(out, (w - 1, righty), (0, lefty), (0, 0, 255), 2)

            cv2.drawContours(out, [best_contour], -1, (0, 255, 0), 2)
            cv2.putText(out, f"Offset: {offset}px", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(out, f"Angle: {angle:.1f} deg", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if cx is None:
        cv2.putText(out, "LINE LOST", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return out, mask, offset, angle


def run_vision(args):
    # 1. Initialize MAVLink Connection
    vehicle = connect_to_drone(args.connect, args.baud)

    print("Initializing Pi 5 Hardware Camera (Picamera2)...")
    calc_height = int(args.width * 0.75)
    picam2 = Picamera2()

    try:
        config = picam2.create_video_configuration(main={"size": (args.width, calc_height), "format": "BGR888"})
        picam2.configure(config)
        picam2.start()
    except Exception as e:
        print(f"FATAL: Failed to start camera hardware.\nError: {e}")
        return

    print(f"Vision & MAVLink Active. Outputting telemetry...")

    try:
        while True:
            start_time = time.time()
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            annotated, mask, offset, angle = process_frame(frame, args)

            # --- MAVLINK CONTROL LOGIC ---
            if offset is not None and angle is not None:
                # Calculate required lateral speed and yaw rate using P-controllers
                vy = offset * KP_OFFSET
                yaw_rate = math.radians(angle) * KP_ANGLE  # MAVLink requires radians/sec

                # Cap maximum speeds for safety
                vy = np.clip(vy, -1.0, 1.0)
                yaw_rate = np.clip(yaw_rate, -1.0, 1.0)

                # Send command: Constant Forward (X), Variable Lateral (Y), Hold Altitude (Z=0), Variable Yaw
                send_body_velocity_command(vehicle, vx=FORWARD_SPEED, vy=vy, vz=0.0, yaw_rate=yaw_rate)

                print(f"[Line Found] Sending -> Vx:{FORWARD_SPEED:.2f} Vy:{vy:.2f} YawRate:{yaw_rate:.2f}")

            else:
                # LINE LOST FAILSAFE: Hover in place. Stop forward and lateral movement.
                send_body_velocity_command(vehicle, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)
                print(f"[LINE LOST] Hovering...")

            fps = 1.0 / (time.time() - start_time)

            if not args.no_gui:
                cv2.imshow("Drone Target View", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nTerminating vision system...")
    finally:
        # Send one last stop command before exiting
        send_body_velocity_command(vehicle, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Line Follower Vision & MAVLink")
    parser.add_argument("--width", type=int, default=320, help="Hardware capture width")
    parser.add_argument("--roi-top-ratio", type=float, default=0.3, help="Ignore top % of image")
    parser.add_argument("--contour-area", type=int, default=150, help="Min pixel area for a line")
    parser.add_argument("--no-gui", action="store_true", help="Disable video windows")

    # MAVLink specific arguments
    parser.add_argument("--connect", default="/dev/ttyAMA0",
                        help="MAVLink connection string (e.g. /dev/ttyAMA0 or udp:127.0.0.1:14550)")
    parser.add_argument("--baud", type=int, default=57600, help="MAVLink baud rate")

    args = parser.parse_args()
    run_vision(args)