import itertools
from enum import Enum
from typing import Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pypot import dynamixel
from deepface.detectors import FaceDetector

WEBCAM_ID = 0

TILT_ID = 1
TILT_RANGE = (-30, 30)
TILT_DEAD_ZONE = 30

PAN_ID = 2
PAN_RANGE = (-100, 100)
PAN_DEAD_ZONE = 20

TURN_DEGREES = 3
TURN_SPEED = 20


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


def get_webcam(video_source: int) -> cv2.VideoCapture:
    # Open stream
    cap = cv2.VideoCapture(video_source)

    # Check if stream is open
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    return cap


def detect_face(frame: np.ndarray, detector_backend='opencv') -> Optional[np.ndarray]:
    # Detect face
    try:
        face_detector = FaceDetector.build_model(detector_backend)
        faces = FaceDetector.detect_faces(face_detector, detector_backend, frame)
        return faces[0]

    # Face could not be detected
    except ValueError:
        return None

    # Face could not be detected
    except IndexError:
        return None


def get_center(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
    return x + w // 2, y + h // 2


def align_centers(face_center: Tuple[int, int], frame_center: Tuple[int, int]) -> Tuple[Optional[Direction], Optional[Direction]]:
    # Calculate difference
    x_diff = face_center[0] - frame_center[0]
    y_diff = face_center[1] - frame_center[1]

    # Add dead zone
    if abs(x_diff) < PAN_DEAD_ZONE:
        x_diff = 0
    if abs(y_diff) < TILT_DEAD_ZONE:
        y_diff = 0

    # Calculate direction
    x_dir = Direction.LEFT if x_diff < 0 else Direction.RIGHT if x_diff > 0 else None
    y_dir = Direction.UP if y_diff < 0 else Direction.DOWN if y_diff > 0 else None

    return x_dir, y_dir


def initialize_motors() -> dynamixel.DxlIO:
    ports = dynamixel.get_available_ports()
    assert len(ports) > 0, "No ports found"

    # Using first port
    port = ports[0]
    dxl_io = dynamixel.DxlIO(port)

    # Scan motors
    found_ids = dxl_io.scan()
    assert len(found_ids) > 0, "No motors found"
    assert PAN_ID in found_ids, f"Motor with ID {PAN_ID} not found"
    assert TILT_ID in found_ids, f"Motor with ID {TILT_ID} not found"

    # Print positions
    ids = [PAN_ID, TILT_ID]
    positions = dxl_io.get_present_position(ids)
    print(f"Found motors at positions {positions}")

    # Enable torque
    dxl_io.enable_torque(ids)

    # Set speed
    speed = dict(zip(ids, itertools.repeat(TURN_SPEED)))
    dxl_io.set_moving_speed(speed)

    # Set at center
    center = dict(zip(ids, itertools.repeat(0)))
    dxl_io.set_goal_position(center)

    return dxl_io


def move_motors(motors: dynamixel.DxlIO, x_dir: Optional[Direction], y_dir: Optional[Direction]):
    print(f"Moving motors in direction ({x_dir}, {y_dir})")

    current_pan_pos: Tuple[float] = motors.get_present_position([PAN_ID])
    if x_dir == Direction.LEFT:
        pan_pos = np.clip(current_pan_pos[0] + TURN_DEGREES, PAN_RANGE[0], PAN_RANGE[1])
        print(f"Pan pos: {pan_pos}")
        motors.set_goal_position({PAN_ID: pan_pos})
    elif x_dir == Direction.RIGHT:
        pan_pos = np.clip(current_pan_pos[0] - TURN_DEGREES, PAN_RANGE[0], PAN_RANGE[1])
        print(f"Pan pos: {pan_pos}")
        motors.set_goal_position({PAN_ID: pan_pos})

    current_tilt_pos: Tuple[float] = motors.get_present_position([TILT_ID])
    if y_dir == Direction.UP:
        tilt_pos = np.clip(current_tilt_pos[0] + TURN_DEGREES, TILT_RANGE[0], TILT_RANGE[1])
        print(f"Tilt pos: {tilt_pos}")
        motors.set_goal_position({TILT_ID: tilt_pos})
    elif y_dir == Direction.DOWN:
        tilt_pos = np.clip(current_tilt_pos[0] - TURN_DEGREES, TILT_RANGE[0], TILT_RANGE[1])
        print(f"Tilt pos: {tilt_pos}")
        motors.set_goal_position({TILT_ID: tilt_pos})


def main():
    print("Initializing...")

    webcam = get_webcam(WEBCAM_ID)
    motors = initialize_motors()

    while True:
        try:
            # Read frame
            ret, frame = webcam.read()
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)
            frame_w = frame.shape[1]
            frame_h = frame.shape[0]
            frame_center = get_center(0, 0, frame_w, frame_h)

            # Check if frame is empty
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Detect face
            detected_face = detect_face(frame)

            # Get bounding box
            if detected_face is not None:
                # Get bounding box center
                face, (x, y, w, h) = detected_face
                print(f"Face detected at ({x}, {y}) with size ({w}, {h})")
                face_center = get_center(x, y, w, h)

                # Align face
                x_dir, y_dir = align_centers(face_center, frame_center)
                move_motors(motors, x_dir, y_dir)

        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            break


if __name__ == '__main__':
    main()
