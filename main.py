import asyncio
import json
import time
import os
import re
import numpy as np
import math
import threading
from config import LOG_LEVEL
from services.logger import set_logger_config


def get_rotation_matrix(roll, pitch, yaw):
    # Roll — вокруг X
    Rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)],
        ]
    )

    # Pitch — вокруг Z
    Rz = np.array(
        [
            [math.cos(pitch), -math.sin(pitch), 0],
            [math.sin(pitch), math.cos(pitch), 0],
            [0, 0, 1],
        ]
    )

    # Yaw — вокруг Y
    Ry = np.array(
        [
            [math.cos(yaw), 0, math.sin(yaw)],
            [0, 1, 0],
            [-math.sin(yaw), 0, math.cos(yaw)],
        ]
    )

    return Ry @ Rz @ Rx  # сначала roll, потом pitch, потом yaw


sqrt2 = math.sqrt(2)
lidar_directions = {
    "f": np.array([0, 0, -1]),  # Forward
    "fr": np.array([-1, 0, -1]) / sqrt2,  # Forward-Right
    "r": np.array([-1, 0, 0]),  # Right
    "br": np.array([-1, 0, 1]) / sqrt2,  # Back-Right
    "b": np.array([0, 0, 1]),  # Back
    "bl": np.array([1, 0, 1]) / sqrt2,  # Back-Left
    "l": np.array([1, 0, 0]),  # Left
    "fl": np.array([1, 0, -1]) / sqrt2,  # Forward-Left
    "up": np.array([0, 1, 0]),  # Up
    "d": np.array([0, -1, 0]),  # Down
}


def get_obstacle_points(drone_pos, roll, pitch, yaw, lidar_info):
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    R = get_rotation_matrix(roll, pitch, yaw)

    obstacle_points = []
    for direction, distance in lidar_info.items():
        if 0.1 < distance < 10:
            local_dir = lidar_directions[direction]
            world_dir = R @ local_dir
            point = drone_pos + world_dir * distance
            obstacle_points.append((direction, point))
    return obstacle_points


def check_last_file_id(folder_path: str) -> int:
    existing_files = os.listdir(folder_path)
    # Фильтруем и извлекаем id из названий
    id_list = []
    pattern = re.compile(r"points_(\d+)\.xyz")
    for filename in existing_files:
        match = pattern.match(filename)
        if match:
            id_list.append(int(match.group(1)))
    # Определяем следующий id
    next_id = max(id_list) + 1 if id_list else 0  # Если файлов нет, начинаем с 0
    return next_id


async def start_websocket():
    from algorithm.fly_main import run_auto_navigation, connection
    from algorithm.PID import concat_engine, concat_engines

    await connection.set_connection()

    received_data = json.loads(connection.receive_data())
    fire_position = received_data["firesPositions"]
    for i in range(len(fire_position)):
        fire_position[i]["y"] += 3
    counter = 0
    points_data = []
    folder_path = os.getcwd() + "\\points_cloud"
    next_file_id = check_last_file_id(folder_path)
    main_drone_id = 0
    connection.send_data("sendData")
    drones_thread = threading.Thread(
        target=run_auto_navigation(fires_positions=fire_position), daemon=True
    )
    drones_thread.start()
    start_time = time.time()
    print("MapMaker started!")
    while True:
        counter += 1
        connection.send_data("sendData")

        received_data = json.loads(connection.receive_data())
        drone_pos = received_data["dronesData"][main_drone_id]["droneVector"]
        print(drone_pos)
        drone_pos = [drone_pos["x"], drone_pos["y"], drone_pos["z"]]
        drone_angles = received_data["dronesData"][main_drone_id]["droneAxisRotation"]
        # print("angles:",drone_angles)
        drone_speed = received_data["dronesData"][main_drone_id]["linearVelocity"]
        # print(drone_speed)
        end_time = time.time()
        execution_time = end_time - start_time
        start_time = end_time

        # print(f"Время выполнения: {execution_time} секунд")
        lidar_data = received_data["dronesData"][main_drone_id]["lidarInfo"]
        raw_points_data = get_obstacle_points(
            drone_pos,
            drone_angles["z"],
            drone_angles["x"],
            drone_angles["y"],
            lidar_data,
        )
        for point in raw_points_data:
            # print(point)
            points_data.append(point[1])

        time.sleep(0.1)
        if counter % 100 == 0:
            print(counter)
        if counter % 1000 == 0 and counter != 0:

            print(f"---Saved-{next_file_id}---")
            np.savetxt(
                f"{folder_path}\\points_{next_file_id}.xyz",
                points_data,
                fmt="%.4f",
                delimiter=" ",
            )
            next_file_id += 1
            points_data = []

    # connection.send_data("restartScene")
    # connection.send_data(concat_engines(concat_engine([0 for _ in range(8)], {"id": 0}), 0))

    connection.close_connection()


if __name__ == "__main__":
    set_logger_config(LOG_LEVEL)
    asyncio.run(start_websocket())
