import time
import json
import math
import numpy as np

from connection.SocketConnection import SocketConnection
from pathfinder import simplify_path
import itertools

# Конфигурация
connection = SocketConnection()
T = 0.1  # Время между пакетами данных симулятора
BASE_POWER = 50  # Базовая мощность для всех моторов
POWER_STEP = 5  # Шаг изменения мощности при управлении
WAYPOINT_THRESHOLD = 0.35  # Расстояние, при котором считаем, что достигли точки (м)
PID_SAMPLE_TIME = 0.1  # Время обновления PID регуляторов
MAX_VELOCITY = 20.0  # Максимальная скорость движения дрона (м/с)
HOVER_POWER = 50  # Мощность моторов для удержания позиции
MAX_ANGLE = 25  # Максимальный угол крена/тангажа (в градусах)


# Количество дронов
NUM_DRONES = 5

# Глобальные переменные для хранения состояния дронов
drone_states = {}
motor_powers = {}
mission_status = {}
current_waypoint_indices = {}

# Инициализация состояний для всех дронов
for drone_id in range(NUM_DRONES):
    # Состояние дрона
    drone_states[drone_id] = {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rot_velocity": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "last_update_time": 0.0,
        "initial_yaw": None,  # Запомним начальную ориентацию
    }

    # Мощность моторов для каждого дрона
    motor_powers[drone_id] = {
        "fr": BASE_POWER,  # Front right
        "fl": BASE_POWER,  # Front left
        "br": BASE_POWER,  # Back right
        "bl": BASE_POWER,  # Back left
        "rf": BASE_POWER,  # Right front
        "rb": BASE_POWER,  # Right back
        "lf": BASE_POWER,  # Left front
        "lb": BASE_POWER,  # Left back
    }

    # Статус миссии
    mission_status[drone_id] = False
    current_waypoint_indices[drone_id] = 0


# PID контроллер
class PIDController:
    def __init__(self, kp, ki, kd, sample_time):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.sample_time = sample_time
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time

        if dt >= self.sample_time:
            # Пропорциональная составляющая
            p_term = self.kp * error

            # Интегральная составляющая с ограничением накопления
            self.integral += error * dt
            self.integral = max(-20, min(20, self.integral))  # Ограничиваем интеграл
            i_term = self.ki * self.integral

            # Дифференциальная составляющая
            d_term = 0
            if dt > 0:
                d_term = self.kd * (error - self.last_error) / dt

            # Обновляем последние значения
            self.last_error = error
            self.last_time = current_time

            # Суммарный выход PID
            output = p_term + i_term + d_term
            return output

        return None

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


# Параметры PID регуляторов
PID_PARAMS = {
    "thrust": {"P": 8.0, "I": 0.1, "D": 5.5},  # для управления тягой
    "attitude": {
        "P": 1.1,
        "I": 0.03,
        "D": 1.65,
    },  # для управления ориентацией (pitch/roll)
}

# PID контроллеры для каждого дрона
pid_controllers = {}

for drone_id in range(NUM_DRONES):
    pid_controllers[drone_id] = {
        # Новые контроллеры для непосредственного управления
        "thrust_pid": PIDController(
            PID_PARAMS["thrust"]["P"],
            PID_PARAMS["thrust"]["I"],
            PID_PARAMS["thrust"]["D"],
            PID_SAMPLE_TIME,
        ),
        "pitch_pid": PIDController(
            PID_PARAMS["attitude"]["P"],
            PID_PARAMS["attitude"]["I"],
            PID_PARAMS["attitude"]["D"],
            PID_SAMPLE_TIME,
        ),
        "roll_pid": PIDController(
            PID_PARAMS["attitude"]["P"],
            PID_PARAMS["attitude"]["I"],
            PID_PARAMS["attitude"]["D"],
            PID_SAMPLE_TIME,
        ),
    }

waypoints = {0: [], 1: [], 2: [], 3: [], 4: []}
# Add these imports at the top if not present
from collections import defaultdict

# Global variables for sorted fires and path assignment
sorted_fires = []
drone_fire_assignments = {0: [], 1: [], 2: [], 3: [], 4: []}
all_fire_paths = defaultdict(dict)  # {drone_id: {fire_id: path}}


def initialize_fire_assignments(fires_positions, start_point):
    global drone_fire_assignments
    """
    Для каждой из трёх групп (ближние, средние, дальние) делает полный перебор
    соответствий дронов и пожаров и выбирает минимальную по суммарному расстоянию схему.
    Возвращает словарь drone_fire_assignments: drone_id → [fire_id_near, fire_id_med, fire_id_far].
    """
    # 1) считаем расстояния от старта до каждого пожара
    fire_distances = []
    for i, fire in enumerate(fires_positions):
        dx = fire["x"] - start_point["x"]
        dy = fire["y"] - start_point["y"]
        dz = fire["z"] - start_point["z"]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        fire_distances.append((i, d))
    # сортируем по росту расстояния
    fire_distances.sort(key=lambda x: x[1])
    sorted_ids = [fid for fid, _ in fire_distances]

    # 2) разбиваем на три группы
    N = len(sorted_ids)
    third = N // 3
    near_ids = sorted_ids[:third]
    medium_ids = sorted_ids[third : 2 * third]
    far_ids = sorted_ids[2 * third :]

    groups = [near_ids, medium_ids, far_ids]
    # lookup-distance по id пожара
    dist_lookup = {fid: d for fid, d in fire_distances}

    # 3) подготовим итоговые назначения
    assignments = {dr: [] for dr in range(NUM_DRONES)}

    # 4) для каждой группы делаем полный перебор
    for group in groups:
        best_cost = float("inf")
        best_perm = None

        # рассматриваем все ways назначить len(group) пожаров на NUM_DRONES дронов
        # если дронов больше, чем пожаров, лишние дроны просто ничего не берут в этой группе
        for perm in itertools.permutations(group, min(len(group), NUM_DRONES)):
            # perm[j] — id пожара, назначенный дрону j
            cost = sum(dist_lookup[perm[j]] for j in range(len(perm)))
            if cost < best_cost:
                best_cost = cost
                best_perm = perm

        # закрепляем best_perm за дронами
        for j, fire_id in enumerate(best_perm):
            assignments[j].append(fire_id)

    drone_fire_assignments = assignments

    for drone_id, fire_ids in drone_fire_assignments.items():
        print(f"Drone {drone_id} assigned fires: {fire_ids}")
    # Создаем словарь для быстрого доступа к расстоянию по fire_id
    fire_distance_lookup = {fire_id: distance for fire_id, distance in fire_distances}
    # Печатаем суммарные расстояния до пожаров для каждого дрона
    for drone_id, fire_ids in drone_fire_assignments.items():
        total_distance = sum(fire_distance_lookup[fire_id] for fire_id in fire_ids)
        print(f"Drone {drone_id} total distance to fires: {total_distance:.2f}")
    return drone_fire_assignments


def get_data(str_data: str):
    """Получение данных от симулятора"""
    data = json.loads(str_data)["dronesData"]
    return data


def concat_engines(engines, t):
    """Форматирование данных для симулятора"""
    result = {
        "drones": engines,
        "returnTimer": 1000 * t,
    }
    return json.dumps(result)


def create_drone_data(drone_id):
    """Создание данных управления дроном на основе текущей мощности моторов"""
    global motor_powers
    return {
        "id": drone_id,
        "engines": motor_powers[drone_id],
        "dropExtinguisher": drone_states[drone_id]["dropExtinguisher"],
    }


def update_drone_state(drone_id, drone_data):
    """Обновление состояния дрона на основе полученных данных"""
    global drone_states

    current_time = time.time()

    # Обновляем положение и ориентацию
    drone_states[drone_id]["position"]["x"] = drone_data["droneVector"]["x"]
    drone_states[drone_id]["position"]["y"] = drone_data["droneVector"]["y"]
    drone_states[drone_id]["position"]["z"] = drone_data["droneVector"]["z"]
    drone_states[drone_id]["rotation"]["yaw"] = drone_data["droneAxisRotation"]["y"]
    drone_states[drone_id]["rotation"]["pitch"] = drone_data["droneAxisRotation"]["x"]
    drone_states[drone_id]["rotation"]["roll"] = drone_data["droneAxisRotation"]["z"]

    # Прямое обновление скоростей из данных симулятора
    drone_states[drone_id]["velocity"]["x"] = drone_data["linearVelocity"]["x"]
    drone_states[drone_id]["velocity"]["y"] = drone_data["linearVelocity"]["y"]
    drone_states[drone_id]["velocity"]["z"] = drone_data["linearVelocity"]["z"]
    drone_states[drone_id]["rot_velocity"]["yaw"] = drone_data["angularVelocity"]["y"]
    drone_states[drone_id]["rot_velocity"]["pitch"] = drone_data["angularVelocity"]["x"]
    drone_states[drone_id]["rot_velocity"]["roll"] = drone_data["angularVelocity"]["z"]

    # Запоминаем начальную ориентацию (yaw), если еще не сохранили
    if drone_states[drone_id]["initial_yaw"] is None:
        drone_states[drone_id]["initial_yaw"] = drone_states[drone_id]["rotation"][
            "yaw"
        ]
        print(f"Drone {drone_id} initial yaw: {drone_states[drone_id]['initial_yaw']}")

    # Обновляем время последнего обновления
    drone_states[drone_id]["last_update_time"] = current_time
    drone_states[drone_id]["dropExtinguisher"] = False


def calculate_distance(point1, point2):
    """Рассчет расстояния между двумя точками"""
    return math.sqrt(
        (point1["x"] - point2["x"]) ** 2
        + (point1["y"] - point2["y"]) ** 2
        + (point1["z"] - point2["z"]) ** 2
    )


def calculate_direction_vector(current_pos, target_pos):
    """Расчет единичного вектора направления к цели"""
    dx = target_pos["x"] - current_pos["x"]
    dy = target_pos["y"] - current_pos["y"]
    dz = target_pos["z"] - current_pos["z"]

    # Нормализация вектора
    magnitude = math.sqrt(dx**2 + dy**2 + dz**2)
    if magnitude > 0:
        dx /= magnitude
        dy /= magnitude
        dz /= magnitude

    return {"x": dx, "y": dy, "z": dz}


def compute_hover_power(
    roll_deg: float, pitch_deg: float, hover_power: float = HOVER_POWER
) -> float:
    """
    Рассчитывает требуемую мощность моторов (0–100) для удержания высоты
    при заданных углах наклона roll и pitch (в градусах).
    """
    # Преобразуем градусы в радианы
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)

    # Вертикальная компонента тяги пропорциональна cos(roll) * cos(pitch)
    tilt_factor = math.cos(roll_rad) * math.cos(pitch_rad)

    # Предотвращаем деление на очень маленькие значения
    if tilt_factor < 0.1:  # Ограничиваем до 10% для предотвращения чрезмерной мощности
        tilt_factor = 0.1

    # Рассчитываем необходимую мощность
    required_power = hover_power / tilt_factor

    # Ограничиваем значение в допустимом диапазоне (0-100)
    return min(100.0, max(0.0, required_power))


def navigate_to_waypoint(drone_id, current_waypoint):
    """
    Логика управления дроном с использованием PID-регуляторов.
    Использует двухуровневую систему управления:
    1. PID для достижения целевой координаты
    2. Управление углами для достижения этой скорости
    Args:
        drone_id: ID дрона
        current_waypoint: Целевая точка {x, y, z}
    """
    global drone_states, pid_controllers, original_distances

    # Инициализация словарей для отслеживания данных, если их нет
    if "original_distances" not in globals():
        original_distances = {}

    # Структура для хранения расстояний для каждого дрона и точки
    drone_key = f"drone_{drone_id}"
    waypoint_key = f"waypoint_{current_waypoint_indices[drone_id]}"

    if drone_key not in original_distances:
        original_distances[drone_key] = {}

    # Текущее состояние дрона
    pos = drone_states[drone_id]["position"]
    vel = drone_states[drone_id]["velocity"]
    rot = drone_states[drone_id]["rotation"]

    # Расчет разницы между текущей позицией и целью
    dx = current_waypoint["x"] - pos["x"]
    dy = current_waypoint["y"] - pos["y"]  # Вертикальная разница
    dz = current_waypoint["z"] - pos["z"]

    # Расчет расстояний
    horizontal_dist = math.sqrt(dx**2 + dz**2)
    height_diff = dy
    dist_total = math.sqrt(dx**2 + dy**2 + dz**2)

    # Начальные значения команд управления
    thrust_cmd = 0
    pitch_cmd = 0
    roll_cmd = 0

    # Используем PID для вертикального движения
    # Если PID не вернул значение, используем базовое значение
    max_speed = 5
    max_dy = 1.8
    # print(vel['z'])
    rvx = max(-max_speed, min(max_speed, (dx - vel["x"])))
    rvz = max(-max_speed, min(max_speed, (dz - vel["z"])))
    dvx = rvx - vel["x"]
    dvz = rvz - vel["z"]

    # if dist_total < 0.30:
    #     dvx = -vel["x"]
    #     dvz = -vel["z"]
    dy = max(-max_dy, min(max_dy, dy))
    thrust_cmd = pid_controllers[drone_id]["thrust_pid"].update(dy)
    roll_cmd = pid_controllers[drone_id]["roll_pid"].update(-dvx)
    pitch_cmd = pid_controllers[drone_id]["pitch_pid"].update(dvz)

    if thrust_cmd is None:
        thrust_cmd = 0
    if roll_cmd is None:
        roll_cmd = 0
    if pitch_cmd is None:
        pitch_cmd = 0

    # Рассчитываем требуемую мощность для зависания с учетом текущих углов
    hover_power = compute_hover_power(rot["roll"], rot["pitch"])

    # Применяем команды к моторам
    # Рассчитываем мощность для каждого мотора на основе базовой мощности и корректировок
    base_power = hover_power + thrust_cmd

    # Применяем команды к моторам
    # Рассчитываем мощность для каждого мотора на основе базовой мощности и корректировок
    # thrust_cmd = max(-10,min(10,thrust_cmd))
    base_power = hover_power + thrust_cmd

    # Применяем команды для каждого мотора
    motor_powers[drone_id]["fr"] = base_power + pitch_cmd - roll_cmd
    motor_powers[drone_id]["fl"] = base_power + pitch_cmd + roll_cmd
    motor_powers[drone_id]["br"] = base_power - pitch_cmd - roll_cmd
    motor_powers[drone_id]["bl"] = base_power - pitch_cmd + roll_cmd
    motor_powers[drone_id]["rf"] = base_power + pitch_cmd - roll_cmd
    motor_powers[drone_id]["rb"] = base_power - pitch_cmd - roll_cmd
    motor_powers[drone_id]["lf"] = base_power + pitch_cmd + roll_cmd
    motor_powers[drone_id]["lb"] = base_power - pitch_cmd + roll_cmd

    # Ограничиваем мощность моторов в пределах [0, 100]
    for motor in motor_powers[drone_id]:
        motor_powers[drone_id][motor] = max(
            0.0, min(100.0, motor_powers[drone_id][motor])
        )

    # Логирование информации
    if time.time() % 5 < 0.1:
        print(f"Dist={horizontal_dist:.2f}, TotalDist={dist_total:.2f}")
        print(f"Current: X={vel['x']:.2f}, Z={vel['z']:.2f}")
        print(
            f"dY={dy:.2f}, Pitch→{pitch_cmd:.1f}, Roll→{roll_cmd:.1f}, Thrust→{thrust_cmd:.1f}"
        )

    return {
        "distance": {
            "total": dist_total,
            "horizontal": horizontal_dist,
            "vertical": height_diff,
        },
        "control": {"thrust": thrust_cmd, "pitch": pitch_cmd, "roll": roll_cmd},
    }


def check_waypoint_reached(drone_id):
    """Проверка, достигнута ли текущая путевая точка"""
    global current_waypoint_indices, mission_status, waypoints, drone_states

    if current_waypoint_indices[drone_id] >= len(waypoints[drone_id]):
        mission_status[drone_id] = True
        return

    # Проверяем расстояние до текущей точки
    current_waypoint = waypoints[drone_id][current_waypoint_indices[drone_id]]
    distance = calculate_distance(drone_states[drone_id]["position"], current_waypoint)
    v_x = drone_states[drone_id]["velocity"]["x"]
    v_y = drone_states[drone_id]["velocity"]["y"]
    v_z = drone_states[drone_id]["velocity"]["z"]
    speed = math.sqrt(v_x**2 + v_y**2 + v_z**2)
    roll = drone_states[drone_id]["rotation"]["roll"]
    yaw = drone_states[drone_id]["rotation"]["yaw"]
    pitch = drone_states[drone_id]["rotation"]["pitch"]
    max_angle = max(roll, pitch, yaw)
    if distance < WAYPOINT_THRESHOLD and speed < 0.5 and max_angle < 20:
        print(
            f"Drone {drone_id}: Waypoint {current_waypoint_indices[drone_id] + 1} reached! Distance: {distance:.2f}m"
        )
        current_waypoint_indices[drone_id] += 1

        # Сбрасываем накопленную интегральную ошибку в PID регуляторах
        # for controller in pid_controllers[drone_id].values():
        #    controller.reset()

        if current_waypoint_indices[drone_id] >= len(waypoints[drone_id]):
            print(f"Drone {drone_id}: Mission complete! All waypoints reached.")
            mission_status[drone_id] = True


def near_fire(px, pz, spx, spz):
    dx = px - spx
    dz = pz - spz
    distance = math.sqrt(dx * dx + dz * dz)
    return distance < WAYPOINT_THRESHOLD * 4.5


def near_start_point(point, second_point):
    px = point["x"]
    py = point["y"]
    pz = point["z"]
    spx = second_point["x"]
    spy = second_point["y"]
    spz = second_point["z"]
    dx = px - spx
    dy = py - spy
    dz = pz - spz
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    return distance < WAYPOINT_THRESHOLD * 3


def schedule_all_paths(fires_positions, path_finder, min_distance, max_distance):
    """
    Schedule path calculations for all drones and all their assigned fires
    """
    global drone_fire_assignments, drones_start_points, waiting_points

    print("Scheduling all path calculations...")

    # For each drone and all its assigned fires
    drone_fire_ass = []
    for drone_id, fire_ids in drone_fire_assignments.items():
        for fire_id in fire_ids:
            drone_fire_ass.append([drone_id, fire_id])
    l = len(drone_fire_ass)
    for i in range(l // NUM_DRONES):
        for j in range(NUM_DRONES):
            drone_id = drone_fire_ass[j * l // NUM_DRONES + i][0]
            fire_id = drone_fire_ass[j * l // NUM_DRONES + i][1]
            start_point = (
                -waiting_points[drone_id]["x"],
                waiting_points[drone_id]["y"],
                waiting_points[drone_id]["z"],
            )
            fire_coord = (
                -fires_positions[fire_id]["x"],
                fires_positions[fire_id]["y"],
                fires_positions[fire_id]["z"],
            )
            # Add task to path calculation queue
            path_calculation_queue.put((drone_id, fire_id, start_point, fire_coord))
            print(f"Scheduled path calculation for drone {drone_id} to fire #{fire_id}")


# Глобальные переменные для управления миссиями
fires_ind = 0
drones_start_points = {
    0: {"x": -1111, "y": -1, "z": -1},
    1: {"x": -1111, "y": -1, "z": -1},
    2: {"x": -1111, "y": -1, "z": -1},
    3: {"x": -1111, "y": -1, "z": -1},
    4: {"x": -1111, "y": -1, "z": -1},
}

import threading
import queue
import time
from collections import deque

# Глобальные структуры данных для хранения состояний дронов
drone_mission_phases = {0: "idle", 1: "idle", 2: "idle", 3: "idle", 4: "idle"}
# Фазы: "idle" -> "waiting" -> "to_fire" -> "at_fire" -> "to_start" -> "idle"
assigned_fires = {0: None, 1: None, 2: None, 3: None, 4: None}
extinguisher_dropped = {0: False, 1: False, 2: False, 3: False, 4: False}

# Очереди для взаимодействия между потоками
path_calculation_queue = queue.Queue()  # Очередь запросов на расчёт путей
ready_paths_queue = queue.Queue()  # Очередь с готовыми путями

# Словарь для отслеживания, какому дрону какой пожар назначен следующим
next_fire_assignments = {}


# Функция для расчёта путей в отдельном потоке
# Updated path calculation worker
def path_calculation_worker(path_finder, min_distance, max_distance):
    """Function running in a separate thread for path calculation"""
    while True:
        try:
            # Get task data from queue
            task = path_calculation_queue.get()

            if task is None:  # Signal to terminate thread
                break

            drone_id, fire_id, start_point, fire_coord = task

            print(
                f"[PathWorker] Calculating path for drone {drone_id} to fire {fire_id}"
            )

            # Calculate path
            list_waypoints = path_finder.find_path_with_min_turns(
                start_point,
                fire_coord,
                drone_id=drone_id,
                min_distance=min_distance,
                max_distance=max_distance,
            )

            # Simplify path
            simplified_path = simplify_path(list_waypoints)

            # Format waypoints
            formatted_waypoints = []
            for j in range(len(simplified_path)):
                formatted_waypoints.append(
                    {
                        "x": -simplified_path[j][0],
                        "y": simplified_path[j][1],
                        "z": simplified_path[j][2],
                    }
                )

            # Store the path in the global dictionary
            all_fire_paths[drone_id][fire_id] = formatted_waypoints

            # Also put in ready queue if a drone is actively waiting for this path
            # if drone_mission_data[drone_id]["phase"] == "waiting" :
            ready_paths_queue.put((drone_id, fire_id, formatted_waypoints))

            print(
                f"[PathWorker] Path for drone {drone_id} to fire {fire_id} calculated, {len(formatted_waypoints)} waypoints"
            )

            # Mark task as done
            path_calculation_queue.task_done()

        except Exception as e:
            print(f"[PathWorker] Error in path calculation: {e}")
            path_calculation_queue.task_done()


# Инициализация потока расчёта путей
def init_path_calculation_thread(path_finder, min_distance, max_distance):
    path_thread = threading.Thread(
        target=path_calculation_worker,
        args=(path_finder, min_distance, max_distance),
        daemon=True,
    )
    path_thread.start()
    return path_thread


# Добавляем точки ожидания для каждого дрона
waiting_points = {
    0: {"x": -77, "y": 8, "z": 75},
    1: {"x": -73, "y": 10, "z": 79},
    2: {"x": -73, "y": 8, "z": 71},
    3: {"x": -81, "y": 10, "z": 71},
    4: {"x": -81, "y": 6, "z": 79},
}

# Обновляем структуру данных для отслеживания миссий
drone_mission_data = {
    drone_id: {
        "phase": "idle",  # "idle" -> "to_waiting" -> "waiting" -> "to_fire" -> "at_fire" -> "to_waiting" -> "to_start" -> "idle"
        "current_fire_index": 0,  # Index in the drone's fire assignment list
        "assigned_fire": None,  # Current fire ID
        "next_fire": None,  # Next fire ID
        "extinguisher_dropped": False,
        "return_to_waiting": False,  # Флаг для определения направления (после тушения пожара -> ожидание)
    }
    for drone_id in range(NUM_DRONES)
}


# Функция для генерации пути к точке ожидания
def generate_waiting_point_path(drone_id, start_point):
    """Generate path from current position to waiting point"""
    waiting_point = waiting_points[drone_id]
    # Простой путь - одна точка (место ожидания)
    return [{"x": waiting_point["x"], "y": waiting_point["y"], "z": waiting_point["z"]}]


def assign_path_to_drone(drone_id, fire_id):
    """Assign a calculated path to a drone if it's ready"""
    global mission_status, drone_mission_data, waypoints, current_waypoint_indices

    mission_data = drone_mission_data.get(drone_id)
    if (
        not mission_data
        or mission_data["phase"] != "waiting"
        or mission_data["next_fire"] != fire_id
    ):
        return False

    # Получаем сохраненный путь
    calculated_waypoints = drone_calculated_paths[drone_id].get(fire_id)
    if not calculated_waypoints:
        return False

    # Назначаем путь дрону
    waypoints[drone_id].clear()
    waypoints[drone_id].extend(calculated_waypoints)

    # Update mission data
    mission_data["assigned_fire"] = fire_id
    mission_data["phase"] = "to_fire"
    mission_data["extinguisher_dropped"] = False

    # Update mission status
    mission_status[drone_id] = True
    current_waypoint_indices[drone_id] = 0

    print(f"Drone {drone_id}: Starting mission to fire #{fire_id}, path received")

    # Удаляем использованный путь из сохраненных
    del drone_calculated_paths[drone_id][fire_id]

    return True


def no_other_drones_nearby(drone_id, drone_states, radius):
    x0 = drone_states[drone_id]["position"]["x"]
    z0 = drone_states[drone_id]["position"]["z"]
    for other_id, st in drone_states.items():
        if other_id == drone_id:
            continue
        dx = st["position"]["x"] - x0
        dz = st["position"]["z"] - z0
        if math.sqrt(dx * dx + dz * dz) < radius:
            return False
    return True


drone_calculated_paths = {}


# Модифицированная функция next_step
def next_step(fires_positions, path_finder, min_distance, max_distance):
    """Process one step of drone movement"""
    global mission_status, drone_mission_data, all_fire_paths, drone_fire_assignments, drone_calculated_paths
    data = get_data(connection.receive_data())
    result = []

    # Check for ready paths
    while not ready_paths_queue.empty():
        drone_id, fire_id, calculated_waypoints = ready_paths_queue.get()

        # Сохраняем рассчитанный путь для дрона
        if drone_id not in drone_calculated_paths:
            drone_calculated_paths[drone_id] = {}
        drone_calculated_paths[drone_id][fire_id] = calculated_waypoints

        # Пробуем назначить путь, если дрон готов
        assign_path_to_drone(drone_id, fire_id)

    # Проверяем всех дронов, для которых есть сохраненные пути
    for drone_id in drone_calculated_paths:
        mission_data = drone_mission_data.get(drone_id)
        if (
            mission_data
            and mission_data["phase"] == "waiting"
            and mission_data["next_fire"] is not None
        ):
            fire_id = mission_data["next_fire"]
            if fire_id in drone_calculated_paths[drone_id]:
                assign_path_to_drone(drone_id, fire_id)
    # Process each drone
    for drone_data in data:
        drone_id = drone_data["id"]

        # Skip drones outside our range
        if drone_id < 0 or drone_id >= NUM_DRONES:
            continue

        # Update drone state
        update_drone_state(drone_id, drone_data)

        # Get mission data for this drone
        mission_data = drone_mission_data[drone_id]

        # Handle drone based on its current phase
        if mission_data["phase"] == "idle":
            # Drone is idle, check if there are more fires assigned to it
            fire_assignments = drone_fire_assignments[drone_id]
            if mission_data["current_fire_index"] < len(fire_assignments):
                # Get next fire ID
                next_fire_id = fire_assignments[mission_data["current_fire_index"]]
                mission_data["next_fire"] = next_fire_id

                # Сначала летим в точку ожидания
                waiting_path = generate_waiting_point_path(
                    drone_id, drone_states[drone_id]["position"]
                )
                waypoints[drone_id].clear()
                waypoints[drone_id].extend(waiting_path)

                # Обновляем фазу
                mission_data["phase"] = "to_waiting"
                mission_status[drone_id] = True
                current_waypoint_indices[drone_id] = 0

                print(
                    f"Drone {drone_id}: Moving to waiting point before fire #{next_fire_id}"
                )
            else:
                # No more fires assigned, remain idle
                for motor in motor_powers[drone_id]:
                    motor_powers[drone_id][motor] = 0

        elif mission_data["phase"] == "to_waiting":
            # Drone is flying to waiting point
            # Check if current waypoint is reached
            check_waypoint_reached(drone_id)

            if current_waypoint_indices[drone_id] >= len(waypoints[drone_id]):
                # Reached waiting point
                if mission_data["return_to_waiting"]:
                    # После тушения пожара - летим обратно на базу
                    mission_data["return_to_waiting"] = False

                    # Генерируем путь от ожидания к старту
                    start_path = [
                        {
                            "x": drones_start_points[drone_id]["x"],
                            "y": drones_start_points[drone_id]["y"],
                            "z": drones_start_points[drone_id]["z"],
                        }
                    ]

                    waypoints[drone_id].clear()
                    waypoints[drone_id].extend(start_path)
                    mission_data["phase"] = "to_start"
                    current_waypoint_indices[drone_id] = 0
                    print(f"Drone {drone_id}: Moving from waiting point to base")
                else:
                    # Перед тушением - проверяем наличие пути к пожару
                    next_fire_id = mission_data["next_fire"]
                    if next_fire_id in all_fire_paths[drone_id]:
                        # Path is ready, get it
                        calculated_path = all_fire_paths[drone_id][next_fire_id]
                        waypoints[drone_id].clear()
                        waypoints[drone_id].extend(calculated_path)

                        # Update mission data
                        mission_data["assigned_fire"] = next_fire_id
                        mission_data["phase"] = "to_fire"
                        mission_data["extinguisher_dropped"] = False

                        # Update mission status
                        mission_status[drone_id] = True
                        current_waypoint_indices[drone_id] = 0

                        print(
                            f"Drone {drone_id}: Starting mission to fire #{next_fire_id} from waiting point"
                        )
                    else:
                        # Path not ready, switch to waiting phase
                        mission_data["phase"] = "waiting"
                        print(
                            f"Drone {drone_id}: At waiting point, waiting for path calculation to fire #{next_fire_id}"
                        )
            else:
                # Navigate to current waypoint
                current_waypoint = waypoints[drone_id][
                    current_waypoint_indices[drone_id]
                ]
                navigate_to_waypoint(drone_id, current_waypoint)

        elif mission_data["phase"] == "waiting":
            # Drone is waiting for path calculation, keep it hovering at waiting point
            hover_point = waiting_points[drone_id]
            navigate_to_waypoint(drone_id, hover_point)

        elif mission_data["phase"] == "to_fire":
            # Drone is flying to fire
            # Check if current waypoint is reached
            check_waypoint_reached(drone_id)

            if current_waypoint_indices[drone_id] >= len(waypoints[drone_id]):
                # Reached final waypoint
                mission_data["phase"] = "at_fire"
                print(
                    f"Drone {drone_id}: Reached fire #{mission_data['assigned_fire']}"
                )
            else:
                # Navigate to current waypoint
                current_waypoint = waypoints[drone_id][
                    current_waypoint_indices[drone_id]
                ]
                nav_info = navigate_to_waypoint(drone_id, current_waypoint)

                # Check if near fire (for dropping extinguisher)
                if (
                    not mission_data["extinguisher_dropped"]
                    and mission_data["assigned_fire"] is not None
                ):
                    fire_x = fires_positions[mission_data["assigned_fire"]]["x"]
                    fire_z = fires_positions[mission_data["assigned_fire"]]["z"]
                    drone_x = drone_states[drone_id]["position"]["x"]
                    drone_z = drone_states[drone_id]["position"]["z"]

                    if near_fire(drone_x, drone_z, fire_x, fire_z):
                        # Drop extinguisher
                        vx = drone_states[drone_id]["velocity"]["x"]
                        vy = drone_states[drone_id]["velocity"]["y"]
                        vz = drone_states[drone_id]["velocity"]["z"]
                        v = math.sqrt(vx**2 + vy**2 + vz**2)
                        if v < 0.8 and no_other_drones_nearby(
                            drone_id, drone_states, 0.2
                        ):
                            drone_states[drone_id]["dropExtinguisher"] = True
                            mission_data["extinguisher_dropped"] = True
                            mission_data["phase"] = "at_fire"
                            print(
                                f"Drone {drone_id}: Dropped extinguisher at fire #{mission_data['assigned_fire']}"
                            )

                # Log information approximately every 2 seconds
                if time.time() % 10 < 0.1:
                    print(
                        f"Drone {drone_id}: Flying to waypoint {current_waypoint_indices[drone_id] + 1}/{len(waypoints[drone_id])}"
                    )
                    if nav_info and "distance" in nav_info:
                        print(f"  Distance: {nav_info['distance']['total']:.2f}m")
                    print(
                        f"  Point: x={current_waypoint['x']:.2f}m, y={current_waypoint['y']:.2f}, z={current_waypoint['z']:.2f}"
                    )
                    print(
                        f"  Position: x={drone_states[drone_id]['position']['x']:.2f}, y={drone_states[drone_id]['position']['y']:.2f}, z={drone_states[drone_id]['position']['z']:.2f}"
                    )

        elif mission_data["phase"] == "at_fire":
            # Drone is at fire, should return to waiting point
            mission_data["return_to_waiting"] = True

            # Генерируем путь к точке ожидания
            temp = []
            for i in range(len(waypoints[drone_id])):
                temp.append(waypoints[drone_id][len(waypoints[drone_id]) - i - 1])
            waypoints[drone_id] = temp

            mission_data["phase"] = "to_waiting"
            current_waypoint_indices[drone_id] = 0
            print(
                f"Drone {drone_id}: Moving back to waiting point after fire #{mission_data['assigned_fire']}"
            )

        elif mission_data["phase"] == "to_start":
            # Drone is flying back to base
            check_waypoint_reached(drone_id)

            if current_waypoint_indices[drone_id] >= len(waypoints[drone_id]):
                # Returned to base, mission complete
                mission_data["phase"] = "idle"
                mission_status[drone_id] = False
                mission_data["current_fire_index"] += 1  # Move to next fire
                print(f"Drone {drone_id}: Returned to base, mission completed")
            else:
                # Navigate to current waypoint
                current_waypoint = waypoints[drone_id][
                    current_waypoint_indices[drone_id]
                ]
                navigate_to_waypoint(drone_id, current_waypoint)

        # Create and add control data for drone
        new_data = create_drone_data(drone_id)
        result.append(new_data)

    # Send data to simulator
    connection.send_data(concat_engines(result, T))
    time.sleep(T)


# Функция для инициализации системы
def init_system(path_finder, min_distance, max_distance):
    # Запускаем поток расчёта путей
    path_thread = init_path_calculation_thread(path_finder, min_distance, max_distance)
    return path_thread


# Функция для завершения работы системы
def shutdown_system(path_thread):
    # Отправляем сигнал для завершения потока расчёта путей
    path_calculation_queue.put(None)
    # Ожидаем завершения потока
    path_thread.join(timeout=2.0)
    print("Path calculation thread stopped")


def run_auto_navigation(waypoints_file=None, fires_positions=None):
    import pathfinder

    """Main function to start automatic navigation by points"""
    global waypoints, current_waypoint_indices, mission_status, drone_mission_data

    # Fix fire position if needed
    for item in fires_positions:
        if int(item["x"]) == -60 and round(item["z"]) == 114:
            item["z"] -= 0.1
            item["x"] -= 0.5
            item["y"] += 8

    # Path to grid file
    grid_file = "grid\\occupancy_grid.npy"
    # Create PathFinder3D instance
    path_finder = pathfinder.PathFinder3D(grid_file, waiting_points)
    # Minimum distance to obstacles (1 to 4)
    min_distance = 3  # Configurable parameter
    max_distance = 3  # Maximum distance for ma2

    # Reset waypoint indices and mission statuses
    for drone_id in range(NUM_DRONES):
        current_waypoint_indices[drone_id] = 0
        mission_status[drone_id] = False
        drone_mission_data[drone_id]["phase"] = "idle"
        drone_mission_data[drone_id]["current_fire_index"] = 0
        drone_mission_data[drone_id]["assigned_fire"] = None
        drone_mission_data[drone_id]["next_fire"] = None
        drone_mission_data[drone_id]["extinguisher_dropped"] = False
        drone_mission_data[drone_id]["return_to_waiting"] = False

    print("Starting autonomous navigation for multiple drones...")
    print(f"Using {WAYPOINT_THRESHOLD}m waypoint threshold")

    # Initialize system and start path calculation thread
    path_thread = init_path_calculation_thread(path_finder, min_distance, max_distance)

    # Wait for drones to report their initial positions
    # We need this to calculate fire assignments
    initial_data = get_data(connection.receive_data())
    # Define common start point for distance calculations
    common_start = {"x": -77, "y": 1, "z": 75}

    for drone_id in range(NUM_DRONES):
        drones_start_points[drone_id]["x"] = initial_data[drone_id]["droneVector"][
            "x"
        ] + 3 * (drone_id == 4)
        drones_start_points[drone_id]["y"] = (
            initial_data[drone_id]["droneVector"]["y"] + 5
        )
        drones_start_points[drone_id]["z"] = initial_data[drone_id]["droneVector"]["z"]

        dx = (drones_start_points[drone_id]["x"] - common_start["x"]) * 0.1
        drones_start_points[drone_id]["x"] -= dx
        dy = (drones_start_points[drone_id]["y"] - common_start["y"]) * 0.1
        drones_start_points[drone_id]["y"] -= dy
        dz = (drones_start_points[drone_id]["z"] - common_start["z"]) * 0.1
        drones_start_points[drone_id]["z"] -= dz

    # Initialize fire assignments based on distance
    initialize_fire_assignments(fires_positions, common_start)

    schedule_all_paths(fires_positions, path_finder, min_distance, max_distance)
    connection.send_data("sendData")
    # Main control loop
    try:
        # Continue while at least one drone hasn't completed its mission
        while True:
            next_step(fires_positions, path_finder, min_distance, max_distance)

            # Check if all drones have completed all their assigned missions
            all_complete = True
            for drone_id in range(NUM_DRONES):
                if drone_mission_data[drone_id]["current_fire_index"] < len(
                    drone_fire_assignments[drone_id]
                ):
                    all_complete = False
                    break

            if all_complete:
                print("All drones have completed all their missions!")
                break

    except KeyboardInterrupt:
        print("Navigation terminated by user")
    except Exception as e:
        print(f"Error during navigation: {e}")
        import traceback

        traceback.print_exc()

    shutdown_system(path_thread)
    print("All drone missions completed or terminated.")


if __name__ == "__main__":
    # Запуск с предопределенными точками или генерация пути
    run_auto_navigation()
