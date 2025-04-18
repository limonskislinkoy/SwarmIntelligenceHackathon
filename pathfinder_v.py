import numpy as np
import heapq
from typing import List, Tuple, Dict, Set, Optional, Any
import open3d as o3d
import time
import os
import copy

class PathFinder3D:
    def __init__(self, grid_file: str):
        """
        Инициализация поиска пути в 3D сетке

        Args:
            grid_file: Путь к файлу с 3D сеткой препятствий (numpy array)
        """
        self.grid = np.load(grid_file)
        self.shape = self.grid.shape
        self.grid_file = grid_file

        # Получаем параметры из формы сетки
        self.x_min, self.y_min, self.z_min = -100, 0, -150
        self.step = 0.5
        self.x_max = self.x_min + (self.shape[0] - 1) * self.step
        self.y_max = self.y_min + (self.shape[1] - 1) * self.step
        self.z_max = self.z_min + (self.shape[2] - 1) * self.step

        # Создаем карту дистанций до препятствий
        self.distance_map = None
        self.distance_map_file = None

        # Добавляем словарь для хранения путей дронов
        self.drone_paths = {}

        # Создаем отдельную сетку для отслеживания занятых дронами клеток
        self.drones_grid = np.zeros_like(self.grid, dtype=np.uint8)

        print(f"Загружена сетка размером {self.shape}")
        print(f"Охват: X=[{self.x_min}, {self.x_max}], Y=[{self.y_min}, {self.y_max}], Z=[{self.z_min}, {self.z_max}]")
        print(f"Количество препятствий: {np.sum(self.grid)}")

    def coords_to_indices(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """
        Преобразует координаты в мире в индексы сетки

        Args:
            x, y, z: Координаты в мировом пространстве

        Returns:
            Tuple[int, int, int]: Индексы в сетке (i, j, k)
        """
        i = int((x - self.x_min) / self.step)
        j = int((y - self.y_min) / self.step)
        k = int((z - self.z_min) / self.step)

        # Проверяем границы
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1] and 0 <= k < self.shape[2]):
            raise ValueError(f"Координаты ({x}, {y}, {z}) выходят за пределы сетки")

        return (i, j, k)

    def indices_to_coords(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """
        Преобразует индексы сетки в координаты в мире

        Args:
            i, j, k: Индексы в сетке

        Returns:
            Tuple[float, float, float]: Координаты в мировом пространстве (x, y, z)
        """
        x = self.x_min + i * self.step
        y = self.y_min + j * self.step
        z = self.z_min + k * self.step
        return (x, y, z)

    def path_to_dict_format(self, path: List[Tuple[float, float, float]]) -> List[Dict[str, float]]:
        """
        Преобразует путь из кортежей в список словарей

        Args:
            path: Список точек пути в формате [(x, y, z), ...]

        Returns:
            List[Dict[str, float]]: Путь в формате [{'x': x, 'y': y, 'z': z}, ...]
        """
        return [{'x': point[0], 'y': point[1], 'z': point[2]} for point in path]

    def dict_format_to_path(self, path_dict: List[Dict[str, float]]) -> List[Tuple[float, float, float]]:
        """
        Преобразует путь из списка словарей в кортежи

        Args:
            path_dict: Путь в формате [{'x': x, 'y': y, 'z': z}, ...]

        Returns:
            List[Tuple[float, float, float]]: Список точек пути в формате [(x, y, z), ...]
        """
        return [(point['x'], point['y'], point['z']) for point in path_dict]

    def add_drone_path(self, drone_id: str, path: List[Tuple[float, float, float]],
                       avoid_distance: int = 1) -> None:
        """
        Добавляет путь дрона в словарь путей и обновляет сетку занятых клеток

        Args:
            drone_id: Идентификатор дрона
            path: Путь дрона в виде списка точек [(x, y, z), ...]
            avoid_distance: Расстояние вокруг пути дрона, которое должно быть свободно от других дронов
        """
        # Преобразуем путь в формат словарей
        path_dict = self.path_to_dict_format(path)

        # Сохраняем путь дрона
        self.drone_paths[drone_id] = path_dict

        # Обновляем сетку занятых дронами клеток
        self._update_drones_grid(avoid_distance)

        print(f"Добавлен путь для дрона {drone_id}, длина пути: {len(path)} точек")

    def remove_path_by_drone_id(self, drone_id: str, avoid_distance: int = 1) -> bool:
        """
        Удаляет путь дрона из словаря путей и обновляет сетку занятых клеток

        Args:
            drone_id: Идентификатор дрона
            avoid_distance: Расстояние вокруг путей для обновления сетки

        Returns:
            bool: True, если путь был успешно удален, иначе False
        """
        if drone_id in self.drone_paths:
            # Удаляем путь дрона
            del self.drone_paths[drone_id]

            # Обновляем сетку занятых дронами клеток
            self._update_drones_grid(avoid_distance)

            print(f"Удален путь для дрона {drone_id}")
            return True
        else:
            print(f"Дрон с ID {drone_id} не найден")
            return False

    def clear_all_drone_paths(self) -> None:
        """
        Очищает все пути дронов и сбрасывает сетку занятых клеток
        """
        self.drone_paths = {}
        self.drones_grid = np.zeros_like(self.grid, dtype=np.uint8)
        print("Все пути дронов удалены")

    def _update_drones_grid(self, avoid_distance: int = 1) -> None:
        """
        Обновляет сетку занятых дронами клеток

        Args:
            avoid_distance: Расстояние вокруг путей дронов, которое должно быть свободно от других дронов
        """
        # Сбрасываем сетку занятых клеток
        self.drones_grid = np.zeros_like(self.grid, dtype=np.uint8)

        # Для каждого дрона отмечаем его путь и зону вокруг него на сетке
        for drone_id, path_dict in self.drone_paths.items():
            path = self.dict_format_to_path(path_dict)

            # Для каждой точки пути
            for point in path:
                try:
                    i, j, k = self.coords_to_indices(*point)
                    # Отмечаем клетку и клетки вокруг неё на расстоянии avoid_distance
                    for di in range(-avoid_distance, avoid_distance + 1):
                        for dj in range(-avoid_distance, avoid_distance + 1):
                            for dk in range(-avoid_distance, avoid_distance + 1):
                                # Проверяем расстояние (манхэттенское)
                                if abs(di) + abs(dj) + abs(dk) <= avoid_distance:
                                    ni, nj, nk = i + di, j + dj, k + dk
                                    if 0 <= ni < self.shape[0] and 0 <= nj < self.shape[1] and 0 <= nk < self.shape[2]:
                                        self.drones_grid[ni, nj, nk] = 1
                except ValueError:
                    # Пропускаем точки вне сетки
                    continue

    def get_distance_map_filename(self, max_distance: int) -> str:
        """
        Создает имя файла для карты расстояний

        Args:
            max_distance: Максимальное расстояние для карты

        Returns:
            str: Имя файла для карты расстояний
        """
        # Получаем имя файла без расширения
        base_name = os.path.splitext(self.grid_file)[0]
        return f"{base_name}_distance_map_md{max_distance}.npy"

    def load_distance_map(self, max_distance: int) -> bool:
        """
        Загружает карту расстояний из файла, если она существует

        Args:
            max_distance: Максимальное расстояние для карты

        Returns:
            bool: True, если карта успешно загружена, иначе False
        """
        distance_map_file = self.get_distance_map_filename(max_distance)
        self.distance_map_file = distance_map_file

        if os.path.exists(distance_map_file):
            try:
                start_time = time.time()
                print(f"Загрузка карты расстояний из файла: {distance_map_file}...")
                self.distance_map = np.load(distance_map_file)
                elapsed_time = time.time() - start_time
                print(f"Карта расстояний загружена за {elapsed_time:.2f} сек")
                return True
            except Exception as e:
                print(f"Ошибка при загрузке карты расстояний: {e}")
                self.distance_map = None
                return False
        return False

    def save_distance_map(self, max_distance: int) -> bool:
        """
        Сохраняет карту расстояний в файл

        Args:
            max_distance: Максимальное расстояние для карты

        Returns:
            bool: True, если карта успешно сохранена, иначе False
        """
        if self.distance_map is None:
            return False

        distance_map_file = self.get_distance_map_filename(max_distance)
        self.distance_map_file = distance_map_file

        try:
            start_time = time.time()
            print(f"Сохранение карты расстояний в файл: {distance_map_file}...")
            np.save(distance_map_file, self.distance_map)
            elapsed_time = time.time() - start_time
            print(f"Карта расстояний сохранена за {elapsed_time:.2f} сек")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении карты расстояний: {e}")
            return False

    def compute_distance_map(self, max_distance: int = 4):
        """
        Вычисляет карту расстояний до препятствий или загружает из файла, если она существует

        Args:
            max_distance: Максимальное расстояние, которое нужно вычислять
        """
        # Сначала пытаемся загрузить карту
        if self.load_distance_map(max_distance):
            return

        print(f"Вычисление карты расстояний до препятствий (макс. расстояние: {max_distance})...")
        start_time = time.time()

        # Создаем карту расстояний, где:
        # 0 - препятствие
        # 1-max_distance - расстояние до ближайшего препятствия
        # max_distance+1 - свободная ячейка, дальше чем max_distance от препятствий
        self.distance_map = np.ones(self.shape, dtype=np.uint8) * (max_distance + 1)

        # Ставим 0 для всех препятствий
        self.distance_map[self.grid == 1] = 0

        # Для каждого расстояния
        for dist in range(1, max_distance + 1):
            # Находим все ячейки с расстоянием dist-1
            indices = np.where(self.distance_map == dist - 1)
            coords = list(zip(indices[0], indices[1], indices[2]))

            # Для каждой такой ячейки
            for i, j, k in coords:
                # Проверяем соседей
                for di, dj, dk in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                    ni, nj, nk = i + di, j + dj, k + dk

                    # Проверяем границы
                    if 0 <= ni < self.shape[0] and 0 <= nj < self.shape[1] and 0 <= nk < self.shape[2]:
                        # Если это свободная ячейка с неопределенным расстоянием
                        if self.grid[ni, nj, nk] == 0 and self.distance_map[ni, nj, nk] > dist:
                            self.distance_map[ni, nj, nk] = dist

        elapsed_time = time.time() - start_time
        print(f"Карта расстояний создана за {elapsed_time:.2f} сек")

        # Сохраняем карту для будущего использования
        self.save_distance_map(max_distance)

    def is_valid_position(self, i: int, j: int, k: int, min_distance: int = 1, exclude_drone_id: str = None) -> bool:
        """
        Проверяет, является ли позиция допустимой с учетом дистанции до препятствий и других дронов

        Args:
            i, j, k: Индексы в сетке
            min_distance: Минимальное требуемое расстояние до препятствий
            exclude_drone_id: ID дрона, путь которого игнорируется при проверке

        Returns:
            bool: True, если позиция допустима, иначе False
        """
        # Проверка границ
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1] and 0 <= k < self.shape[2]):
            return False

        # Проверка на отсутствие препятствия
        if self.grid[i, j, k] == 1:
            return False

        # Проверка на отсутствие пересечений с путями других дронов
        if exclude_drone_id is not None:
            # Если мы строим путь для конкретного дрона, и эта клетка занята другим дроном
            if self.drones_grid[i, j, k] == 1:
                # Нам нужно проверить, не является ли эта клетка частью пути текущего дрона
                # (это возможно, если мы перепланируем путь)
                # Преобразуем индексы в координаты
                x, y, z = self.indices_to_coords(i, j, k)

                # Проверяем, находится ли точка на пути исключаемого дрона
                if exclude_drone_id in self.drone_paths:
                    path_dict = self.drone_paths[exclude_drone_id]
                    for point in path_dict:
                        # Проверяем, совпадает ли точка с какой-либо точкой исключаемого пути
                        if (abs(point['x'] - x) < self.step/2 and
                                abs(point['y'] - y) < self.step/2 and
                                abs(point['z'] - z) < self.step/2):
                            # Эта клетка принадлежит пути текущего дрона, можно использовать
                            return True

                # Если мы здесь, значит клетка занята другими дронами
                return False

        # Если используем карту расстояний
        if self.distance_map is not None:
            # Проверяем, что расстояние до препятствий >= min_distance
            return self.distance_map[i, j, k] >= min_distance

        return True

    def get_neighbors(self, position: Tuple[int, int, int], min_distance: int = 1, exclude_drone_id: str = None) -> List[Tuple[int, int, int]]:
        """
        Получает всех допустимых соседей для данной позиции

        Args:
            position: Текущая позиция (i, j, k)
            min_distance: Минимальное требуемое расстояние до препятствий
            exclude_drone_id: ID дрона, путь которого игнорируется при проверке

        Returns:
            List[Tuple[int, int, int]]: Список допустимых соседних позиций
        """
        i, j, k = position
        neighbors = []

        # 6-связность: соседи по основным осям
        for di, dj, dk in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            ni, nj, nk = i + di, j + dj, k + dk
            if self.is_valid_position(ni, nj, nk, min_distance, exclude_drone_id):
                neighbors.append((ni, nj, nk))

        return neighbors

    def heuristic(self, position: Tuple[int, int, int], goal: Tuple[int, int, int]) -> float:
        """
        Эвристическая функция для A* (манхэттенское расстояние)

        Args:
            position: Текущая позиция (i, j, k)
            goal: Целевая позиция (i, j, k)

        Returns:
            float: Эвристическая оценка расстояния
        """
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1]) + abs(position[2] - goal[2])

    def get_direction(self, from_pos: Tuple[int, int, int], to_pos: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Определяет направление движения между двумя точками

        Args:
            from_pos: Начальная позиция (i, j, k)
            to_pos: Конечная позиция (i, j, k)

        Returns:
            Tuple[int, int, int]: Направление движения (di, dj, dk)
        """
        di = 0 if to_pos[0] == from_pos[0] else (1 if to_pos[0] > from_pos[0] else -1)
        dj = 0 if to_pos[1] == from_pos[1] else (1 if to_pos[1] > from_pos[1] else -1)
        dk = 0 if to_pos[2] == from_pos[2] else (1 if to_pos[2] > from_pos[2] else -1)
        return (di, dj, dk)

    def find_path_with_min_turns(self, start_coords: Tuple[float, float, float],
                                 goal_coords: Tuple[float, float, float],
                                 drone_id: str = None,
                                 min_distance: int = 1,
                                 max_distance: int = 4,
                                 turn_penalty: float = 2.0,
                                 auto_add_path: bool = True) -> Optional[List[Tuple[float, float, float]]]:
        """
        Находит путь между двумя точками с минимальным количеством поворотов,
        учитывая уже существующие пути других дронов

        Args:
            start_coords: Начальная точка в мировом пространстве (x, y, z)
            goal_coords: Конечная точка в мировом пространстве (x, y, z)
            drone_id: Идентификатор дрона, для которого строится путь
            min_distance: Минимальное расстояние до препятствий (от 1 до max_distance)
            max_distance: Максимальное расстояние для карты расстояний
            turn_penalty: Штраф за изменение направления движения
            auto_add_path: Автоматически добавлять найденный путь в словарь путей

        Returns:
            Optional[List[Tuple[float, float, float]]]: Список точек пути в мировом пространстве,
                                                     или None, если путь не найден
        """
        # Вычисляем карту расстояний, если её еще нет или max_distance изменился
        if self.distance_map is None:
            self.compute_distance_map(max_distance)

        try:
            start = self.coords_to_indices(*start_coords)
            goal = self.coords_to_indices(*goal_coords)
        except ValueError as e:
            print(f"Ошибка: {e}")
            return None

        # Проверяем начальную и конечную точки
        if not self.is_valid_position(*start, min_distance, exclude_drone_id=drone_id):
            # Удаляем старый путь, если он существует
            if drone_id in self.drone_paths:
                self.remove_path_by_drone_id(drone_id)
            # Если начальная точка слишком близко к препятствию, можно попробовать найти ближайшую допустимую
            print(f"Предупреждение: Начальная позиция {start_coords} слишком близко к препятствию или другому дрону")
            found_valid = False
            for radius in range(1, 10):  # Ищем в радиусе до 10 ячеек
                for di in range(-radius, radius+1):
                    for dj in range(-radius, radius+1):
                        for dk in range(-radius, radius+1):
                            if abs(di) + abs(dj) + abs(dk) == radius:  # Только по периметру
                                ni, nj, nk = start[0] + di, start[1] + dj, start[2] + dk
                                if self.is_valid_position(ni, nj, nk, min_distance, exclude_drone_id=drone_id):
                                    start = (ni, nj, nk)
                                    start_coords = self.indices_to_coords(*start)
                                    print(f"Использую ближайшую допустимую позицию: {start_coords}")
                                    found_valid = True
                                    break
                        if found_valid:
                            break
                    if found_valid:
                        break
                if found_valid:
                    break
            if not found_valid:
                print(f"Ошибка: Не удалось найти допустимую позицию рядом с начальной точкой")
                return None

        if not self.is_valid_position(*goal, min_distance, exclude_drone_id=drone_id):
            # То же самое для конечной точки
            print(f"Предупреждение: Конечная позиция {goal_coords} слишком близко к препятствию или другому дрону")
            found_valid = False
            for radius in range(1, 10):
                for di in range(-radius, radius+1):
                    for dj in range(-radius, radius+1):
                        for dk in range(-radius, radius+1):
                            if abs(di) + abs(dj) + abs(dk) == radius:
                                ni, nj, nk = goal[0] + di, goal[1] + dj, goal[2] + dk
                                if self.is_valid_position(ni, nj, nk, min_distance, exclude_drone_id=drone_id):
                                    goal = (ni, nj, nk)
                                    goal_coords = self.indices_to_coords(*goal)
                                    print(f"Использую ближайшую допустимую позицию: {goal_coords}")
                                    found_valid = True
                                    break
                        if found_valid:
                            break
                    if found_valid:
                        break
                if found_valid:
                    break
            if not found_valid:
                print(f"Ошибка: Не удалось найти допустимую позицию рядом с конечной точкой")
                return None

        # Инициализация модифицированного A* для минимизации поворотов
        # Используем состояние (позиция, направление)
        initial_direction = (0, 0, 0)  # Начальное направление не определено
        open_set = [(0, 0, (start, initial_direction))]  # (f, turns, (position, direction))
        heapq.heapify(open_set)

        g_score = {(start, initial_direction): 0}  # Стоимость от начала до текущей точки
        turns_count = {(start, initial_direction): 0}  # Количество поворотов на пути
        came_from = {}  # Для восстановления пути

        f_score = {(start, initial_direction): self.heuristic(start, goal)}  # f = g + h + turns_penalty * turns

        closed_set = set()  # Посещенные узлы

        while open_set:
            # Получаем узел с наименьшей f-оценкой
            _, _, (current, current_direction) = heapq.heappop(open_set)

            if current == goal:
                # Путь найден, восстанавливаем его
                path = []
                state = (current, current_direction)
                while state in came_from:
                    path.append(self.indices_to_coords(*state[0]))
                    state = came_from[state]
                path.append(self.indices_to_coords(*start))

                # Меняем порядок, чтобы путь был от начала к концу
                path = path[::-1]

                # Если задан drone_id и auto_add_path=True, добавляем путь в словарь путей
                if drone_id is not None and auto_add_path:
                    # Удаляем старый путь, если он существует
                    if drone_id in self.drone_paths:
                        self.remove_path_by_drone_id(drone_id)

                    # Добавляем новый путь
                    self.add_drone_path(drone_id, path, min_distance)

                return path

            closed_set.add((current, current_direction))

            # Проверяем всех соседей с учетом минимального расстояния и исключаемого дрона
            for neighbor in self.get_neighbors(current, min_distance, exclude_drone_id=drone_id):
                # Определяем новое направление
                new_direction = self.get_direction(current, neighbor)

                # Если мы еще не двигались (начальная точка) или изменили направление
                turn_occurred = current_direction != (0, 0, 0) and new_direction != current_direction

                neighbor_state = (neighbor, new_direction)

                if neighbor_state in closed_set:
                    continue

                # Базовое расстояние до соседа (всегда 1 в нашем случае)
                move_cost = 1.0

                # Если произошел поворот, добавляем штраф
                if turn_occurred:
                    move_cost += turn_penalty

                tentative_g_score = g_score[(current, current_direction)] + move_cost

                # Количество поворотов на пути
                tentative_turns = turns_count[(current, current_direction)] + (1 if turn_occurred else 0)

                if neighbor_state not in g_score or tentative_g_score < g_score[neighbor_state]:
                    # Нашли лучший путь к соседу
                    came_from[neighbor_state] = (current, current_direction)
                    g_score[neighbor_state] = tentative_g_score
                    turns_count[neighbor_state] = tentative_turns

                    # f-оценка = g-оценка + эвристика
                    f_value = g_score[neighbor_state] + self.heuristic(neighbor, goal)

                    # Добавляем в открытый набор
                    heapq.heappush(open_set, (f_value, tentative_turns, neighbor_state))

        print("Путь не найден")
        return None


    def count_turns_in_path(path):
        """
        Считает количество поворотов в пути

        Args:
            path: Список точек пути

        Returns:
            int: Количество поворотов в пути
        """
        if len(path) < 3:
            return 0

        turns = 0
        prev_direction = None

        for i in range(1, len(path)):
            # Вычисляем текущее направление
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            dz = path[i][2] - path[i - 1][2]

            current_direction = (dx, dy, dz)

            # Если направление изменилось, считаем это поворотом
            if prev_direction is not None and current_direction != prev_direction:
                turns += 1

            prev_direction = current_direction

        return turns


    # Улучшенная функция визуализации для показа всех маршрутов дронов
    def visualize_paths_with_open3d(self,
                                    highlight_path: List[Tuple[float, float, float]] = None,
                                    start_coords: Tuple[float, float, float] = None,
                                    goal_coords: Tuple[float, float, float] = None,
                                    min_distance: int = 1,
                                    sample_obstacles: int = 99999999):
        """
        Визуализирует все маршруты дронов, а также выделенный маршрут (если указан)

        Args:
            highlight_path: Маршрут для выделения (опционально)
            start_coords: Начальная точка выделенного маршрута (опционально)
            goal_coords: Конечная точка выделенного маршрута (опционально)
            min_distance: Минимальное расстояние до препятствий
            sample_obstacles: Количество препятствий для отображения (выборка)
        """
        # Создаем геометрические объекты для визуализации
        geometries = []

        # Получаем образец препятствий для визуализации
        obstacle_indices = np.where(self.grid == 1)
        obstacle_count = len(obstacle_indices[0])

        if obstacle_count > 0:
            # Выбираем случайную выборку препятствий, если их слишком много
            if obstacle_count > sample_obstacles:
                sample_idx = np.random.choice(obstacle_count, sample_obstacles, replace=False)
                obs_i = obstacle_indices[0][sample_idx]
                obs_j = obstacle_indices[1][sample_idx]
                obs_k = obstacle_indices[2][sample_idx]
            else:
                obs_i, obs_j, obs_k = obstacle_indices

            # Преобразуем индексы в координаты
            obstacle_coords = np.array([self.indices_to_coords(i, j, k) for i, j, k in zip(obs_i, obs_j, obs_k)])

            # Создаем PointCloud из координат препятствий
            obstacle_pcd = o3d.geometry.PointCloud()
            obstacle_pcd.points = o3d.utility.Vector3dVector(obstacle_coords)
            obstacle_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # красный цвет
            geometries.append(obstacle_pcd)

        # Показываем все занятые дронами клетки
        drone_grid_indices = np.where(self.drones_grid == 1)
        drone_grid_count = len(drone_grid_indices[0])

        if drone_grid_count > 0:
            # Ограничиваем количество отображаемых точек
            sample_drone_grid = min(drone_grid_count, sample_obstacles)
            if drone_grid_count > sample_drone_grid:
                sample_idx = np.random.choice(drone_grid_count, sample_drone_grid, replace=False)
                drone_i = drone_grid_indices[0][sample_idx]
                drone_j = drone_grid_indices[1][sample_idx]
                drone_k = drone_grid_indices[2][sample_idx]
            else:
                drone_i, drone_j, drone_k = drone_grid_indices

            # Преобразуем индексы в координаты
            drone_grid_coords = np.array([self.indices_to_coords(i, j, k) for i, j, k in zip(drone_i, drone_j, drone_k)])

            # Создаем PointCloud для занятых дронами клеток
            drone_grid_pcd = o3d.geometry.PointCloud()
            drone_grid_pcd.points = o3d.utility.Vector3dVector(drone_grid_coords)
            drone_grid_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # серый цвет
            geometries.append(drone_grid_pcd)

        # Показываем буферную зону (клетки недостаточно удаленные от препятствий)
        if min_distance > 1 and self.distance_map is not None:
            buffer_zone_indices = np.where((self.distance_map > 0) & (self.distance_map < min_distance))
            buffer_count = len(buffer_zone_indices[0])

            if buffer_count > 0:
                # Выборка, если слишком много точек
                sample_buffer = min(buffer_count, sample_obstacles)
                if buffer_count > sample_buffer:
                    sample_idx = np.random.choice(buffer_count, sample_buffer, replace=False)
                    buf_i = buffer_zone_indices[0][sample_idx]
                    buf_j = buffer_zone_indices[1][sample_idx]
                    buf_k = buffer_zone_indices[2][sample_idx]
                else:
                    buf_i, buf_j, buf_k = buffer_zone_indices

                # Координаты буферной зоны
                buffer_coords = np.array([self.indices_to_coords(i, j, k) for i, j, k in zip(buf_i, buf_j, buf_k)])

                # Показываем буферную зону
                buffer_pcd = o3d.geometry.PointCloud()
                buffer_pcd.points = o3d.utility.Vector3dVector(buffer_coords)
                buffer_pcd.paint_uniform_color([1.0, 0.5, 0.0])  # оранжевый цвет
                geometries.append(buffer_pcd)

        # Набор цветов для разных маршрутов дронов
        drone_colors = [
            [0.0, 0.0, 1.0],  # синий
            [0.0, 0.8, 0.8],  # голубой
            [0.0, 0.8, 0.0],  # зеленый
            [0.8, 0.8, 0.0],  # желтый
            [0.8, 0.4, 0.0],  # оранжевый
            [0.8, 0.0, 0.8],  # фиолетовый
        ]

        # Отображаем пути всех дронов
        for idx, (drone_id, path_dict) in enumerate(self.drone_paths.items()):
            path = self.dict_format_to_path(path_dict)
            color_idx = idx % len(drone_colors)

            # Создаем линию для маршрута
            line_points = o3d.utility.Vector3dVector(path)
            line_set = o3d.geometry.LineSet()
            line_set.points = line_points

            # Создаем линии между последовательными точками
            line_indices = [[i, i+1] for i in range(len(path)-1)]
            line_set.lines = o3d.utility.Vector2iVector(line_indices)
            line_set.paint_uniform_color(drone_colors[color_idx])
            geometries.append(line_set)

            # Точки маршрута в виде сфер
            for point in path:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.step/2)
                sphere.translate(point)
                sphere.paint_uniform_color(drone_colors[color_idx])
                geometries.append(sphere)

            # Отображаем ID дрона у начальной точки
            # (В Open3D нет прямого способа отображения текста, поэтому это заменено визуальным маркером)
            start_marker = o3d.geometry.TriangleMesh.create_sphere(radius=self.step)
            start_marker.translate(path[0])
            start_marker.paint_uniform_color(drone_colors[color_idx])
            geometries.append(start_marker)

            # Отображаем конечную точку
            end_marker = o3d.geometry.TriangleMesh.create_sphere(radius=self.step*0.7)
            end_marker.translate(path[-1])
            end_marker.paint_uniform_color([1.0, 1.0, 1.0])  # белый цвет
            geometries.append(end_marker)

        # Отдельно выделяем текущий маршрут (если есть)
        if highlight_path:
            # Рисуем выделенный маршрут жирными кубами
            for point in highlight_path:
                cube = o3d.geometry.TriangleMesh.create_box(width=self.step*1.2, height=self.step*1.2, depth=self.step*1.2)
                cube.compute_vertex_normals()
                cube.translate((point[0] - self.step*0.6, point[1] - self.step*0.6, point[2] - self.step*0.6))
                cube.paint_uniform_color([0.0, 0.5, 1.0])  # светло-синий цвет
                geometries.append(cube)

        # Отображаем начальную и конечную точки выделенного маршрута (если указаны)
        if start_coords:
            start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.step * 2)
            start_sphere.translate(start_coords)
            start_sphere.paint_uniform_color([0.0, 1.0, 0.2])  # зеленый цвет
            geometries.append(start_sphere)

        if goal_coords:
            goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.step * 2)
            goal_sphere.translate(goal_coords)
            goal_sphere.paint_uniform_color([1.0, 0.0, 1.0])  # пурпурный цвет
            geometries.append(goal_sphere)

        # Создаем координатную систему для ориентации
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
        geometries.append(coord_frame)

        # Визуализация
        drone_paths_count = len(self.drone_paths)
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"3D пути дронов (всего: {drone_paths_count}) с дистанцией {min_distance} до препятствий",
            width=1024,
            height=768,
            point_show_normal=False
        )






def simplify_path(points):
    if len(points) <= 2:
        return points

    simplified = [points[0]]  # Добавляем первую точку
    current_direction = None
    last_added_index = 0

    for i in range(1, len(points) - 1):
        # Вычисляем текущее направление от последней добавленной точки
        prev_point = points[last_added_index]
        current_point = points[i]
        next_point = points[i+1]

        # Текущее направление (от последней добавленной точки к текущей)
        current_direction = (
            current_point[0] - prev_point[0],
            current_point[1] - prev_point[1],
            current_point[2] - prev_point[2]
        )

        # Следующее направление (от текущей точки к следующей)
        next_direction = (
            next_point[0] - current_point[0],
            next_point[1] - current_point[1],
            next_point[2] - current_point[2]
        )

        # Прямое направление от последней добавленной точки к следующей
        direct_direction = (
            next_point[0] - prev_point[0],
            next_point[1] - prev_point[1],
            next_point[2] - prev_point[2]
        )

        # Проверяем, лежат ли три точки на одной прямой
        # Если точки не на одной линии, значит произошла смена направления
        if not are_points_collinear(prev_point, current_point, next_point):
            simplified.append(current_point)
            last_added_index = i

    # Всегда добавляем последнюю точку
    simplified.append(points[-1])

    return simplified

def are_points_collinear(p1, p2, p3, epsilon=1e-6):
    """
    Проверяет, лежат ли три точки на одной прямой.
    Используем векторное произведение: если объем параллелепипеда,
    образованного векторами p1p2 и p1p3, равен 0, точки коллинеарны.
    """
    # Векторы p1p2 и p1p3
    v1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    v2 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])

    # Проверяем коллинеарность через пропорциональность
    # Пытаемся найти такое число k, что v2 = k * v1

    # Находим первую ненулевую компоненту в v1
    for i in range(3):
        if abs(v1[i]) > epsilon:
            # Если нашли, проверяем пропорциональность
            k = v2[i] / v1[i]

            # Проверяем, что все компоненты v2 = k * v1
            return all(abs(v2[j] - k * v1[j]) <= epsilon for j in range(3))

    # Если v1 - нулевой вектор, проверяем, является ли v2 нулевым
    return all(abs(v) <= epsilon for v in v2)

def main():
    # Путь к файлу с сеткой
    grid_file = "grid\\occupancy_grid.npy"

    # Создаем экземпляр PathFinder3D
    path_finder = PathFinder3D(grid_file)

    # Минимальное расстояние до препятствий (от 1 до 4)
    min_distance = 3  # Настраиваемый параметр
    max_distance = 4  # Максимальное расстояние для карты

    # Начальная точка общая для всех маршрутов
    start_coords = (77, 1, 75)  # Начальная точка (x, y, z)

    # Строим первый маршрут
    goal_coords_1 = (80, 5, 45)    # Конечная точка первого маршрута
    print(f"\n[Маршрут 1] Поиск пути от {start_coords} до {goal_coords_1} с дистанцией {min_distance} до препятствий")
    path_1 = path_finder.find_path_with_min_turns(
        start_coords, goal_coords_1,
        drone_id="drone_1",
        min_distance=min_distance,
        max_distance=max_distance
    )

    if path_1:
        path_1 = simplify_path(path_1)
        print(f"Маршрут 1 найден! Длина пути: {len(path_1)} точек")
        print(f"Начальная точка: {path_1[0]}")
        print(f"Конечная точка: {path_1[-1]}")

        # Визуализируем первый маршрут
        path_finder.visualize_paths_with_open3d(
            highlight_path=path_1,
            start_coords=start_coords,
            goal_coords=goal_coords_1,
            min_distance=min_distance
        )
    else:
        print("Не удалось найти маршрут 1")

    # Строим второй маршрут из той же начальной точки
    goal_coords_2 = (75, 5, 55)    # Конечная точка второго маршрута
    print(f"\n[Маршрут 2] Поиск пути от {start_coords} до {goal_coords_2} с дистанцией {min_distance} до препятствий")
    path_2 = path_finder.find_path_with_min_turns(
        start_coords, goal_coords_2,
        drone_id="drone_1",
        min_distance=min_distance,
        max_distance=max_distance
    )

    if path_2:
        path_2 = simplify_path(path_2)
        print(f"Маршрут 2 найден! Длина пути: {len(path_2)} точек")
        print(f"Начальная точка: {path_2[0]}")
        print(f"Конечная точка: {path_2[-1]}")

        # Визуализируем оба маршрута, выделяя второй
        path_finder.visualize_paths_with_open3d(
            highlight_path=path_2,
            start_coords=start_coords,
            goal_coords=goal_coords_2,
            min_distance=min_distance
        )
    else:
        print("Не удалось найти маршрут 2")

    # Строим третий маршрут из той же начальной точки в другое место
    goal_coords_3 = (85, 5, 60)    # Конечная точка третьего маршрута
    print(f"\n[Маршрут 3] Поиск пути от {start_coords} до {goal_coords_3} с дистанцией {min_distance} до препятствий")
    path_3 = path_finder.find_path_with_min_turns(
        start_coords, goal_coords_3,
        drone_id="drone_3",
        min_distance=min_distance,
        max_distance=max_distance
    )

    if path_3:
        path_3 = simplify_path(path_3)
        print(f"Маршрут 3 найден! Длина пути: {len(path_3)} точек")
        print(f"Начальная точка: {path_3[0]}")
        print(f"Конечная точка: {path_3[-1]}")

        # Визуализируем все маршруты, выделяя третий
        path_finder.visualize_paths_with_open3d(
            highlight_path=path_3,
            start_coords=start_coords,
            goal_coords=goal_coords_3,
            min_distance=min_distance
        )
    else:
        print("Не удалось найти маршрут 3")

    # Визуализируем все пути вместе без выделения
    path_finder.visualize_paths_with_open3d(min_distance=min_distance)

if __name__ == "__main__":
    main()
