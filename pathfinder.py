import numpy as np
import heapq
from typing import List, Tuple, Dict, Set, Optional, Any
import open3d as o3d
import time
import os
import copy


class PathFinder3D:
    def __init__(self, grid_file: str, waiting_points):
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
        # Учитываем точки ожиданий дронов
        self.waiting_points = waiting_points

        print(f"Загружена сетка размером {self.shape}")
        print(
            f"Охват: X=[{self.x_min}, {self.x_max}], Y=[{self.y_min}, {self.y_max}], Z=[{self.z_min}, {self.z_max}]"
        )
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
        if not (
            0 <= i < self.shape[0] and 0 <= j < self.shape[1] and 0 <= k < self.shape[2]
        ):
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

    def path_to_dict_format(
        self, path: List[Tuple[float, float, float]]
    ) -> List[Dict[str, float]]:
        """
        Преобразует путь из кортежей в список словарей

        Args:
            path: Список точек пути в формате [(x, y, z), ...]

        Returns:
            List[Dict[str, float]]: Путь в формате [{'x': x, 'y': y, 'z': z}, ...]
        """
        return [{"x": point[0], "y": point[1], "z": point[2]} for point in path]

    def dict_format_to_path(
        self, path_dict: List[Dict[str, float]]
    ) -> List[Tuple[float, float, float]]:
        """
        Преобразует путь из списка словарей в кортежи

        Args:
            path_dict: Путь в формате [{'x': x, 'y': y, 'z': z}, ...]

        Returns:
            List[Tuple[float, float, float]]: Список точек пути в формате [(x, y, z), ...]
        """
        return [(point["x"], point["y"], point["z"]) for point in path_dict]

    def add_drone_path(
        self,
        drone_id: str,
        path: List[Tuple[float, float, float]],
        avoid_distance: int = 1,
    ) -> None:
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
                                    if (
                                        0 <= ni < self.shape[0]
                                        and 0 <= nj < self.shape[1]
                                        and 0 <= nk < self.shape[2]
                                    ):
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

        print(
            f"Вычисление карты расстояний до препятствий (макс. расстояние: {max_distance})..."
        )
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
                for di, dj, dk in [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ]:
                    ni, nj, nk = i + di, j + dj, k + dk

                    # Проверяем границы
                    if (
                        0 <= ni < self.shape[0]
                        and 0 <= nj < self.shape[1]
                        and 0 <= nk < self.shape[2]
                    ):
                        # Если это свободная ячейка с неопределенным расстоянием
                        if (
                            self.grid[ni, nj, nk] == 0
                            and self.distance_map[ni, nj, nk] > dist
                        ):
                            self.distance_map[ni, nj, nk] = dist

        elapsed_time = time.time() - start_time
        print(f"Карта расстояний создана за {elapsed_time:.2f} сек")

        # Сохраняем карту для будущего использования
        self.save_distance_map(max_distance)

    def is_valid_position(
        self,
        i: int,
        j: int,
        k: int,
        min_distance: int = 1,
        exclude_drone_id: str = None,
        waiting_point_radius: float = 2.0,
    ) -> bool:
        """
        Проверяет, является ли позиция допустимой с учетом дистанции до препятствий, других дронов и точек ожидания
        Args:
            i, j, k: Индексы в сетке
            min_distance: Минимальное требуемое расстояние до препятствий
            exclude_drone_id: ID дрона, путь которого игнорируется при проверке
            waiting_point_radius: Радиус безопасности вокруг точек ожидания
        Returns:
            bool: True, если позиция допустима, иначе False
        """
        # Проверка границ
        if not (
            0 <= i < self.shape[0] and 0 <= j < self.shape[1] and 0 <= k < self.shape[2]
        ):
            return False
        # Проверка на отсутствие препятствия
        if self.grid[i, j, k] == 1:
            return False

        # Проверка на близость к точкам ожидания дронов
        x, y, z = self.indices_to_coords(i, j, k)
        for drone_id, wait_point in self.waiting_points.items():
            # Если это точка ожидания исключаемого дрона, пропускаем проверку
            if exclude_drone_id is not None and str(drone_id) == str(exclude_drone_id):
                continue

            # Вычисляем расстояние до точки ожидания
            distance = (
                (x - wait_point["x"]) ** 2
                + (y - wait_point["y"]) ** 2
                + (z - wait_point["z"]) ** 2
            ) ** 0.5

            # Если точка находится слишком близко к точке ожидания
            if distance < waiting_point_radius:
                return False

        # Проверка на отсутствие пересечений с путями других дронов
        if exclude_drone_id is not None:
            # Если мы строим путь для конкретного дрона, и эта клетка занята другим дроном
            if self.drones_grid[i, j, k] == 1:
                # Нам нужно проверить, не является ли эта клетка частью пути текущего дрона
                # (это возможно, если мы перепланируем путь)
                # Преобразуем индексы в координаты уже было сделано выше
                # Проверяем, находится ли точка на пути исключаемого дрона
                if exclude_drone_id in self.drone_paths:
                    path_dict = self.drone_paths[exclude_drone_id]
                    for point in path_dict:
                        # Проверяем, совпадает ли точка с какой-либо точкой исключаемого пути
                        if (
                            abs(point["x"] - x) < self.step * 4
                            and abs(point["y"] - y) < self.step * 4
                            and abs(point["z"] - z) < self.step * 4
                        ):
                            # Эта клетка принадлежит пути текущего дрона, можно использовать
                            return True
                # Если мы здесь, значит клетка занята другими дронами
                return False

        # Если используем карту расстояний
        if self.distance_map is not None:
            # Проверяем, что расстояние до препятствий >= min_distance
            return self.distance_map[i, j, k] >= min_distance

        return True

    def get_neighbors(
        self,
        position: Tuple[int, int, int],
        min_distance: int = 1,
        exclude_drone_id: str = None,
    ) -> List[Tuple[int, int, int]]:
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
        for di, dj, dk in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]:
            ni, nj, nk = i + di, j + dj, k + dk
            if self.is_valid_position(ni, nj, nk, min_distance, exclude_drone_id):
                neighbors.append((ni, nj, nk))

        return neighbors

    def heuristic(
        self, position: Tuple[int, int, int], goal: Tuple[int, int, int]
    ) -> float:
        """
        Эвристическая функция для A* (манхэттенское расстояние)

        Args:
            position: Текущая позиция (i, j, k)
            goal: Целевая позиция (i, j, k)

        Returns:
            float: Эвристическая оценка расстояния
        """
        return (
            abs(position[0] - goal[0])
            + abs(position[1] - goal[1])
            + abs(position[2] - goal[2])
        )

    def get_direction(
        self, from_pos: Tuple[int, int, int], to_pos: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
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

    def find_path_with_min_turns(
        self,
        start_coords: Tuple[float, float, float],
        goal_coords: Tuple[float, float, float],
        drone_id: str = None,
        min_distance: int = 1,
        max_distance: int = 4,
        turn_penalty: float = 10.0,
        auto_add_path: bool = True,
        drone_avoid_distance: int = 4,
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Находит путь между двумя точками с минимальным количеством поворотов,
        используя двунаправленный A* алгоритм поиска.

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
        # Удаляем старый путь, если он существует
        if drone_id in self.drone_paths:
            self.remove_path_by_drone_id(drone_id, avoid_distance=drone_avoid_distance)

        # Вычисляем карту расстояний, если её еще нет или max_distance изменился
        if self.distance_map is None:
            self.compute_distance_map(max_distance)

        try:
            start = self.coords_to_indices(*start_coords)
            goal = self.coords_to_indices(*goal_coords)
        except ValueError as e:
            print(f"Ошибка: {e}")
            return None

        # Проверяем начальную и конечную точки и находим ближайшие валидные если нужно
        start = self._find_valid_position(start, start_coords, min_distance)
        if start is None:
            return None

        goal = self._find_valid_position(
            goal, goal_coords, min_distance, end_point=True
        )
        if goal is None:
            return None

        # Поднимаем целевую точку вверх, если возможно
        temp = list(goal)
        while self.is_valid_position(temp[0], temp[1] + 1, temp[2], min_distance):
            if temp[1] >= 15:
                break
            temp[1] += 1
        goal = tuple(temp)

        print(f"Запуск двунаправленного A* поиска: {start} -> {goal}")
        start_time = time.time()

        # Начальные направления
        initial_direction = (0, 0, 0)  # Начальное направление не определено

        # Структуры данных для поиска из начальной точки (вперед)
        open_forward = [
            (0, 0, (start, initial_direction))
        ]  # (f, turns, (position, direction))
        heapq.heapify(open_forward)

        g_forward = {(start, initial_direction): 0}  # Стоимость от начала
        turns_forward = {(start, initial_direction): 0}  # Количество поворотов
        came_from_forward = {}  # Для восстановления пути

        # Структуры данных для поиска из конечной точки (назад)
        open_backward = [
            (0, 0, (goal, initial_direction))
        ]  # (f, turns, (position, direction))
        heapq.heapify(open_backward)

        g_backward = {(goal, initial_direction): 0}  # Стоимость от цели
        turns_backward = {(goal, initial_direction): 0}  # Количество поворотов
        came_from_backward = {}  # Для восстановления пути

        # Посещенные узлы
        closed_forward = set()
        closed_backward = set()

        # Для отслеживания пересечения поисков
        best_path = None
        best_cost = float("inf")
        meeting_nodes = None

        # Счетчик итераций для периодического отчета
        iterations = 0
        max_iterations = 500000  # Ограничение на число итераций

        # Поочередно выполняем шаги в прямом и обратном направлениях
        while open_forward and open_backward and iterations < max_iterations:
            iterations += 1

            # Отчет о прогрессе каждые 5000 итераций
            if iterations % 5000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Поиск пути: выполнено {iterations} итераций за {elapsed:.2f} сек"
                )

            # Шаг в прямом направлении
            if open_forward:
                best_path, meeting_nodes = self._bidirectional_step(
                    open_forward,
                    g_forward,
                    turns_forward,
                    came_from_forward,
                    closed_forward,
                    g_backward,
                    closed_backward,
                    goal,
                    min_distance,
                    drone_id,
                    turn_penalty,
                    best_path,
                    best_cost,
                    meeting_nodes,
                    forward=True,
                )

            # Шаг в обратном направлении
            if open_backward:
                best_path, meeting_nodes = self._bidirectional_step(
                    open_backward,
                    g_backward,
                    turns_backward,
                    came_from_backward,
                    closed_backward,
                    g_forward,
                    closed_forward,
                    start,
                    min_distance,
                    drone_id,
                    turn_penalty,
                    best_path,
                    best_cost,
                    meeting_nodes,
                    forward=False,
                )

            # Если нашли путь, и наилучший путь не изменился несколько итераций, можно завершать
            if best_path is not None and (iterations % 1000 == 0):
                print(
                    f"Найден путь с {best_path[1]} поворотами после {iterations} итераций"
                )
                break

        elapsed_time = time.time() - start_time

        # Восстанавливаем полный путь, если он найден
        if meeting_nodes:
            forward_node, backward_node = meeting_nodes

            # Восстанавливаем путь от начала до точки встречи
            forward_path = []
            state = forward_node
            while state in came_from_forward:
                forward_path.append(self.indices_to_coords(*state[0]))
                state = came_from_forward[state]
            forward_path.append(self.indices_to_coords(*start))
            forward_path = forward_path[::-1]  # Разворачиваем, чтобы путь шел от начала

            # Восстанавливаем путь от точки встречи до цели
            backward_path = []
            state = backward_node
            while state in came_from_backward:
                backward_path.append(self.indices_to_coords(*state[0]))
                state = came_from_backward[state]
            backward_path.append(self.indices_to_coords(*goal))

            # Объединяем пути, исключая дублирование точки встречи
            full_path = forward_path + backward_path[1:]

            # Сглаживаем путь для устранения потенциальных артефактов в точке соединения
            full_path = self._smooth_path(full_path)

            print(
                f"Путь найден за {elapsed_time:.2f} сек, длина: {len(full_path)}, повороты: {self.count_turns_in_path(full_path)}"
            )

            # Если задан drone_id и auto_add_path=True, добавляем путь в словарь путей
            if drone_id is not None and auto_add_path:
                self.add_drone_path(drone_id, full_path, drone_avoid_distance)

            return full_path

        print(f"Путь не найден за {elapsed_time:.2f} сек после {iterations} итераций")
        return None

    def _bidirectional_step(
        self,
        open_set,
        g_score,
        turns_count,
        came_from,
        closed_set,
        other_g_score,
        other_closed_set,
        target,
        min_distance,
        drone_id,
        turn_penalty,
        best_path,
        best_cost,
        meeting_nodes,
        forward=True,
    ):
        """
        Выполняет один шаг двунаправленного A* алгоритма

        Args:
            open_set: Открытый список текущего направления
            g_score: G-оценки для текущего направления
            turns_count: Счетчик поворотов для текущего направления
            came_from: История пути для текущего направления
            closed_set: Закрытый список для текущего направления
            other_g_score: G-оценки для противоположного направления
            other_closed_set: Закрытый список для противоположного направления
            target: Целевая точка для текущего направления
            min_distance: Минимальное расстояние до препятствий
            drone_id: ID дрона
            turn_penalty: Штраф за поворот
            best_path: Текущий лучший найденный путь (общая стоимость, кол-во поворотов)
            meeting_nodes: Текущие точки встречи прямого и обратного поиска
            forward: True если это шаг прямого поиска, False для обратного

        Returns:
            Tuple: Обновленный best_path и meeting_nodes
        """
        # Получаем узел с наименьшей f-оценкой
        _, curr_turns, (current, current_direction) = heapq.heappop(open_set)

        # Если мы уже обработали этот узел, пропускаем
        if (current, current_direction) in closed_set:
            return best_path, meeting_nodes

        # Добавляем в обработанные
        closed_set.add((current, current_direction))

        # Проверяем, не встретились ли поиски
        if any(
            (current, dir) in other_closed_set
            for dir in [
                (0, 0, 0),
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ]
        ):
            # Нашли пересечение! Проверяем все возможные комбинации направлений в точке встречи
            for other_dir in [
                (0, 0, 0),
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ]:
                if (current, other_dir) in other_g_score:
                    # Вычисляем общую стоимость пути
                    total_g = (
                        g_score[(current, current_direction)]
                        + other_g_score[(current, other_dir)]
                    )
                    total_turns = turns_count[
                        (current, current_direction)
                    ] + other_g_score.get((current, other_dir), 0)

                    # Если этот путь лучше предыдущего лучшего
                    if total_g < best_cost:
                        best_cost = total_g
                        best_path = (total_g, total_turns)
                        # Сохраняем ноды встречи для восстановления пути
                        if forward:
                            meeting_nodes = (
                                (current, current_direction),
                                (current, other_dir),
                            )
                        else:
                            meeting_nodes = (
                                (current, other_dir),
                                (current, current_direction),
                            )

        # Получаем всех соседей
        for neighbor in self.get_neighbors(
            current, min_distance, exclude_drone_id=drone_id
        ):
            # Определяем новое направление
            if forward:
                new_direction = self.get_direction(current, neighbor)
            else:
                # Для обратного поиска направление обратное
                new_direction = tuple(-d for d in self.get_direction(neighbor, current))

            # Проверяем, был ли поворот
            turn_occurred = (
                current_direction != (0, 0, 0) and new_direction != current_direction
            )

            neighbor_state = (neighbor, new_direction)

            if neighbor_state in closed_set:
                continue

            # Базовое расстояние до соседа плюс штраф за поворот
            move_cost = 1.0 + (turn_penalty if turn_occurred else 0.0)

            tentative_g_score = g_score[(current, current_direction)] + move_cost
            tentative_turns = turns_count[(current, current_direction)] + (
                1 if turn_occurred else 0
            )

            if (
                neighbor_state not in g_score
                or tentative_g_score < g_score[neighbor_state]
            ):
                # Нашли лучший путь к соседу
                came_from[neighbor_state] = (current, current_direction)
                g_score[neighbor_state] = tentative_g_score
                turns_count[neighbor_state] = tentative_turns

                # f-оценка = g-оценка + эвристика
                if forward:
                    h_value = self.heuristic(neighbor, target)
                else:
                    h_value = self.heuristic(neighbor, target)

                f_value = tentative_g_score + h_value

                # Добавляем в открытый набор
                heapq.heappush(open_set, (f_value, tentative_turns, neighbor_state))

        return best_path, meeting_nodes

    def _find_valid_position(self, position, coords, min_distance, end_point=False):
        """
        Находит допустимую позицию с приоритетом вертикальных смещений
        Args:
            position: Позиция в индексах сетки (i, j, k)
            coords: Исходные координаты в мировом пространстве
            min_distance: Минимальное расстояние до препятствий
        Returns:
            Tuple или None: Допустимая позиция или None, если не найдена
        """
        # Проверяем текущую позицию
        if self.is_valid_position(*position, min_distance):
            return position

        print(f"Предупреждение: Позиция {coords} слишком близко к препятствию")

        i, j, k = position
        max_vertical_search = 20  # Максимальное вертикальное смещение
        horizontal_radius = 2  # Небольшое горизонтальное смещение

        # Создаем и сортируем смещения
        offsets = []
        for di in range(-horizontal_radius, horizontal_radius + 1):
            for dk in range(-horizontal_radius, horizontal_radius + 1):
                for dj in range(0, max_vertical_search + 1):
                    manhattan_dist = abs(di) + abs(dj) + abs(dk)
                    offsets.append((manhattan_dist, di, dj, dk))

        # Сортируем смещения
        offsets.sort()

        # Проверяем отсортированные смещения
        for _, di, dj, dk in offsets:
            new_i, new_j, new_k = i + di, j + dj, k + dk

            if self.is_valid_position(new_i, new_j, new_k, min_distance) and end_point:
                # Проверяем условие дополнительного поднятия
                new_position = (new_i, new_j, new_k)
                if new_j < 25 and self.is_valid_position(
                    new_i, new_j + 1, new_k, min_distance
                ):
                    new_position = (new_i, new_j + 1, new_k)

                new_coords = self.indices_to_coords(*new_position)
                print(f"Использую допустимую позицию со смещением: {new_coords}")
                return new_position

        print(f"Ошибка: Не удалось найти допустимую позицию рядом с точкой {coords}")
        return None

    def _smooth_path(self, path, window_size=5):
        """
        Сглаживает путь для устранения неоптимальных сегментов в месте соединения
        прямого и обратного путей

        Args:
            path: Исходный путь
            window_size: Размер окна сглаживания

        Returns:
            List: Сглаженный путь
        """
        if len(path) <= window_size:
            return path

        result = list(path)  # Копируем путь

        # Применяем скользящее окно для выявления и исправления зигзагов
        for i in range(2, len(result) - 2):
            # Проверяем, есть ли зигзаг (частая смена направления)
            prev_direction = (
                result[i][0] - result[i - 1][0],
                result[i][1] - result[i - 1][1],
                result[i][2] - result[i - 1][2],
            )

            next_direction = (
                result[i + 1][0] - result[i][0],
                result[i + 1][1] - result[i][1],
                result[i + 1][2] - result[i][2],
            )

            # Если направления противоположны или сильно отличаются, это может быть артефактом
            # объединения прямого и обратного путей - попробуем сгладить
            if (
                prev_direction[0] * next_direction[0] < 0
                or prev_direction[1] * next_direction[1] < 0
                or prev_direction[2] * next_direction[2] < 0
            ):

                # Пытаемся найти лучший промежуточный маршрут
                # Простейший вариант - просто удалить точку, создающую зигзаг
                # В более сложной реализации здесь можно было бы запустить локальный A*
                result[i] = (
                    (result[i - 1][0] + result[i + 1][0]) / 2,
                    (result[i - 1][1] + result[i + 1][1]) / 2,
                    (result[i - 1][2] + result[i + 1][2]) / 2,
                )

        return result

    def count_turns_in_path(self, path):
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
            # Вычисляем текущее направление (с округлением для устойчивости к мелким флуктуациям)
            dx = round(path[i][0] - path[i - 1][0], 5)
            dy = round(path[i][1] - path[i - 1][1], 5)
            dz = round(path[i][2] - path[i - 1][2], 5)

            # Нормализуем направление для устойчивости расчета
            norm = max(abs(dx), abs(dy), abs(dz), 0.0001)  # Избегаем деления на 0
            dx, dy, dz = dx / norm, dy / norm, dz / norm

            current_direction = (dx, dy, dz)

            # Если направление существенно изменилось, считаем это поворотом
            if prev_direction is not None:
                # Вычисляем скалярное произведение для определения угла между векторами
                dot_product = (
                    prev_direction[0] * current_direction[0]
                    + prev_direction[1] * current_direction[1]
                    + prev_direction[2] * current_direction[2]
                )

                # Если угол существенный (скалярное произведение < 0.95 ≈ ~18°), считаем поворотом
                if dot_product < 0.95:
                    turns += 1

            prev_direction = current_direction

        return turns

    def visualize_path_with_open3d(
        self,
        path: List[Tuple[float, float, float]],
        start_coords: Tuple[float, float, float],
        goal_coords: Tuple[float, float, float],
        min_distance: int = 1,
        sample_obstacles: int = 99999999,
    ):
        """
        Визуализирует найденный путь и образец препятствий с помощью Open3D

        Args:
            path: Список точек пути в мировом пространстве
            start_coords: Начальная точка в мировом пространстве
            goal_coords: Конечная точка в мировом пространстве
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
                sample_idx = np.random.choice(
                    obstacle_count, sample_obstacles, replace=False
                )
                obs_i = obstacle_indices[0][sample_idx]
                obs_j = obstacle_indices[1][sample_idx]
                obs_k = obstacle_indices[2][sample_idx]
            else:
                obs_i, obs_j, obs_k = obstacle_indices

            # Преобразуем индексы в координаты
            obstacle_coords = np.array(
                [
                    self.indices_to_coords(i, j, k)
                    for i, j, k in zip(obs_i, obs_j, obs_k)
                ]
            )

            # Создаем PointCloud из координат препятствий
            obstacle_pcd = o3d.geometry.PointCloud()
            obstacle_pcd.points = o3d.utility.Vector3dVector(obstacle_coords)
            obstacle_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # красный цвет
            geometries.append(obstacle_pcd)

        # Если хотим показать зону буфера, добавим ее
        if min_distance > 1 and self.distance_map is not None:
            buffer_zone_indices = np.where(
                (self.distance_map > 0) & (self.distance_map < min_distance)
            )
            buffer_count = len(buffer_zone_indices[0])

            if buffer_count > 0:
                # Выборка, если слишком много точек
                sample_buffer = min(buffer_count, sample_obstacles)
                if buffer_count > sample_buffer:
                    sample_idx = np.random.choice(
                        buffer_count, sample_buffer, replace=False
                    )
                    buf_i = buffer_zone_indices[0][sample_idx]
                    buf_j = buffer_zone_indices[1][sample_idx]
                    buf_k = buffer_zone_indices[2][sample_idx]
                else:
                    buf_i, buf_j, buf_k = buffer_zone_indices

                # Координаты буферной зоны
                buffer_coords = np.array(
                    [
                        self.indices_to_coords(i, j, k)
                        for i, j, k in zip(buf_i, buf_j, buf_k)
                    ]
                )

                # Показываем буферную зону
                buffer_pcd = o3d.geometry.PointCloud()
                buffer_pcd.points = o3d.utility.Vector3dVector(buffer_coords)
                buffer_pcd.paint_uniform_color([1.0, 0.5, 0.0])  # оранжевый цвет
                geometries.append(buffer_pcd)

        # Отображаем путь в виде синих кубов
        if path:
            for point in path:
                # Создаем куб для каждой точки пути
                cube = o3d.geometry.TriangleMesh.create_box(
                    width=self.step, height=self.step, depth=self.step
                )
                cube.compute_vertex_normals()
                # Центрируем куб на точке
                cube.translate(
                    (
                        point[0] - self.step / 2,
                        point[1] - self.step / 2,
                        point[2] - self.step / 2,
                    )
                )
                cube.paint_uniform_color([0.0, 0.0, 1.0])  # синий цвет
                geometries.append(cube)

        # Отображаем начальную и конечную точки в виде больших сфер
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.step * 2)
        start_sphere.translate(start_coords)
        start_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # зеленый цвет
        geometries.append(start_sphere)

        goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.step * 2)
        goal_sphere.translate(goal_coords)
        goal_sphere.paint_uniform_color([1.0, 0.0, 1.0])  # пурпурный цвет
        geometries.append(goal_sphere)

        # Создаем координатную систему для ориентации
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5.0, origin=[0, 0, 0]
        )
        geometries.append(coord_frame)

        # Визуализация
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"3D Путь с дистанцией {min_distance} до препятствий",
            width=1024,
            height=768,
            point_show_normal=False,
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
        next_point = points[i + 1]

        # Текущее направление (от последней добавленной точки к текущей)
        current_direction = (
            current_point[0] - prev_point[0],
            current_point[1] - prev_point[1],
            current_point[2] - prev_point[2],
        )

        # Следующее направление (от текущей точки к следующей)
        next_direction = (
            next_point[0] - current_point[0],
            next_point[1] - current_point[1],
            next_point[2] - current_point[2],
        )

        # Прямое направление от последней добавленной точки к следующей
        direct_direction = (
            next_point[0] - prev_point[0],
            next_point[1] - prev_point[1],
            next_point[2] - prev_point[2],
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
    path_finder = PathFinder3D(grid_file, waiting_points={})

    # Минимальное расстояние до препятствий (от 1 до 4)
    min_distance = 3  # Настраиваемый параметр
    max_distance = 3  # Максимальное расстояние для карты

    # Пример использования
    start_coords = (77, 1, 75)  # Начальная точка (x, y, z)
    goal_coords = (
        68.65518951416016,
        8.089997291564941,
        56.14305114746094,
    )  # Конечная точка (x, y, z)

    print(
        f"Поиск пути от {start_coords} до {goal_coords} с дистанцией {min_distance} до препятствий"
    )

    # Находим путь с учетом дистанции
    path = path_finder.find_path_with_min_turns(
        start_coords, goal_coords, min_distance=min_distance, max_distance=max_distance
    )
    path = simplify_path(path)
    if path:
        print(f"Путь найден! Длина пути: {len(path)} точек")
        print(f"Начальная точка: {path[0]}")
        print(f"Конечная точка: {path[-1]}")

        # Визуализируем путь с помощью Open3D
        path_finder.visualize_path_with_open3d(
            path, start_coords, goal_coords, min_distance
        )
    else:
        print("Не удалось найти путь")


if __name__ == "__main__":
    main()
