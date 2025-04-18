import asyncio
import logging
import queue
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, APIRouter

from config import PORT, HOST, QUESTION_FOR_SIM, CORRECT_ANSWER
from connection.Connection import Connection


class SocketConnection(Connection):
    def __init__(self):
        self.__app = FastAPI()
        self.__websocket: Optional[WebSocket] = None
        self.__server = None

        self.__received_queue = queue.Queue()
        self.__sending_queue = queue.Queue()

        self.__create_route()

    async def set_connection(self):
        """Запуск сервера в отдельном потоке и ожидание подключения симулятора.
        Функция блокирует поток во время ожидания"""
        threading.Thread(target=self.__run_server).start()

        logging.info("Сервер запущен")
        logging.info("Ждем запуска симулятора...")

        while not self.__websocket:
            await asyncio.sleep(1)
        logging.info("Есть соединение с симулятором!")

    def receive_data(self):
        data = self.__received_queue.get()
        logging.debug(f"Данные с симулятора: {data}")
        return data

    def send_data(self, data: str):
        logging.debug(f"Ответ: {data}")
        self.__sending_queue.put(data)

    def close_connection(self):
        self.send_data("close_connection")
        logging.info("Разорвали соединение с вебсокетом")

        self.__server.should_exit = True
        logging.info("Отключение сервера...")

    def __create_route(self):
        """Добавляем обработчик на адрес /"""
        router = APIRouter()
        router.add_api_websocket_route("/", self.__websocket_sim)
        self.__app.include_router(router)

    async def __websocket_sim(self, websocket: WebSocket):
        await websocket.accept()
        await self.__connect_sim(websocket)

        while True:
            data = await websocket.receive_text()
            self.__received_queue.put(data)

            new_data = self.__sending_queue.get()
            await websocket.send_text(new_data)

    async def __connect_sim(self, websocket):
        await websocket.send_text(QUESTION_FOR_SIM)
        answer = await websocket.receive_text()

        if answer != CORRECT_ANSWER:
            logging.info(f"Вебсокет прислал не верный ответ: \"{answer}\"")
            return

        self.__websocket = websocket
        logging.info(f"Симулятор подключен: {websocket}")

    def __run_server(self):
        self.__server = uvicorn.Server(uvicorn.Config(self.__app, host=HOST, port=PORT, log_level="critical"))
        self.__server.run()
        logging.info("Сервер отключен")
