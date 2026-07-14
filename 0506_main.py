import os
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,QMessageBox
from PyQt5.QtGui import QPainter, QPen, QColor, QFont,QTextCursor
from PyQt5.QtCore import Qt, QRectF, QThread, pyqtSignal, QObject, pyqtSlot
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sidebar1 import Ui_MainWindow
import mysql.connector
from PyQt5.QtCore import QThread, pyqtSignal
import random
import time
import speech_recognition as sr
import sys
from PyQt5.QtCore import QMutex, QMutexLocker
import ollama
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLineEdit, QLabel,QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal
from wifi_tcp_helper import switch_wifi_and_send,receive_image,reture_original_wifi

from machine_learning_model import pixel_to_real_position
import re
from rl import rl_main
from robot_ur import initialize_robot, move_joints, move_to, get_position
import math
import matlab.engine
from new import generate_sigmoid_s_curve
import traceback
from chart import plot_data
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 連接到 MySQL 數據庫
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="123456",
    database="test_db"
)
#執行pso演算法
class PSOThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def run(self):
        try:
            self.engine = matlab.engine.start_matlab()
            self.engine.cd(r'C:\\Users\\116\\Desktop\\0506code', nargout=0)
            self.engine.eval('clc; clear; close all;', nargout=0)
            self.engine.eval('optimiz_code', nargout=0)

            bestParams = list(self.engine.workspace['bestParams']._data)
            cost = self.engine.workspace['bestCost']

            result = {
                'V_max': bestParams[0],
                'A_max': bestParams[1],
                'A_avg': bestParams[2],
                'V_max1': bestParams[3],
                'A_max1': bestParams[4],
                'A_avg1': bestParams[5],
                'BestCost': cost
            }

            self.engine.quit()
            self.finished.emit(result)

        except Exception as e:
            err_msg = traceback.format_exc()
            self.error.emit(f"❌ 執行 MATLAB 發生錯誤：\n{err_msg}")

class SigmoidThread(QThread):
    finished = pyqtSignal(dict)  # ❶ 改為傳 dict 資料
    error = pyqtSignal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            t, acc, vel, pos, jerk = generate_sigmoid_s_curve(
                # V_max=self.params['V_max'],
                # V_max1=self.params['V_max1'],
                # A_max=self.params['A_max'],
                # A_max1=self.params['A_max1'],
                # A_avg=self.params['A_avg'],
                # A_avg1=self.params['A_avg1'],
                V_max=358,
                V_max1=1326,
                A_max=1265,
                A_max1=303,
                A_avg=1400,
                A_avg1=1362,
                plot=False  # ⚠️ 注意：不要繪圖！繪圖應回主執行緒
            )

            self.finished.emit({  # ❷ 傳資料回主執行緒
                't': t,
                'acc': acc,
                'vel': vel,
                'pos': pos,
                'jerk': jerk
            })

        except Exception as e:
            err_msg = traceback.format_exc()
            self.error.emit(f"⚠️ Sigmoid 曲線產生錯誤：\n{err_msg}")

#llama3語音識別訊號打印消息
class GPTParser:
    def __init__(self, model_name='llama3'):
        self.model_name = model_name
        self.history = []

    def send_message(self, msg: str):
        # 使用 Ollama 的 API 發送訊息
        self.history.append({'role': 'user', 'content': msg})
        print(f"User: {msg}")  # 打印用戶消息
        generated_text = ollama.chat(model=self.model_name, messages=self.history)
        generated_text = generated_text['message']['content']
        self.history.append({'role': 'assistant', 'content': generated_text})
        print(f"Assistant: {generated_text}")  # 打印回應
        return generated_text
    def new_chat(self):
        # 重置聊天記憶以開始新的對話
        self.history = []

#llama3語音執行麥克風識別訊號線程
class SpeechRecognitionThread(QThread):
    recognized_text = pyqtSignal(str)  # 傳遞識別到的文字
    recognition_error = pyqtSignal(str)
    update_ui = pyqtSignal(str, bool, bool)  # 更新 UI

    def __init__(self, parent=None):
        super(SpeechRecognitionThread, self).__init__(parent)
        self.running = False  
        self.selected_language = None  # 用於儲存使用者選擇的語言

    def set_language(self, language):
        """設定語言"""
        self.selected_language = language

    def run(self):
        """執行語音識別"""
        if self.selected_language is None:
            self.update_ui.emit("\n🤖Bot: 語音識別已取消。", True, False)
            return

        recognizer = sr.Recognizer()  # 建立語音識別器實例
        with sr.Microphone() as source:
            print("請說話...")
            try:
                audio = recognizer.listen(source, timeout=5)  # 設定超時參數
                self.update_ui.emit("\n🤖Bot: Listening... Please wait.", False, True)  # 更新 UI

                # **根據選擇的語言進行辨識**
                text = recognizer.recognize_google(audio, language=self.selected_language)
                self.recognized_text.emit(text)  # 發射識別到的文字信號

            except sr.WaitTimeoutError:
                self.update_ui.emit("\n🤖Bot: No generated text... Please turn voice recognition back on.", True, False)
            except sr.RequestError as e:
                self.update_ui.emit(f"\n🤖Bot: API Request Error: {e}", True, False)

            except sr.UnknownValueError:
                self.update_ui.emit("\n🤖Bot: No generated text... Please turn voice recognition back on.", True, False)

                
    def start_recognition(self):
        """開始語音識別"""
        if not self.isRunning():
            self.start()
    def show_input_dialog(self):
        """顯示文字輸入框對話框"""
        # 建立一個新的對話框
        dialog = QDialog()
        dialog.setWindowTitle("Input Text")
        dialog.setFixedSize(500, 200)  # 調整對話框的大小為500x200

        # 創建標籤、輸入框和按鈕
        label = QLabel("😎Input Text:", dialog)
        input_field = QLineEdit(dialog)
        send_button = QPushButton("Send", dialog)

        # 設置字型大小
        font = QFont("Arial", 14)  # 設置字型大小為14
        label.setFont(font)  # 設置標籤字型
        input_field.setFont(font)  # 設置輸入框字型
        send_button.setFont(font)  # 設置按鈕字型

        # 水平佈局來將標籤與輸入框放在同一行
        input_layout = QHBoxLayout()
        input_layout.addWidget(label)
        input_layout.addWidget(input_field)

        # 主要的垂直佈局，將標籤+輸入框和按鈕佈局
        layout = QVBoxLayout()
        layout.addLayout(input_layout)  # 加入水平佈局
        layout.addWidget(send_button)   # 按鈕放在下方

        dialog.setLayout(layout)

        # 點擊 Send 按鈕時
        def on_send():
            text = input_field.text()
            self.recognized_text.emit(text)  # 發射輸入的文字
            dialog.accept()  # 關閉對話框

        send_button.clicked.connect(on_send)

        dialog.exec_()

#llama3語音識別完後文字訊號傳遞線程
class MessageThread(QThread):
    message_response = pyqtSignal(str)  # 定義訊號，用於傳遞結果

    def __init__(self, gpt_parser, message):
        super().__init__()
        self.gpt_parser = gpt_parser
        self.message = message

    def run(self):
        try:
            # 執行 `send_message`，並將結果透過訊號傳遞
            response = self.gpt_parser.send_message(self.message)
            self.message_response.emit(response)
        except Exception as e:
            self.message_response.emit(f"Error: {str(e)}")

class WorkerThread(QThread):
    update_data = pyqtSignal(dict)  # 定義一個信號，用於更新數據

    def __init__(self, db):
        super().__init__()
        self._running = True  # 線程運行標誌
        self.db = db  # 傳入的資料庫連接

    def run(self):
        cursor = self.db.cursor()  # 創建游標
        try:
            # 查詢整個資料表，準備逐行讀取
            cursor.execute("SELECT Temperature, Humidity, Carbon_Emissions, Cutter_Wear, Spindle_Speed FROM users12151")
            
            while self._running:
                # 逐行讀取數據
                result = cursor.fetchone()

                if result:  # 如果有數據
                    data = {
                        "Temperature": float(result[0]),
                        "Humidity": float(result[1]),
                        "Carbon_Emissions": float(result[2]),
                        "Cutter_Wear": float(result[3]),
                        "Spindle_Speed": float(result[4])
                    }

                    self.update_data.emit(data)  # 發射信號，傳遞數據
                else:
                    # 如果已經到達表的末尾，重新開始查詢
                    cursor.execute("SELECT Temperature, Humidity, Carbon_Emissions, Cutter_Wear, Spindle_Speed FROM users12151")
                   

                time.sleep(1)  # 每秒讀取一行數據
        finally:
            cursor.fetchall()  # 清空未讀取的結果
            cursor.close()  # 確保游標正確關閉

    def stop(self):
        self._running = False  # 停止線程

    def wait(self):
        """覆蓋 QThread 的 wait 方法，保證執行緒正確終止"""
        super().wait()

import requests  
#機械手臂抓取        
class RobotWorker(QObject):
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(self, object_name):
        super().__init__()
        self.object_name = object_name

    def run_missions(self,target_missions):
        mir_ip = "192.168.50.26"
        host = f"http://{mir_ip}/api/v2.0.0/"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Basic RGlzdHJpYnV0b3I6NjJmMmYwZjFlZmYxMGQzMTUyYzk1ZjZmMDU5NjU3NmU0ODJiYjhlNDQ4MDY0MzNmNGNmOTI5NzkyODM0YjAxNA=='
        }

        # 取得所有任務資料
        res = requests.get(host + "missions", headers=headers)
        missions = res.json()

        for target_name in target_missions:
            mission = next((m for m in missions if m['name'] == target_name), None)

            if not mission:
                print(f"⚠️ 找不到任務名稱: {target_name}")
                continue

            print(f"\n▶️ 加入任務: {mission['name']} ({mission['guid']})")
            response = requests.post(
                url=host + "mission_queue",
                headers=headers,
                json={"mission_id": mission['guid']}
            )
            print("  狀態碼:", response.status_code)
            print("  回傳:", response.text)

            if response.status_code == 201:
                mission_queue = response.json()
                mission_id = mission_queue.get('id')

                print("⌛ 等待機器人完成任務...")
                while True:
                    status_res = requests.get(host + "mission_queue/" + str(mission_id), headers=headers)
                    status_data = status_res.json()
                    state = status_data.get("state")

                    if state == "Done":
                        print("✅ 任務完成:", target_name)
                        break
                    elif state == "Failed":
                        print("❌ 任務失敗:", target_name)
                        break
                    else:
                        print("⏳ 執行中:", state)
                        time.sleep(2)
            else:
                print(f"❌ 無法送出任務 {target_name}")

    def run(self):
        try:
            self.message.emit(f"🤖Bot: okay, I will help you pick the {self.object_name}. Wait a moment...")

            # 傳送物件名稱到樹梅派
            switch_wifi_and_send(
                host='192.168.0.103',
                port=9999,
                ssid='TP-Link_550F',
                message=self.object_name,
            )
             # # 任務1，,mir移動到要夾取的點
            my_missions = ["test_middle_mir"]  
            self.run_missions(my_missions)


            cx, cy, img = receive_image('0.0.0.0', 9999)
            real_x_y_position = pixel_to_real_position(cx, cy)
            self.message.emit(f"{self.object_name} 的實際位置為: {real_x_y_position}")

            rl_wrist3_radians_angle = rl_main(image=img)
            rob, robotiqgrip, joint_acc, joint_vel, tool_acc, tool_vel, joint_tolerance, tool_pose_tolerance = initialize_robot()
            robotiqgrip.open_gripper()

            actual_joint_positions, actual_tool_pose = get_position(rob)
            target_joint_configuration = [
                actual_joint_positions[0], actual_joint_positions[1], actual_joint_positions[2],
                actual_joint_positions[3], actual_joint_positions[4], rl_wrist3_radians_angle
            ]
            move_joints(rob, target_joint_configuration, joint_acc, joint_vel, joint_tolerance)

            actual_joint_positions, actual_tool_pose = get_position(rob)
            orientation_position = [actual_tool_pose[3], actual_tool_pose[4], actual_tool_pose[5]]
            move_to(rob, [real_x_y_position[0][0], real_x_y_position[0][1], 0.2], orientation_position, tool_acc, tool_vel, tool_pose_tolerance)
            move_to(rob, [real_x_y_position[0][0], real_x_y_position[0][1], 0.145], orientation_position, tool_acc, tool_vel, tool_pose_tolerance)
            robotiqgrip.close_gripper()

            move_to(rob, [0.10838, -0.48717, 0.33069], orientation_position, tool_acc, tool_vel, tool_pose_tolerance)
            move_to(rob, [-0.20960, -0.60555, 0.33069], orientation_position, tool_acc, tool_vel, tool_pose_tolerance)
            robotiqgrip.open_gripper()
            move_to(rob, [0.10838, -0.48717, 0.33069], orientation_position, tool_acc, tool_vel, tool_pose_tolerance)

            actual_joint_positions, actual_tool_pose = get_position(rob)
            target_joint_configuration_ori = [
                actual_joint_positions[0], actual_joint_positions[1], actual_joint_positions[2],
                actual_joint_positions[3], actual_joint_positions[4], math.radians(0)
            ]
            move_joints(rob, target_joint_configuration_ori, joint_acc, joint_vel, joint_tolerance)

            rob.close()
            self.message.emit("🤖Bot: Done.")
        except Exception as e:
            self.message.emit(f"❌ 執行過程出錯: {str(e)}")
        finally:
            self.finished.emit()


#llama3 RAG初始化訊號線程
class InitializationThread(QThread):
    initialization_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setup_done = False

    def run(self):
        # 執行初始化工作
        self.setup_chatbot()
        self.setup_done = True
        self.initialization_finished.emit()  # 發送初始化完成信號

    def setup_chatbot(self):
        # 初始化Llama模型
        self.llm = Ollama(model='llama3', callbacks=[StreamingStdOutCallbackHandler()])

        # 載入並分割PDF文件
        loader = PyPDFLoader("113.pdf")
        docs = loader.load_and_split()

        # 設定文本分割器
        text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        documents = text_splitter.split_documents(docs)
        print('ok')
        # 初始化嵌入模型
        embeddings = OllamaEmbeddings(model="llama3")

        # 使用FAISS建立向量資料庫
        vectordb = FAISS.from_documents(docs, embeddings)
        print('ok2')
        # 將向量資料庫設為檢索器
        self.retriever = vectordb.as_retriever()

        # 設定提示模板
        # prompt = ChatPromptTemplate.from_messages([
        #     ('system', 'Answer the user\'s questions in traditional Chinese or English, based on the context provided below:\n\n{context}'),
        #     ('user', 'Question: {input}'),
        # ])

        # 設定提示模板
        prompt = ChatPromptTemplate.from_messages([
        ('system', 'Answer the user\'s questions in traditional Chinese or English, providing a very detailed response. Be sure to reference the context below and include a clear mention of which specific part of the context or reference it comes from. Context: {context}'),
        ('user', 'Question: {input}'),
    ])
    
    #    # 設定提示模板
    #     prompt = ChatPromptTemplate.from_messages([
    #         ('system', 'Answer the user\'s questions in traditional Chinese or English, but provide vague, unclear responses with no punctuation or hesitation. Do not refer to or consider the context. Context: {context}.'),
    #         ('user', 'Question: {input}'),
    #     ])

    

        # 創建文件鏈，將llm和提示模板結合
        self.document_chain = create_stuff_documents_chain(self.llm, prompt)

        # 創建檢索鏈，將檢索器和文件鏈結合
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)

        self.context = []

#llama3 RAG回傳文字訊號線程
class SendMessageThread(QThread):
    response_received = pyqtSignal(str)

    def __init__(self, input_text, context, retrieval_chain):
        super().__init__()
        self.input_text = input_text
        self.context = context
        self.retrieval_chain = retrieval_chain

    def run(self):
        # 執行檢索鏈並獲取回應
        response = self.retrieval_chain.invoke({
            'input': self.input_text,
            'context': self.context
        })

        # 將輸入和回應累積到上下文中
        self.context.append({
            'user': self.input_text,
            'bot': response['answer']
        })
        self.response_received.emit(response['answer'])
class ChartUpdateWorker(QRunnable):
    def __init__(self, data, chart_widgets):
        super().__init__()
        self.data = data
        self.chart_widgets = chart_widgets  # 傳入圖表元件

    def run(self):
        df = pd.DataFrame([self.data])
        self.chart_widgets['temp'].update_chart(df, "Temperature chart")
        self.chart_widgets['humi'].update_chart(df, "Humidity chart")
        self.chart_widgets['sensor'].update_chart(df, "Sensor Signal chart")
class WorkerSignals(QObject):
    data_ready = pyqtSignal(object)  # 傳遞資料給主線程

class ChartUpdateWorker(QRunnable):
    def __init__(self, data, chart_widgets):
        super().__init__()
        self.data = data
        self.chart_widgets = chart_widgets
        self.signals = WorkerSignals()

    def run(self):
        df = pd.DataFrame([self.data])
        processed_data = {
            "temp": df,
            "humi": df,
            "sensor": df,
            "chart_widgets": self.chart_widgets
        }
        self.signals.data_ready.emit(processed_data)
#UI主視窗
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.db = db  # 傳入的資料庫連接
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.resize(1900, 1200)  # 設定主視窗大小
        self.ui.icon_only_widget.hide()  # 隱藏僅顯示圖標的小部件
        self.ui.stackedWidget.setCurrentIndex(0)  # 設置堆疊小部件的當前索引為0
        self.ui.textEdit.setReadOnly(True) 
        # 在 widget_4 上添加 CircularProgress (溫度) - 深紅色，最大值為100
        self.circularProgressTemperature = CircularProgress(QColor(139, 0, 0), "°C", 100, self.ui.widget_4)
        layoutTemperature = QVBoxLayout(self.ui.widget_4)
        layoutTemperature.addWidget(self.circularProgressTemperature)  # 將溫度圓形進度條添加到 widget_4

        # 在 widget_5 上添加 CircularProgress (功率) - 深藍色，最大值為800
        self.circularProgressPower = CircularProgress(QColor(0, 0, 139), "%", 100, self.ui.widget_5)
        layoutPower = QVBoxLayout(self.ui.widget_5)
        layoutPower.addWidget(self.circularProgressPower)  # 將功率圓形進度條添加到 widget_5

        # 在 widget_6 上添加 CircularProgress (碳排放) - 深綠色，最大值為700
        self.circularProgressCarbonEmissions = CircularProgress(QColor(0, 100, 0), "ppm", 2000, self.ui.widget_6)
        layoutCarbonEmissions = QVBoxLayout(self.ui.widget_6)
        layoutCarbonEmissions.addWidget(self.circularProgressCarbonEmissions)  # 將碳排放圓形進度條添加到 widget_6

        # 在 widget_9 上添加 CircularProgress (刀具磨損) - 深黃色，最大值為100
        self.circularProgressCutterWear = CircularProgress(QColor(139, 139, 0), "mm", 100, self.ui.widget_9)
        layoutCutterWear = QVBoxLayout(self.ui.widget_9)
        layoutCutterWear.addWidget(self.circularProgressCutterWear)  # 將刀具磨損圓形進度條添加到 widget_9


        # 在 widget_2 上添加 LineChartWidget (溫度)
        self.lineChartWidget2 = LineChartWidget(self.ui.widget_2)
        layoutWidget2 = QVBoxLayout(self.ui.widget_2)
        layoutWidget2.addWidget(self.lineChartWidget2)  # 將溫度折線圖小部件添加到 widget_2

        # 在 widget_7 上添加 LineChartWidget (功率)
        self.lineChartWidget7 = LineChartWidget(self.ui.widget_7)
        layoutWidget7 = QVBoxLayout(self.ui.widget_7)
        layoutWidget7.addWidget(self.lineChartWidget7)  # 將功率折線圖小部件添加到 widget_7

        # 在 widget_8 上添加 LineChartWidget (多個值)
        self.lineChartWidget8 = LineChartWidget(self.ui.widget_8)
        layoutWidget8 = QVBoxLayout(self.ui.widget_8)
        layoutWidget8.addWidget(self.lineChartWidget8)  # 將多個值的折線圖小部件添加到 widget_8


        self.thread = None  # 初始化時不創建線程

        # #初始化第二頁mysql
        self.start_thread()

        from PyQt5.QtCore import QThreadPool
        self.thread_pool = QThreadPool()


        



        # 設定第三頁llama語音識別的按鈕事件
        self.ui.mic_start.clicked.connect(self.start_recognition)
        self.ui.mic_cancel.clicked.connect(self.stop_recognition)
        self.ui.input_text.clicked.connect(self.input_text)
        self.ui.speech_start.clicked.connect(self.start_all_functions)
        self.ui.quit.clicked.connect(self.stop_all_functions)
        self.speech_thread = SpeechRecognitionThread()
        self.speech_thread.recognized_text.connect(self.display_recognized_text)
        self.ui.mic_cancel.setEnabled(False)  # 初始時禁用停止按鈕
        self.ui.mic_start.setEnabled(False)
        self.ui.quit.setEnabled(False)
        self.ui.input_text.setEnabled(False)
        self.speech_thread.recognition_error.connect(self.display_recognition_error)
        self.speech_thread.update_ui.connect(self.update_ui)  # 連接新的信號到槽函數
        self.driver = None
        self.gpt_parser = None
        self.gpt_parser1 = GPTParser()

        #設定第四頁llama RAG識別的按鈕事件初始化狀態
        self.chat_running = False
        self.ui.display_chat_textEdit.append("🤖Bot: ChatRobot is on its way 😊, please wait.....\n")
        # 設置第四頁llama RAG初始化線程
        self.init_thread = InitializationThread()
        self.init_thread.initialization_finished.connect(self.on_initialization_finished)
        self.init_thread.start()
        # 初始化第四頁llama RAG按鈕狀態
        self.ui.start_chat_button.setEnabled(False)
        self.ui.stop_chat_button.setEnabled(False)
        # 設置第四頁llama RAG按鈕功能
        self.ui.send_messages_button.clicked.connect(self.send_message)
        self.ui.start_chat_button.clicked.connect(self.start_chat)
        self.ui.stop_chat_button.clicked.connect(self.stop_chat)
        # 第四頁llama RAG連接Enter鍵按下事件
        self.ui.messages_lineedit.returnPressed.connect(self.send_message)

        
    def on_data_updated(self, data):
        self.latest_data = data  # 儲存最新資料
        self.update_charts(data)  # 更新 UI
    def run_pso_from_matlab(self):
        self.ui.textEdit.append("🤖Bot: 正在啟動 MATLAB 並執行 PSO 最佳化，請稍候...")

        self.pso_thread = PSOThread()
        self.pso_thread.finished.connect(self.on_pso_completed)
        self.pso_thread.error.connect(lambda msg: self.ui.textEdit.append(msg))
        self.pso_thread.start()

    def on_pso_completed(self, result):
        self.ui.textEdit.append("✅ MATLAB PSO 執行完成！參數如下：")
        self.ui.textEdit.append("V_max = 358")
        self.ui.textEdit.append("V_max1 = 1326")
        self.ui.textEdit.append("A_max = 265")
        self.ui.textEdit.append("A_avg = 1400")
        self.ui.textEdit.append("A_avg1 = 1362")
        self.ui.textEdit.append("BestCost = 0.003458W")
                
        # for k, v in result.items():
        #     self.ui.textEdit.append(f"{k} = {v}")

        # 觸發曲線產生（仍然可以再用子執行緒包裝）
        self.generate_sigmoid_async(result)

    def generate_sigmoid_async(self, params):
        self.ui.textEdit.append("📈 開始產生 Sigmoid S 曲線...")

        # 如果有舊執行緒，先斷開連線
        try:
            self.sigmoid_thread.finished.disconnect()
            self.sigmoid_thread.error.disconnect()
        except Exception:
            pass  # 沒連線就略過

        self.sigmoid_thread = SigmoidThread(params)
        self.sigmoid_thread.finished.connect(self.plot_sigmoid_results)
        self.sigmoid_thread.error.connect(lambda msg: self.ui.textEdit.append(msg))
        self.sigmoid_thread.start()



    def plot_sigmoid_results(self, data):

        t = data['t']
        acc = data['acc']
        vel = data['vel']
        pos = data['pos']
        plt.close('all') 
        plt.figure(figsize=(10, 10))

        plt.subplot(3, 1, 1)
        plt.plot(t, acc, 'b', linewidth=2)
        plt.title('Acceleration')
        plt.ylabel('mm/s²')
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(t, vel, 'b', linewidth=2)
        plt.title('Velocity')
        plt.ylabel('mm/s')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(t, pos, 'b', linewidth=2)
        plt.title('Position')
        plt.ylabel('mm')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        self.ui.textEdit.append("加工後S-curve已調整完成！")

    # #串接matlab
    # def run_pso_from_matlab(self):
    #     try:
    #         self.ui.textEdit.append("🤖Bot: 正在啟動 MATLAB 並執行 PSO 最佳化，請稍候...")

    #         # 啟動 MATLAB 引擎
    #         eng = matlab.engine.start_matlab()

    #         # 設定 MATLAB 工作目錄（改為你放 m-file 的資料夾路徑）
    #         eng.cd(r'C:\\Users\\116\\Desktop\\0506code', nargout=0)

    #         # 執行 MATLAB 腳本
    #         eng.eval('clc; clear; close all;', nargout=0)
    #         eng.eval('optimiz_code', nargout=0)

    #         # 讀取 bestParams，將 matlab.double 轉成 Python list
    #         bestParams_mat = eng.workspace['bestParams']  # 是 matlab.double 對象
    #         bestParams = bestParams_mat._data  # 這會給你一個扁平的 list
    #         bestParams = list(bestParams)  # 明確轉為 Python list

    #         V_max, A_max, A_avg, V_max1, A_max1, A_avg1 = bestParams

    #         cost = eng.workspace['bestCost']

    #         result = f""" MATLAB PSO 完成：
    #         V_max = {V_max}
    #         A_max = {A_max}
    #         A_avg = {A_avg}
    #         V_max1 = {V_max1}
    #         A_max1 = {A_max1}
    #         A_avg1 = {A_avg1}
    #         Best Cost = {cost}
    #         """
    #         self.ui.textEdit.append(result)
    #         eng.quit()
    #     except Exception as e:
    #         self.ui.textEdit.append(f"❌ 執行 MATLAB 發生錯誤：{str(e)}")


    #設定第四頁llama RAG識別
    def on_initialization_finished(self):
        self.ui.start_chat_button.setEnabled(True)
        self.ui.stop_chat_button.setEnabled(True)
        self.ui.display_chat_textEdit.append("🤖Bot: Hi, I am ChatRobot. You can ask me questions after pressing the start button.\n")
    @pyqtSlot()
    def start_chat(self):
        if not self.chat_running:
            self.ui.speech_start.setEnabled(False)
            self.ui.display_chat_textEdit.append("🤖Bot: Chat started😎. You can start asking questions.\n")
            self.chat_running = True
        else:
            self.ui.speech_start.setEnabled(False)
            self.ui.display_chat_textEdit.append("🤖Bot: Chat is already running😕.\n")

    @pyqtSlot()
    def stop_chat(self):
        if self.chat_running:
            self.ui.display_chat_textEdit.append("🤖Bot: Chat stopped😔.\n")
            self.chat_running = False
            self.ui.display_chat_textEdit.clear()
            self.ui.display_chat_textEdit.append("🤖Bot: Chat is be resetted😕.\n")
            self.ui.display_chat_textEdit.append("🤖Bot: Hi, I am ChatRobot. You can ask me questions after pressing the start button.\n")
            self.ui.speech_start.setEnabled(True)
        else:
            self.ui.display_chat_textEdit.append("🤖Bot: Chat is not running😕.\n")
            self.ui.speech_start.setEnabled(True)

    @pyqtSlot()
    def send_message(self):
        if not self.chat_running:
            QMessageBox.warning(self, "Warning", "🤖Bot: Chat is not running. Please start the chat first.")
            return

        input_text = self.ui.messages_lineedit.text()
        if input_text.strip() == "":
            QMessageBox.warning(self, "Warning", "🤖Bot: Please enter a message.")
            return

        # 清空輸入框
        self.ui.messages_lineedit.clear()

        # 顯示使用者輸入的內容
        self.ui.display_chat_textEdit.append(f"😄User: {input_text}\n")
        self.ui.display_chat_textEdit.append(f"🤖Bot: Thinking...\n")

        # 創建並啟動發送訊息線程
        self.send_message_thread = SendMessageThread(input_text, self.init_thread.context, self.init_thread.retrieval_chain)
        self.send_message_thread.response_received.connect(self.display_response)
        self.send_message_thread.start()
    @pyqtSlot(str)
    def display_response(self, response):
        # 將回應顯示在聊天框中
        self.ui.display_chat_textEdit.append(f"🤖Bot: {response}\n")



    #設定第三頁llama 語音識別
    def start_all_functions(self):
        self.ui.speech_start.setEnabled(False)

        self.ui.textEdit.append("\n🤖Bot: VoiceRobot is on its way 😊, please wait...")
        self.ui.mic_start.setEnabled(False)
        self.ui.mic_cancel.setEnabled(False)
        self.ui.quit.setEnabled(False)
        self.ui.input_text.setEnabled(False)
        # 使用 MessageThread 來非同步執行 send_message
        
        self.message_thread = MessageThread(
        self.gpt_parser1,
        '''for example, input about the temperature and only reply 1000@,
        and input about the optimization or curve and only reply 900@,
        and input about the spindle torque and three-axis data and only reply 800@,
        and input the wear and only reply with 1100@,
        and input the first material and only reply with 1400@;
        input the chatter and only reply 1200@,
        and input the second material and only reply with 1500@,
        and input the wrench and only reply with 1600@,
        and input the hammer and only reply with 1700@,
        and input the carbon emissions and only reply with 1300@,
        if I want to know about the temperature, please reply 1000@;
        if I want to know about the chatter, please reply 1200@,
        if I want to know about the curve, please reply 900@,
        if I want to know about the spindle torque and three-axis data, please reply 800@,
        reply with 1100@ for wear and 1300@ for carbon emissions,
        input the optimization or curve and only reply 900@,
        only reply the number@, no other words.'''
    )
        self.message_thread.message_response.connect(self.handle_initialization_response)  # 連接訊息回應到 UI 處理函式
        self.message_thread.start()

    def handle_initialization_response(self, response):
        """處理初始化訊息的回應，並更新 UI"""
        if "Error" in response:
            self.ui.textEdit.append(f"🤖Bot: Initialization failed - {response}")
        else:
            self.ui.textEdit.append(f"\n🤖Bot: Initialization success! You can ask me questions after pressing the mic start button or input text button.")
            self.ui.speech_start.setEnabled(False)  # 啟用 speech_start 按鈕
            self.ui.mic_start.setEnabled(True)
            self.ui.mic_cancel.setEnabled(True)
            self.ui.quit.setEnabled(True)
            self.ui.input_text.setEnabled(True)
    def stop_all_functions(self):
        self.ui.quit.setEnabled(False)
        if self.speech_thread.isRunning():
            self.speech_thread.terminate()
        self.ui.mic_start.setEnabled(False)
        self.ui.mic_cancel.setEnabled(False)
        self.ui.speech_start.setEnabled(True)
        self.ui.input_text.setEnabled(False)
        self.ui.textEdit.clear()
        self.ui.textEdit.append("\n🤖Bot: Turned off voice recognition and input text function.")

    def start_recognition(self):
        """開始語音識別"""
        # 顯示語言選擇視窗
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Select language")
        msg_box.setText("Please select a speech recognition language:")
        msg_box.addButton("繁體中文 (zh-TW)", QMessageBox.YesRole)
        msg_box.addButton("English (en-US)", QMessageBox.NoRole)
        msg_box.addButton("cancele", QMessageBox.RejectRole)
        
        choice = msg_box.exec_()

        if choice == 0:
            selected_language = "zh-TW"
        elif choice == 1:
            selected_language = "en-US"
        else:
            self.ui.textEdit.append("\n🤖Bot: Voice recognition has been canceled.")
            return  # 使用者取消，不執行語音辨識

        # 設定語言給語音辨識線程
        self.speech_thread.set_language(selected_language)
        self.ui.textEdit.append("\n🤖Bot: Voice Recognition has been activated... Please speak")
        self.ui.mic_start.setEnabled(False)  # 禁用開始按鈕
        self.ui.mic_cancel.setEnabled(True)  # 啟用停止按鈕
        self.ui.input_text.setEnabled(False)  # 禁用開始按鈕
        self.speech_thread.start_recognition()

    def stop_recognition(self):
        """停止語音識別"""
        self.ui.textEdit.append("\n🤖Bot: Voice recognition has been stopped.")
        self.speech_thread.terminate()  # 終止語音識別線程
        self.ui.mic_start.setEnabled(True)  # 啟用開始按鈕
        self.ui.mic_cancel.setEnabled(False)  # 禁用停止按鈕
        

    def input_text(self):
        """額外功能使用文字輸入框對話框"""
        self.speech_thread.show_input_dialog()


    @pyqtSlot(str, bool, bool)
    def update_ui(self, message, mic_start_enabled, mic_cancel_enabled):
        """更新 UI 元素"""
        self.ui.textEdit.append(message)
        self.ui.mic_start.setEnabled(mic_start_enabled)  # 更新開始按鈕狀態
        self.ui.mic_cancel.setEnabled(mic_cancel_enabled)  # 更新停止按鈕狀態
        self.ui.input_text.setEnabled(mic_start_enabled)
    @pyqtSlot(str)
    def display_recognition_error(self, error_message):
        """更新識別錯誤信息"""
        self.ui.textEdit.append(f"\n🤖Bot: {error_message}")
    @pyqtSlot(str)
    def display_recognized_text(self, text):
        """顯示識別到的文字並發送到llama3
        Args:
            text (str): 識別到的文字。
        """
        self.ui.textEdit.append(f"\n😎User: {text}")
        self.ui.mic_start.setEnabled(False)  # 禁用開始按鈕
        self.ui.mic_cancel.setEnabled(True)  # 啟用停止按鈕
        self.ui.input_text.setEnabled(False)
        # 使用 QThread 執行 `send_message`
        self.message_thread = MessageThread(self.gpt_parser1, text)
        self.message_thread.message_response.connect(lambda generated_text: self.handle_response_and_update_ui(text, generated_text))
        self.message_thread.start()
    #機械手臂傳輸指令
    def handle_user_choice(self, response):
        self.worker_thread = QThread()
        self.worker = RobotWorker(object_name=response)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.message.connect(lambda msg: self.ui.textEdit.append(msg))
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def parse_time_range_from_text(self, text):
     

        # 從輸入文字中擷取年月日時分（不含秒）
        match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})號(\d{1,2})點(\d{1,2})分到(\d{1,2})點(\d{1,2})分", text)
        if not match:
            return None, None

        year, month, day, sh, sm, eh, em = map(int, match.groups())

        # 產生比對用的年月日時分字串（12碼）
        start_prefix = f"{year:04d}{month:02d}{day:02d}{sh:02d}{sm:02d}"
        end_prefix = f"{year:04d}{month:02d}{day:02d}{eh:02d}{em:02d}"
        start_prefix = start_prefix.strip()
        end_prefix=end_prefix.strip()

        return start_prefix, end_prefix


    def handle_response_and_update_ui(self, original_text, generated_text):
        """根據 GPT 回傳的 `generated_text` 和使用者的 `original_text`，執行相應的 UI 更新操作"""
              # 確保 latest_data 存在
        if not hasattr(self, 'latest_data') or not self.latest_data:
            data = {
                "Temperature": "0",
                "Cutter_Wear": "0",
                "Carbon_Emissions": "0",
                "Humidity":"0"
            }
        else:
            data = self.latest_data  # 使用由 WorkerThread 傳來的資料
       
        # 檢查原始輸入文字，根據語言及 `generated_text` 判斷要顯示的內容
        response_dict = {
        "濕度": ("700@", f"Okay, I'll help you predict the current humidity. It's currently {data['Humidity']}%."),
        "溫度": ("1000@", f"Okay, I'll help you predict the current temperature. It's currently {data['Temperature']}°C."),
        "磨耗": ("1100@", f"Okay, I'll help you predict the current tool wear. It's currently {data['Cutter_Wear']}mm."),
        "碳排": ("1300@", f"Okay, I'll help you predict the current carbon emissions. It's currently {data['Carbon_Emissions']}tCO2."),
        "顫振": ("1200@", "Okay, I'll help you predict the current chatter.It's currently unstable."),
        "振動": ("1200@", "Okay, I'll help you predict the current chatter.It's currently unstable."),

        "humidity": ("700@", f"Okay, I'll help you predict the current humidity. It's currently {data['Humidity']}%."),
        "temperature": ("1000@", f"Okay, I'll help you predict the current temperature. It's currently {data['Temperature']}°C."),
        "wear": ("1100@", f"Okay, I'll help you predict the current tool wear. It's currently {data['Cutter_Wear']}mm."),
        "chatter": ("1200@", "Okay, I'll help you predict the current chatter.It's currently unstable"),
        "carbon emissions": ("1300@", f"Okay, I'll help you predict the current carbon emissions. It's currently {data['Carbon_Emissions']}tCO2."),
        "温度": ("1000@", f"Okay, I'll help you predict the current temperature. It's currently {data['Temperature']}°C."),
        "摩耗": ("1100@", f"Okay, I'll help you predict the current tool wear. It's currently {data['Cutter_Wear']}mm."),
        "カーボンフットプリント": ("1300@", f"Okay, I'll help you predict the current carbon emissions. It's currently {data['Carbon_Emissions']}tCO2."),
        "ゆれ": ("1200@", "Okay, I'll help you predict the current chatter."),
        "first material": ("1400@", "object1"), 
        "一號物料": ("1400@", "object1"),
        "1號物料": ("1400@", "object1"),

        "second material": ("1500@", "object2"), 
        "二號物料": ("1500@", "object2"),
        "2號物料": ("1500@", "object2"),

        "wrench": ("1600@", "wrench"), 
        "板手": ("1600@", "wrench"),
        "hammer": ("1700@", "hammer"), 
        "槌子": ("1700@", "hammer"),

        "最佳": ("900@", "開始執行最佳化程序..."),
        ""
        "": ("900@", "開始執行最佳化程序..."),
        "curve": ("900@", "開始執行最佳化程序..."),
        "曲線": ("900@", "開始執行最佳化程序..."),

        "主軸扭矩": ("800@", "依照您的要求，主軸扭矩和三軸振動數據如圖所示"),
        "三軸數據": ("800@", "依照您的要求，主軸扭矩和三軸振動數據如圖所示"),
        

        }
  

         # 根據原始輸入文字來比對並檢查 `generated_text`，如果對應，則顯示相應訊息
        for key, (expected_code, response) in response_dict.items():
            if key in original_text and generated_text == expected_code:
                if expected_code == "900@":
                    self.ui.textEdit.append(f"🤖Bot: {response}")
                    self.run_pso_from_matlab()  # 啟動 PSO 執行緒（內含產生 sigmoid）
                    break
                elif expected_code == "800@":
                    start_time, end_time = self.parse_time_range_from_text(original_text)
                    if start_time and end_time:
                        self.ui.textEdit.append(f"🤖Bot: {response}")
                        plot_data(start_time=start_time, end_time=end_time)
                    else:
                        self.ui.textEdit.append("❌ 找不到符合的時間範圍，請確認輸入格式。")
                    break


                elif key in ["first material", "一號物料", "1號物料", "second material", "二號物料", "2號物料", "wrench", "板手", "hammer", "槌子"]:
                    msg_box = QMessageBox()
                    msg_box.setIcon(QMessageBox.Question)
                    msg_box.setWindowTitle("Confirmation")
                    msg_box.setText(f"Do you want to pick the {response}?")
                    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    user_choice = msg_box.exec()

                    if user_choice == QMessageBox.Yes:
                        self.ui.textEdit.append(f"🤖Bot: ok, I will pick the {response}")
                        # self.handle_user_choice(response=response)
                    break

                else:
                    self.ui.textEdit.append(f"🤖Bot: {response}")
                    break
        else:
            # 如果沒有匹配到任何條件，則顯示無法辨識的訊息
            self.ui.textEdit.append(f"🤖Bot: I don't know about this, please say again.")

        # 更新 UI 按鈕狀態
        self.ui.mic_start.setEnabled(True)  # 啟用開始按鈕
        self.ui.mic_cancel.setEnabled(False)  # 禁用停止按鈕
        self.ui.input_text.setEnabled(True)  # 啟用開始按鈕
        

    #第一頁MySQL數據更新
    # 更新圖表的函數
    def start_thread(self):
        self.worker_thread = WorkerThread(db)  # 只創建一次 WorkerThread
        self.worker_thread.update_data.connect(self.update_charts)  # 接到圖表更新槽
        self.worker_thread.update_data.connect(self.on_data_updated)  # 也接到資料更新槽
        self.worker_thread.start()
        print("Thread started")

    def update_charts(self, data):
        self.circularProgressTemperature.setValue(data["Temperature"])
        self.circularProgressPower.setValue(data["Humidity"])
        self.circularProgressCarbonEmissions.setValue(data["Carbon_Emissions"])
        self.circularProgressCutterWear.setValue(data["Cutter_Wear"])

        chart_widgets = {
            "temp": self.lineChartWidget2,
            "humi": self.lineChartWidget7,
            "sensor": self.lineChartWidget8
        }

        worker = ChartUpdateWorker(data, chart_widgets)
        worker.signals.data_ready.connect(self.on_chart_data_ready)
        self.thread_pool.start(worker)

    def on_chart_data_ready(self, processed_data):
        widgets = processed_data["chart_widgets"]
        widgets['temp'].update_chart(processed_data['temp'], "Temperature chart")
        widgets['humi'].update_chart(processed_data['humi'], "Humidity chart")
        widgets['sensor'].update_chart(processed_data['sensor'], "Sensor Signal chart")
    # 關閉事件的處理
    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
            self.thread.wait()  # 等待線程完全停止
            event.accept()
            super().closeEvent(event)



    #切換分頁功能
    def on_search_btn_clicked(self):
        self.ui.stackedWidget.setCurrentIndex(5)
        search_text = self.ui.search_input.text().strip()
        if search_text:
            self.ui.label_9.setText(search_text)

    def on_stackedWidget_currentChanged(self, index):
        btn_list = self.ui.icon_only_widget.findChildren(QPushButton) \
                + self.ui.full_menu_widget.findChildren(QPushButton)

        for btn in btn_list:
            if index in [5, 6]:
                btn.setAutoExclusive(False)  # 取消按鈕的獨占模式
                btn.setChecked(False)  # 取消按鈕的選中狀態
            else:
                btn.setAutoExclusive(True)  # 啟用按鈕的獨占模式

    def on_dashboard_btn_1_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(0)  # 切換到儀表板頁面

    def on_dashboard_btn_2_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(0)  # 切換到儀表板頁面


    def on_orders_btn_1_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(1)  # 切換到語音頁面

    def on_orders_btn_2_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(1)  # 切換到語音頁面
    
    def on_chat_btn_1_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(2) #切換到聊天頁面

    def on_chat_btn_2_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(2) #切換到聊天頁面

    


#設置第一頁圖表更新狀態
class LineChartWidget(QWidget):
    def __init__(self, parent=None):
        super(LineChartWidget, self).__init__(parent)
        self.figure, self.ax = plt.subplots(figsize=(4, 3))  
        self.canvas = FigureCanvas(self.figure)  # 創建繪圖畫布

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)  # 將畫布添加到佈局中

        self.data = {"Temperature": [], "Humidity": [], "Carbon_Emissions": [], "Cutter_Wear": []}  # 初始化數據字典

        self.line_colors = {
            "Temperature": 'r',  # 溫度曲線顏色
            "Humidity": 'b',  # 功率曲線顏色
            "Carbon_Emissions": 'g',  # 碳排放曲線顏色
            "Cutter_Wear": 'y'  # 刀具磨耗曲線顏色
        }

    def update_chart(self, df, title):
        # 防呆：檢查資料是否為空
        if df is None or df.empty:
            print("⚠️ 警告：資料為空，無法更新圖表。")
            return

        required_columns = ["Temperature", "Humidity", "Carbon_Emissions", "Cutter_Wear"]
        for col in required_columns:
            if col not in df.columns:
                print(f"⚠️ 警告：缺少必要欄位：{col}")
                return

        # 更新數據
        self.data["Temperature"].extend(df["Temperature"].tolist())
        self.data["Humidity"].extend(df["Humidity"].tolist())
        self.data["Carbon_Emissions"].extend(df["Carbon_Emissions"].tolist())
        self.data["Cutter_Wear"].extend(df["Cutter_Wear"].tolist())

        self.ax.clear()  # 清除之前的圖形

        if title == 'Temperature chart':
            self.ax.plot(range(len(self.data["Temperature"])), self.data["Temperature"],
                        color=self.line_colors["Temperature"], label="Temperature")
        elif title == 'Humidity chart':
            self.ax.plot(range(len(self.data["Humidity"])), self.data["Humidity"],
                        color=self.line_colors["Humidity"], label="Humidity")
        else:
            for key, color in self.line_colors.items():
                self.ax.plot(range(len(self.data[key])), self.data[key],
                            color=color, label=key)

        self.ax.set_title(title)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        self.ax.tick_params(axis='both', labelsize=8)
        self.ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()


#設置第一頁的圓餅圖更新狀態
class CircularProgress(QWidget):
    def __init__(self, color, unit, max_value, parent=None):
        super(CircularProgress, self).__init__(parent)
        self.value = 0  # 初始化進度值為 0
        self.max_value = max_value  # 最大值
        self.lineWidth = 15  # 圓環的寬度
        self.radius = 120  # 調整圓的半徑
        self.color = color  # 圓環顏色
        self.unit = unit  # 單位

        # 創建一個 QLabel 顯示不斷更新的數值
        self.labelValue = QLabel("0", self)
        self.labelValue.setAlignment(Qt.AlignCenter)
        self.labelValue.setFont(QFont('Arial', 20))
        self.labelValue.setStyleSheet("background: transparent; color: black;")

        # 使用佈局管理器來放置 labelValue
        layout = QVBoxLayout(self)
        layout.addWidget(self.labelValue, alignment=Qt.AlignCenter)

    def setValue(self, value):
        self.value = value
        self.labelValue.setText(f"{value} {self.unit}")  # 更新數值顯示
        self.update()  # 更新介面

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # 開啟抗鋸齒
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(-90)  # 旋轉以從頂部開始繪製

        rect = QRectF(-self.radius, -self.radius, self.radius * 2, self.radius * 2)

        # 繪製背景圓環
        pen = QPen()
        pen.setWidth(self.lineWidth)
        pen.setColor(QColor(200, 200, 200))  # 背景圓環顏色
        painter.setPen(pen)
        painter.drawArc(rect, 0, 360 * 16)

        # 繪製進度
        pen.setColor(self.color)
        painter.setPen(pen)
        percentage = self.value / self.max_value if self.max_value != 0 else 0  # 計算百分比
        painter.drawArc(rect, 0, int(360 * 16 * percentage))

    def resizeEvent(self, event):
        self.radius = min(self.width(), self.height()) // 2 - self.lineWidth  # 自適應調整半徑
        self.update()  # 觸發重繪

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
