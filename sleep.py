import sys
import os
import time
import math
import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

# ===== pygame (sound) =====
import pygame
pygame.init()
pygame.mixer.init()

# ===== MediaPipe =====
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

# =========================
# 당신이 만든 계산 함수들
# =========================
def calculate_ear(landmarks, eye_indices):
    left_point = landmarks[eye_indices[0]]
    right_point = landmarks[eye_indices[3]]
    top_mid = ((landmarks[eye_indices[1]].x + landmarks[eye_indices[2]].x) / 2,
               (landmarks[eye_indices[1]].y + landmarks[eye_indices[2]].y) / 2)
    bottom_mid = ((landmarks[eye_indices[4]].x + landmarks[eye_indices[5]].x) / 2,
                  (landmarks[eye_indices[4]].y + landmarks[eye_indices[5]].y) / 2)

    horizontal_length = ((left_point.x - right_point.x) ** 2 + (left_point.y - right_point.y) ** 2) ** 0.5
    vertical_length = ((top_mid[0] - bottom_mid[0]) ** 2 + (top_mid[1] - bottom_mid[1]) ** 2) ** 0.5
    return vertical_length / horizontal_length if horizontal_length != 0 else 0.0


def calculate_mar(landmarks, mouth_indices):
    top_mid = ((landmarks[mouth_indices[0]].x + landmarks[mouth_indices[1]].x) / 2,
               (landmarks[mouth_indices[0]].y + landmarks[mouth_indices[1]].y) / 2)
    bottom_mid = ((landmarks[mouth_indices[2]].x + landmarks[mouth_indices[3]].x) / 2,
                  (landmarks[mouth_indices[2]].y + landmarks[mouth_indices[3]].y) / 2)
    left_point = landmarks[mouth_indices[4]]
    right_point = landmarks[mouth_indices[5]]

    horizontal_length = ((left_point.x - right_point.x) ** 2 + (left_point.y - right_point.y) ** 2) ** 0.5
    vertical_length = ((top_mid[0] - bottom_mid[0]) ** 2 + (top_mid[1] - bottom_mid[1]) ** 2) ** 0.5
    return vertical_length / horizontal_length if horizontal_length != 0 else 0.0


def calculate_head_tilt(landmarks):
    left_shoulder = landmarks[234]
    right_shoulder = landmarks[454]

    dx = right_shoulder.x - left_shoulder.x
    dy = right_shoulder.y - left_shoulder.y
    angle = math.degrees(math.atan2(dy, dx))
    return abs(angle)


def draw_label_box(image, text, position, text_color=(255, 255, 255)):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)


# =========================
# 비디오 처리 쓰레드
# =========================
class VideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    status_msg = QtCore.pyqtSignal(str)
    flags_ready = QtCore.pyqtSignal(dict)

    def __init__(self, device_index=0, parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self.running = False

        # 임계값 (당신 코드 그대로)
        self.EAR_THRESHOLD = 0.2
        self.CLOSED_EYES_FRAMES = 30
        self.MAR_THRESHOLD = 0.7
        self.OPEN_MOUTH_FRAMES = 30
        self.HEAD_TILT_THRESHOLD = 15
        self.HEAD_TILT_FRAMES = 30

        # 상태 카운터
        self.closed_eyes_frame_count = 0
        self.open_mouth_frame_count = 0
        self.head_tilt_frame_count = 0

        # 소리
        self.phone_path = r"C:/나영/학교/3학년/phone-ringtone-house-418509.mp3"
        self.animal_path = r"C:/나영/학교/3학년/cicada-419563.mp3"
        self.robot_path = r"C:/나영/학교/3학년/캡스톤/voice/robot_ko.mp3"

        self.phone_sound = None
        self.animal_sound = None
        self.robot_sound = None

        self.phone_played = False
        self.animal_played = False
        self.robot_played = False

        self.show_landmarks = True  # 향후 필요하면 사용 (지금은 HUD만 표시)

    def _load_sounds(self):
        try:
            self.phone_sound = pygame.mixer.Sound(self.phone_path)
            self.animal_sound = pygame.mixer.Sound(self.animal_path)
            self.robot_sound = pygame.mixer.Sound(self.robot_path)
            self.status_msg.emit("사운드 로드 완료")
        except Exception as e:
            self.status_msg.emit(f"사운드 로드 오류: {e}")

    def _stop_all_sounds(self):
        if self.phone_played and self.phone_sound:
            self.phone_sound.stop()
        if self.animal_played and self.animal_sound:
            self.animal_sound.stop()
        if self.robot_played and self.robot_sound:
            self.robot_sound.stop()
        self.phone_played = self.animal_played = self.robot_played = False

    def run(self):
        self.running = True
        self._load_sounds()

        cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.status_msg.emit("카메라를 열 수 없습니다.")
            return

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:

            while self.running:
                ok, frame = cap.read()
                if not ok:
                    self.status_msg.emit("프레임을 읽을 수 없습니다.")
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = face_mesh.process(image_rgb)
                image_rgb.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                eyes_flag = False
                yawn_flag = False
                head_flag = False

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark

                        left_eye_indices = [362, 385, 387, 263, 373, 380]
                        right_eye_indices = [33, 160, 158, 133, 153, 144]
                        mouth_indices = [13, 14, 17, 18, 78, 308]

                        left_ear = calculate_ear(landmarks, left_eye_indices)
                        right_ear = calculate_ear(landmarks, right_eye_indices)
                        ear = (left_ear + right_ear) / 2.0
                        mar = calculate_mar(landmarks, mouth_indices)
                        head_tilt_angle = calculate_head_tilt(landmarks)

                        # 카운트 로직 (원본 그대로)
                        if ear < self.EAR_THRESHOLD:
                            self.closed_eyes_frame_count += 1
                        else:
                            self.closed_eyes_frame_count = 0

                        if mar > self.MAR_THRESHOLD:
                            self.open_mouth_frame_count += 1
                        else:
                            self.open_mouth_frame_count = 0

                        if head_tilt_angle > self.HEAD_TILT_THRESHOLD:
                            self.head_tilt_frame_count += 1
                        else:
                            self.head_tilt_frame_count = 0

                        # 경고 조건 + 소리 로직
                        if self.closed_eyes_frame_count >= self.CLOSED_EYES_FRAMES:
                            draw_label_box(image, "Eyes Closed", (30, 60), (0, 0, 255))
                            eyes_flag = True
                            # animal_sound loop
                            if self.animal_sound and not self.animal_played:
                                self.animal_sound.play(-1)
                                self.animal_played = True
                            if self.robot_played and self.robot_sound:
                                self.robot_sound.stop()
                                self.robot_played = False
                            if self.phone_played and self.phone_sound:
                                self.phone_sound.stop()
                                self.phone_played = False

                        elif self.open_mouth_frame_count >= self.OPEN_MOUTH_FRAMES:
                            draw_label_box(image, "Yawn Detected", (30, 60), (0, 165, 255))
                            yawn_flag = True
                            # robot_sound loop
                            if self.robot_sound and not self.robot_played:
                                self.robot_sound.play(-1)
                                self.robot_played = True
                            if self.animal_played and self.animal_sound:
                                self.animal_sound.stop()
                                self.animal_played = False
                            if self.phone_played and self.phone_sound:
                                self.phone_sound.stop()
                                self.phone_played = False

                        elif self.head_tilt_frame_count >= self.HEAD_TILT_FRAMES:
                            draw_label_box(image, "Sleep Detected", (30, 60), (255, 0, 0))
                            head_flag = True
                            # phone_sound loop
                            if self.phone_sound and not self.phone_played:
                                self.phone_sound.play(-1)
                                self.phone_played = True
                            if self.animal_played and self.animal_sound:
                                self.animal_sound.stop()
                                self.animal_played = False
                            if self.robot_played and self.robot_sound:
                                self.robot_sound.stop()
                                self.robot_played = False

                        else:
                            # 아무 조건도 아닐 때: 전부 stop
                            self._stop_all_sounds()

                        break  # 한 얼굴만 처리
                else:
                    draw_label_box(image, "No face detected", (30, 60), (200, 200, 200))
                    # 얼굴 없으면 소리 다 끔
                    self._stop_all_sounds()

                # PyQt로 전달
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
                self.frame_ready.emit(qimg.copy())

                self.flags_ready.emit({
                    "eyes": eyes_flag,
                    "yawn": yawn_flag,
                    "head": head_flag
                })

            cap.release()
            self._stop_all_sounds()
            self.status_msg.emit("카메라 종료")

    def stop(self):
        self.running = False
        self.wait(1000)


# =========================
# PyQt 메인 윈도우
# =========================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness, Yawn & Head Tilt · PyQt UI")
        self.resize(1280, 800)

        self._make_toolbar()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QHBoxLayout(central)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(10)

        # 왼쪽: 비디오 카드
        video_card = QtWidgets.QFrame()
        video_card.setObjectName("card")
        video_card.setStyleSheet(
            "#card{background:#202225;border-radius:14px;border:1px solid #2f3136;}"
        )
        vlay = QtWidgets.QVBoxLayout(video_card)
        vlay.setContentsMargins(12, 12, 12, 12)

        head_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Live Preview")
        title.setStyleSheet("color:#e6e6e6;font-size:16px;font-weight:700;")
        self.pill = QtWidgets.QLabel("Stopped")
        self.pill.setStyleSheet(
            "QLabel{background:#3f3f46;color:#e5e7eb;padding:4px 10px;border-radius:10px;font-weight:700;}"
        )
        head_row.addWidget(title)
        head_row.addStretch(1)
        head_row.addWidget(self.pill)
        vlay.addLayout(head_row)

        self.video_label = QtWidgets.QLabel("Start 버튼을 눌러 카메라를 실행하세요.")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(900, 600)
        self.video_label.setStyleSheet("color:#bdbdbd;")
        vlay.addWidget(self.video_label, 1)

        main.addWidget(video_card, 1)

        # 오른쪽: 간단한 상태 패널
        side = QtWidgets.QFrame()
        side.setObjectName("side")
        side.setFixedWidth(260)
        side.setStyleSheet(
            "#side{background:#18191c;border-radius:14px;border:1px solid #2f3136;}"
        )
        s = QtWidgets.QVBoxLayout(side)
        s.setContentsMargins(14, 14, 14, 14)
        s.setSpacing(10)

        sec = QtWidgets.QLabel("상태")
        sec.setStyleSheet("color:#e6e6e6;font-weight:700;font-size:14px;")
        s.addWidget(sec)

        self.badge_eyes = self._make_badge("Eyes", "#374151")
        self.badge_yawn = self._make_badge("Yawn", "#374151")
        self.badge_head = self._make_badge("Sleep", "#374151")

        s.addWidget(self.badge_eyes)
        s.addWidget(self.badge_yawn)
        s.addWidget(self.badge_head)
        s.addStretch(1)

        main.addWidget(side, 0)

        # status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.status.setStyleSheet("QStatusBar{background:#0f1012;color:#9ca3af;}")

        # thread
        self.thread = VideoThread()
        self.thread.frame_ready.connect(self.on_frame)
        self.thread.status_msg.connect(self.status.showMessage)
        self.thread.flags_ready.connect(self.on_flags)

        # style
        self.setStyleSheet(
            "QMainWindow{background:#0f1012;}"
            "QToolBar{background:#0f1012;border:0;}"
            "QToolButton{color:#e6e6e6;padding:6px 10px;border-radius:6px;}"
            "QToolButton:hover{background:#202225;}"
            "QPushButton{background:#5865f2;color:#fff;padding:6px 10px;"
            "border-radius:8px;font-weight:600;}"
            "QPushButton:disabled{background:#40444b;color:#b9bbbe;}"
        )

    def _make_toolbar(self):
        tb = QtWidgets.QToolBar()
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(18, 18))
        self.addToolBar(tb)

        self.act_start = QtWidgets.QAction("▶ Start", self)
        self.act_stop = QtWidgets.QAction("■ Stop", self)

        tb.addAction(self.act_start)
        tb.addAction(self.act_stop)

        self.act_start.triggered.connect(self.start_camera)
        self.act_stop.triggered.connect(self.stop_camera)

    def _make_badge(self, text, bg):
        lab = QtWidgets.QLabel(text)
        lab.setAlignment(QtCore.Qt.AlignCenter)
        lab.setStyleSheet(
            f"QLabel{{background:{bg};color:#e5e7eb;padding:4px 10px;border-radius:10px;font-weight:700;}}"
        )
        return lab

    @QtCore.pyqtSlot()
    def start_camera(self):
        if self.thread.isRunning():
            return
        self.thread.start()
        self.pill.setText("Live")
        self.pill.setStyleSheet(
            "QLabel{background:#14532d;color:#bbf7d0;padding:4px 10px;border-radius:10px;font-weight:700;}"
        )
        self.status.showMessage("카메라 시작", 2000)

    @QtCore.pyqtSlot()
    def stop_camera(self):
        if self.thread.isRunning():
            self.thread.stop()
        self.pill.setText("Stopped")
        self.pill.setStyleSheet(
            "QLabel{background:#3f3f46;color:#e5e7eb;padding:4px 10px;border-radius:10px;font-weight:700;}"
        )
        self.status.showMessage("카메라 중지", 2000)

    @QtCore.pyqtSlot(QtGui.QImage)
    def on_frame(self, qimg):
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    @QtCore.pyqtSlot(dict)
    def on_flags(self, flags):
        # eyes
        self.badge_eyes.setStyleSheet(
            "QLabel{background:%s;color:#e5e7eb;padding:4px 10px;border-radius:10px;font-weight:700;}"
            % ("#b91c1c" if flags.get("eyes") else "#374151")
        )
        # yawn
        self.badge_yawn.setStyleSheet(
            "QLabel{background:%s;color:#e5e7eb;padding:4px 10px;border-radius:10px;font-weight:700;}"
            % ("#b45309" if flags.get("yawn") else "#374151")
        )
        # head
        self.badge_head.setStyleSheet(
            "QLabel{background:%s;color:#e5e7eb;padding:4px 10px;border-radius:10px;font-weight:700;}"
            % ("#7f1d1d" if flags.get("head") else "#374151")
        )

    def closeEvent(self, event):
        try:
            if self.thread.isRunning():
                self.thread.stop()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Drowsiness PyQt Dashboard")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
