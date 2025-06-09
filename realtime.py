import cv2
import os
import warnings
import logging

# Tắt các cảnh báo và thông báo thông tin của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=tất cả, 1=thông tin, 2=cảnh báo, 3=lỗi
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import cvlib
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
import threading

# Tải các mô hình học máy với thông báo tùy chỉnh
print("Loading machine learning models...")
gender_model = load_model('Gender1.h5', compile=False)  # Mô hình nhận diện giới tính
emotion_model = load_model('Emotion1.h5', compile=False)  # Mô hình nhận diện cảm xúc

# Nhãn cho giới tính và cảm xúc
gender_labels = ['Male', 'Female']
emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprised', 'Angry']

# Tạo cửa sổ tkinter chính
root = tk.Tk()
root.title('Face Analysis System - HUIT')

# Lấy kích thước màn hình
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Thiết lập ứng dụng toàn màn hình
root.geometry(f"{screen_width}x{screen_height}")
root.state('zoomed')  # Windows
# root.attributes('-zoomed', True)  # Linux

# Thiết lập icon cho ứng dụng
try:
    icon = PhotoImage(file='img/logo_huit.png')
    root.iconphoto(True, icon)
except:
    pass

# Khởi tạo các biến trạng thái
is_running = False  # Trạng thái chạy camera
capture = None      # Đối tượng camera
result_data = {     # Dữ liệu kết quả phân tích
    'gender': None,
    'emotion': None,
    'confidence': 0
}

# Định nghĩa bảng màu (BGR cho OpenCV)
PRIMARY_COLOR_BGR = (219, 152, 52)  # Xanh dương trong BGR
SECONDARY_COLOR_BGR = (80, 62, 44)  # Xanh đen trong BGR
ACCENT_COLOR_BGR = (60, 76, 231)    # Đỏ trong BGR
TEXT_COLOR_BGR = (0, 0, 0)          # Đen

# Định nghĩa bảng màu (RGB cho Tkinter)
PRIMARY_COLOR = "#3498db"     # Xanh dương
SECONDARY_COLOR = "#2c3e50"   # Xanh đen
ACCENT_COLOR = "#e74c3c"      # Đỏ
BG_COLOR = "#ecf0f1"          # Xám nhạt
TEXT_COLOR = "#2c3e50"        # Xanh đen
HIGHLIGHT_COLOR = "#2ecc71"   # Xanh lá

def use_camera():
    """Hàm khởi động camera và bắt đầu phân tích"""
    global is_running, capture
    is_running = True
    start_button.config(state="disabled")  # Vô hiệu hóa nút bắt đầu
    stop_button.config(state="normal")     # Kích hoạt nút dừng

    # Khởi tạo camera nếu chưa có
    if capture is None:
        capture = cv2.VideoCapture(0)

    # Kiểm tra xem camera có mở thành công không
    if not capture.isOpened():
        status_label.config(text="Lỗi: Không thể mở camera!")
        is_running = False
        start_button.config(state="normal")
        stop_button.config(state="disabled")
        return

    # Tạo thread riêng cho việc xử lý camera
    worker_thread = threading.Thread(target=camera_worker)
    worker_thread.daemon = True
    worker_thread.start()

def quit_program():
    """Hàm thoát chương trình với xác nhận"""
    answer = messagebox.askyesno("Thoát", "Bạn có muốn thoát khỏi chương trình?")
    if answer:
        global is_running, capture
        is_running = False
        if capture:
            capture.release()  # Giải phóng camera
        root.destroy()         # Đóng cửa sổ

def cancel_feed():
    """Hàm dừng camera feed"""
    global is_running
    is_running = False
    start_button.config(state="normal")    # Kích hoạt lại nút bắt đầu
    stop_button.config(state="disabled")   # Vô hiệu hóa nút dừng

    status_label.config(text="Đã dừng camera")

def update_result_panel():
    """Hàm cập nhật bảng kết quả phân tích"""
    if result_data['gender'] and result_data['emotion']:
        # Cập nhật hiển thị giới tính và cảm xúc
        gender_value.config(text=result_data['gender'])
        emotion_value.config(text=result_data['emotion'])

        # Thay đổi màu sắc dựa trên cảm xúc
        if result_data['emotion'] == 'Happy':
            emotion_value.config(fg="#f1c40f")  # Vàng cho vui vẻ
        elif result_data['emotion'] == 'Sad':
            emotion_value.config(fg="#3498db")  # Xanh dương cho buồn
        elif result_data['emotion'] == 'Angry':
            emotion_value.config(fg="#e74c3c")  # Đỏ cho tức giận
        elif result_data['emotion'] == 'Surprised':
            emotion_value.config(fg="#9b59b6")  # Tím cho ngạc nhiên
        else:
            emotion_value.config(fg=TEXT_COLOR)  # Màu mặc định

        # Cập nhật thanh tiến trình độ tin cậy
        confidence_progress['value'] = result_data['confidence'] * 100

def camera_worker():
    """Hàm worker chính để xử lý camera và phân tích khuôn mặt"""
    global is_running, result_data, capture

    status_label.config(text="Camera đang hoạt động...")

    while is_running:
        # Đọc frame từ camera
        ret, frame = capture.read()
        if not ret:
            status_label.config(text="Lỗi: Không thể đọc khung hình từ camera!")
            break

        # Phát hiện khuôn mặt trong frame
        faces, confidences = cvlib.detect_face(frame)

        # Vẽ khung phân tích xung quanh toàn bộ frame
        frame_height, frame_width = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (frame_width-10, frame_height-10), (50, 50, 50), 2)

        # Thêm thông tin thời gian lên frame
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        cv2.putText(frame, f"Time: {current_time:.2f}s", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Cập nhật số lượng khuôn mặt được phát hiện
        face_count = len(faces)
        root.after(100, lambda: face_count_label.config(text=f"Số khuôn mặt: {face_count}"))

        # Xử lý từng khuôn mặt được phát hiện
        for face_idx, (face, confidence) in enumerate(zip(faces, confidences)):
            # Lấy tọa độ của khung khuôn mặt
            (startX, startY) = face[0], face[1]
            (endX, endY) = face[2], face[3]

            # Vẽ khung chữ nhật xung quanh khuôn mặt
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Thêm khung nền cho thông tin text
            cv2.rectangle(frame, (startX, endY+5), (endX, endY+30), (0, 0, 0), -1)

            # Cắt vùng khuôn mặt để phân tích
            face_crop = np.copy(frame[startY:endY, startX:endX])

            # Bỏ qua nếu khuôn mặt quá nhỏ
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Tiền xử lý khuôn mặt cho việc dự đoán
            face_crop = cv2.resize(face_crop, (150, 150))  # Resize về kích thước chuẩn
            face_crop = face_crop.astype("float") / 255.0  # Chuẩn hóa pixel về [0,1]
            face_crop = img_to_array(face_crop)            # Chuyển đổi thành array
            face_crop = np.expand_dims(face_crop, axis=0)  # Thêm batch dimension

            # Đo thời gian xử lý
            start_process = cv2.getTickCount()

            # Dự đoán giới tính
            conf_model_gender = gender_model.predict(face_crop)[0]
            idx_model_gender = np.argmax(conf_model_gender)
            label_model_gender = gender_labels[idx_model_gender]

            # Dự đoán cảm xúc
            conf_model_emotion = emotion_model.predict(face_crop)[0]
            idx_model_emotion = np.argmax(conf_model_emotion)
            label_model_emotion = emotion_labels[idx_model_emotion]

            # Tính toán thời gian xử lý
            end_process = cv2.getTickCount()
            process_time = (end_process - start_process) / cv2.getTickFrequency() * 1000
            root.after(100, lambda t=process_time: processing_time_label.config(text=f"Thời gian xử lý: {t:.1f}ms"))

            # Lưu kết quả cho khuôn mặt đầu tiên (chính)
            if face_idx == 0:
                result_data['gender'] = label_model_gender
                result_data['emotion'] = label_model_emotion
                result_data['confidence'] = max(conf_model_gender[idx_model_gender], 
                                               conf_model_emotion[idx_model_emotion])

                # Cập nhật bảng kết quả trên giao diện
                root.after(100, update_result_panel)

            # Tạo nhãn hiển thị trên frame
            label = f"Face #{face_idx+1}: {label_model_gender}, {label_model_emotion}"

            # Chọn màu text dựa vào cảm xúc
            if label_model_emotion == 'Happy':
                text_color = (0, 255, 255)  # Vàng (BGR)
            elif label_model_emotion == 'Sad':
                text_color = (255, 0, 0)    # Xanh dương (BGR)
            elif label_model_emotion == 'Angry':
                text_color = (0, 0, 255)    # Đỏ (BGR)
            elif label_model_emotion == 'Surprised':
                text_color = (255, 0, 255)  # Tím (BGR)
            else:
                text_color = (255, 255, 255)  # Trắng (BGR)

            # Vẽ text thông tin lên frame
            Y = endY + 25  # Vị trí text dưới khuôn mặt
            cv2.putText(frame, label, (startX+5, Y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.65, text_color, 2)

        # Thêm watermark của trường
        cv2.putText(frame, "HUIT - AI Face Analysis", 
                    (frame_width-250, frame_height-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Chuyển đổi frame từ BGR (OpenCV) sang RGB (PIL) và hiển thị
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Tính toán tỷ lệ để fit vào khung hiển thị
            display_width = camera_frame.winfo_width()
            display_height = camera_frame.winfo_height()

            if display_width > 0 and display_height > 0:
                # Tính tỷ lệ scaling
                img_ratio = frame_width / frame_height
                display_ratio = display_width / display_height

                if display_ratio > img_ratio:
                    # Chiều cao bị giới hạn
                    new_height = display_height
                    new_width = int(new_height * img_ratio)
                else:
                    # Chiều rộng bị giới hạn
                    new_width = display_width
                    new_height = int(new_width / img_ratio)

                # Resize ảnh với chất lượng cao
                resize_method = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS
                image = image.resize((new_width, new_height), resize_method)

            # Chuyển đổi PIL Image thành ImageTk để hiển thị trên Tkinter
            imgtk = ImageTk.PhotoImage(image=image)

            # Cập nhật ảnh trên label
            camera_label.configure(image=imgtk)
            camera_label.image = imgtk
        except Exception as e:
            print(f"Error displaying image: {e}")

        # Thoát nếu nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Reset trạng thái khi thoát vòng lặp
    if is_running:
        is_running = False
        start_button.config(state="normal")
        stop_button.config(state="disabled")

# === PHẦN THIẾT KẾ GIAO DIỆN ===

# Tạo layout chính
main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.pack(fill=BOTH, expand=True)

# Tạo header (thanh tiêu đề)
header_frame = tk.Frame(main_frame, bg=SECONDARY_COLOR, height=80)
header_frame.pack(fill=X)

# Thêm logo và tiêu đề vào header
try:
    logo_img = Image.open('img/logo_huit.png')
    logo_size = min(60, logo_img.width, logo_img.height)

    resize_method = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS
    logo_img = logo_img.resize((logo_size, logo_size), resize_method)
    logo_photo = ImageTk.PhotoImage(logo_img)
    logo_label = tk.Label(header_frame, image=logo_photo, bg=SECONDARY_COLOR)
    logo_label.image = logo_photo
    logo_label.pack(side=LEFT, padx=20)
except Exception as e:
    print(f"Could not load logo: {e}")

# Frame chứa tiêu đề
title_frame = tk.Frame(header_frame, bg=SECONDARY_COLOR)
title_frame.pack(side=LEFT, padx=10)

# Tiêu đề chính
main_title = tk.Label(title_frame, 
                    text="HỆ THỐNG NHẬN DIỆN KHUÔN MẶT VÀ CẢM XÚC", 
                    font=("Arial", 18, "bold"),
                    fg="white",
                    bg=SECONDARY_COLOR)
main_title.pack(anchor=W)

# Tiêu đề phụ
sub_title = tk.Label(title_frame, 
                    text="Deep Learning - Nhóm 5", 
                    font=("Arial", 12),
                    fg="white",
                    bg=SECONDARY_COLOR)
sub_title.pack(anchor=W)

# Tạo khung nội dung chính
content_frame = tk.Frame(main_frame, bg=BG_COLOR)
content_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

# Khung hiển thị camera (bên trái)
camera_frame = tk.Frame(content_frame, bg=BG_COLOR, bd=2, relief=RIDGE)
camera_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

# Thiết lập kích thước cho khung camera
camera_frame.update()
max_height = int(screen_height * 0.6)  # Giới hạn chiều cao

camera_frame.configure(height=max_height)
camera_frame.pack_propagate(False)  # Ngăn tự động thay đổi kích thước

# Tiêu đề khung camera
camera_title = tk.Label(camera_frame, 
                       text="Camera Feed", 
                       font=("Arial", 14, "bold"),
                       bg=PRIMARY_COLOR,
                       fg="white",
                       padx=10,
                       pady=5)
camera_title.pack(fill=X)

# Label hiển thị video camera
camera_label = tk.Label(camera_frame, bg='black')
camera_label.pack(fill=BOTH, expand=True, padx=5, pady=5)

# Label hiển thị trạng thái
status_label = tk.Label(camera_frame, 
                       text="Chờ kết nối camera...", 
                       font=("Arial", 10),
                       bg=BG_COLOR,
                       fg=TEXT_COLOR)
status_label.pack(pady=5)

# Khung hiển thị kết quả phân tích (bên phải)
result_frame = tk.Frame(content_frame, bg=BG_COLOR, width=300, bd=2, relief=RIDGE)
result_frame.pack(side=RIGHT, fill=Y, padx=(10, 0))
result_frame.pack_propagate(False)

# Tiêu đề khung kết quả
result_title = tk.Label(result_frame, 
                       text="Kết Quả Phân Tích", 
                       font=("Arial", 14, "bold"),
                       bg=PRIMARY_COLOR,
                       fg="white",
                       padx=10,
                       pady=5)
result_title.pack(fill=X)

# Nội dung khung kết quả
result_content = tk.Frame(result_frame, bg=BG_COLOR, padx=20, pady=20)
result_content.pack(fill=BOTH, expand=True)

# Hiển thị thông tin giới tính
gender_frame = tk.Frame(result_content, bg=BG_COLOR, pady=10)
gender_frame.pack(fill=X)

gender_label = tk.Label(gender_frame, 
                       text="Giới tính:", 
                       font=("Arial", 12, "bold"),
                       bg=BG_COLOR,
                       fg=TEXT_COLOR)
gender_label.pack(side=LEFT)

gender_value = tk.Label(gender_frame, 
                       text="--", 
                       font=("Arial", 12),
                       bg=BG_COLOR,
                       fg=ACCENT_COLOR)
gender_value.pack(side=RIGHT)

# Hiển thị thông tin cảm xúc
emotion_frame = tk.Frame(result_content, bg=BG_COLOR, pady=10)
emotion_frame.pack(fill=X)

emotion_label = tk.Label(emotion_frame, 
                        text="Cảm xúc:", 
                        font=("Arial", 12, "bold"),
                        bg=BG_COLOR,
                        fg=TEXT_COLOR)
emotion_label.pack(side=LEFT)

emotion_value = tk.Label(emotion_frame, 
                        text="--", 
                        font=("Arial", 12),
                        bg=BG_COLOR,
                        fg=ACCENT_COLOR)
emotion_value.pack(side=RIGHT)

# Hiển thị độ tin cậy
confidence_frame = tk.Frame(result_content, bg=BG_COLOR, pady=10)
confidence_frame.pack(fill=X)

confidence_label = tk.Label(confidence_frame, 
                           text="Độ tin cậy:", 
                           font=("Arial", 12, "bold"),
                           bg=BG_COLOR,
                           fg=TEXT_COLOR)
confidence_label.pack(side=LEFT)

# Thanh tiến trình cho độ tin cậy
from tkinter import ttk
confidence_progress = ttk.Progressbar(result_content, orient="horizontal", 
                                     length=250, mode="determinate")
confidence_progress.pack(pady=10, fill=X)

# Khung hiển thị thông tin bổ sung
info_frame = tk.LabelFrame(result_content, text="Thông tin", bg=BG_COLOR, fg=TEXT_COLOR, padx=10, pady=10)
info_frame.pack(fill=X, pady=10)

# Hiển thị số lượng khuôn mặt
face_count_label = tk.Label(info_frame, text="Số khuôn mặt: 0", bg=BG_COLOR, fg=TEXT_COLOR)
face_count_label.pack(anchor=W)

# Hiển thị thời gian xử lý
processing_time_label = tk.Label(info_frame, text="Thời gian xử lý: 0ms", bg=BG_COLOR, fg=TEXT_COLOR)
processing_time_label.pack(anchor=W)

# Tạo container cho các nút điều khiển
button_container = tk.Frame(main_frame, bg=BG_COLOR)
button_container.pack(side=BOTTOM, fill=X, padx=20, pady=10)

button_frame = tk.Frame(button_container, bg=BG_COLOR)
button_frame.pack(fill=X)

# Nút bắt đầu
start_button = tk.Button(button_frame, 
                       text="BẮT ĐẦU", 
                       font=('Arial', 12, 'bold'), 
                       fg='white', 
                       bg=HIGHLIGHT_COLOR,
                       padx=20,
                       pady=5,
                       command=use_camera)
start_button.pack(side=LEFT, padx=20)

# Nút dừng
stop_button = tk.Button(button_frame, 
                      text="DỪNG", 
                      font=('Arial', 12, 'bold'),
                      fg='white',
                      bg=ACCENT_COLOR,
                      padx=20,
                      pady=5,
                      command=cancel_feed, 
                      state="disabled")
stop_button.pack(side=LEFT, padx=20)

# Nút thoát
exit_button = tk.Button(button_frame, 
                      text="THOÁT", 
                      font=('Arial', 12, 'bold'), 
                      fg='white', 
                      bg=SECONDARY_COLOR,
                      padx=20,
                      pady=5,
                      command=quit_program)
exit_button.pack(side=RIGHT, padx=20)

# Footer (chân trang)
footer_frame = tk.Frame(main_frame, bg=SECONDARY_COLOR, height=30)
footer_frame.pack(fill=X, side=BOTTOM)

footer_text = tk.Label(footer_frame, 
                     text="© 2025 Ho Chi Minh University Of Industry And Trade - AI Face Analysis System", 
                     font=("Arial", 9),
                     fg="white",
                     bg=SECONDARY_COLOR)
footer_text.pack(pady=5)

# Căn giữa các widget
for widget in [button_frame]:
    widget.pack_configure(anchor=CENTER)

# Xử lý sự kiện đóng cửa sổ
root.protocol("WM_DELETE_WINDOW", quit_program)

# Bắt đầu vòng lặp chính của ứng dụng
root.mainloop()