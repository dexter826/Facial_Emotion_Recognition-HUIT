import cv2
import os
import warnings
import logging
# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cvlib
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
import threading

# Load models with custom message
print("Loading machine learning models...")
gender_model = load_model('Gender1.h5', compile=False)
emotion_model = load_model('Emotion1.h5', compile=False)

gender_labels = ['Male', 'Female']
emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprised', 'Angry']

# Create tkinter window
root = tk.Tk()
root.title('Face Analysis System - HUIT')

# Lấy kích thước màn hình
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Thiết lập ứng dụng toàn màn hình
root.geometry(f"{screen_width}x{screen_height}")
root.state('zoomed')  # Windows
# root.attributes('-zoomed', True)  # Linux

# Icon
try:
    icon = PhotoImage(file='img/logo_huit.png')
    root.iconphoto(True, icon)
except:
    pass

# Biến trạng thái
is_running = False
capture = None
result_data = {
    'gender': None,
    'emotion': None,
    'confidence': 0
}

# Định nghĩa màu sắc (BGR cho OpenCV)
PRIMARY_COLOR_BGR = (219, 152, 52)  # Xanh dương trong BGR
SECONDARY_COLOR_BGR = (80, 62, 44)  # Xanh đen trong BGR
ACCENT_COLOR_BGR = (60, 76, 231)    # Đỏ trong BGR
TEXT_COLOR_BGR = (0, 0, 0)          # Đen

# Định nghĩa màu sắc (RGB cho Tkinter)
PRIMARY_COLOR = "#3498db"     # Xanh dương
SECONDARY_COLOR = "#2c3e50"   # Xanh đen
ACCENT_COLOR = "#e74c3c"      # Đỏ
BG_COLOR = "#ecf0f1"          # Xám nhạt
TEXT_COLOR = "#2c3e50"        # Xanh đen
HIGHLIGHT_COLOR = "#2ecc71"   # Xanh lá

def use_camera():
    global is_running, capture
    is_running = True
    start_button.config(state="disabled")
    stop_button.config(state="normal")
    
    if capture is None:
        capture = cv2.VideoCapture(0)
        
    # Check if camera opened successfully
    if not capture.isOpened():
        status_label.config(text="Lỗi: Không thể mở camera!")
        is_running = False
        start_button.config(state="normal")
        stop_button.config(state="disabled")
        return
    
    worker_thread = threading.Thread(target=camera_worker)
    worker_thread.daemon = True
    worker_thread.start()
    
def quit_program():
    answer = messagebox.askyesno("Thoát", "Bạn có muốn thoát khỏi chương trình?")
    if answer:
        global is_running, capture
        is_running = False
        if capture:
            capture.release()
        root.destroy()

def cancel_feed():
    global is_running
    is_running = False
    start_button.config(state="normal")
    stop_button.config(state="disabled")
    # Hiển thị thông báo dừng
    status_label.config(text="Đã dừng camera")

def update_result_panel():
    if result_data['gender'] and result_data['emotion']:
        gender_value.config(text=result_data['gender'])
        emotion_value.config(text=result_data['emotion'])
        
        # Cập nhật màu sắc dựa trên cảm xúc
        if result_data['emotion'] == 'Happy':
            emotion_value.config(fg="#f1c40f")  # Vàng
        elif result_data['emotion'] == 'Sad':
            emotion_value.config(fg="#3498db")  # Xanh dương
        elif result_data['emotion'] == 'Angry':
            emotion_value.config(fg="#e74c3c")  # Đỏ
        elif result_data['emotion'] == 'Surprised':
            emotion_value.config(fg="#9b59b6")  # Tím
        else:
            emotion_value.config(fg=TEXT_COLOR)  # Màu mặc định
            
        confidence_progress['value'] = result_data['confidence'] * 100
        
def camera_worker():
    global is_running, result_data, capture
    
    status_label.config(text="Camera đang hoạt động...")
    
    while is_running:
        ret, frame = capture.read()
        if not ret:
            status_label.config(text="Lỗi: Không thể đọc khung hình từ camera!")
            break

        # Face detection
        faces, confidences = cvlib.detect_face(frame)
        
        # Vẽ khung phân tích
        frame_height, frame_width = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (frame_width-10, frame_height-10), (50, 50, 50), 2)
        
        # Thêm thời gian/ngày trên khung hình
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        cv2.putText(frame, f"Time: {current_time:.2f}s", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Cập nhật số lượng khuôn mặt phát hiện được
        face_count = len(faces)
        root.after(100, lambda: face_count_label.config(text=f"Số khuôn mặt: {face_count}"))

        for face_idx, (face, confidence) in enumerate(zip(faces, confidences)):
            # Get the coordinates of the face rectangle
            (startX, startY) = face[0], face[1]
            (endX, endY) = face[2], face[3]

            # Draw rectangle around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            # Thêm khung thông tin
            cv2.rectangle(frame, (startX, endY+5), (endX, endY+30), (0, 0, 0), -1)

            # Crop the detected face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Preprocess the face for prediction
            face_crop = cv2.resize(face_crop, (150, 150))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Đo thời gian xử lý
            start_process = cv2.getTickCount()

            # Predict gender
            conf_model_gender = gender_model.predict(face_crop)[0]
            idx_model_gender = np.argmax(conf_model_gender)
            label_model_gender = gender_labels[idx_model_gender]
            
            # Predict emotion
            conf_model_emotion = emotion_model.predict(face_crop)[0]
            idx_model_emotion = np.argmax(conf_model_emotion)
            label_model_emotion = emotion_labels[idx_model_emotion]

            # Tính thời gian xử lý
            end_process = cv2.getTickCount()
            process_time = (end_process - start_process) / cv2.getTickFrequency() * 1000
            root.after(100, lambda t=process_time: processing_time_label.config(text=f"Thời gian xử lý: {t:.1f}ms"))

            # Lưu kết quả cho face đầu tiên
            if face_idx == 0:
                result_data['gender'] = label_model_gender
                result_data['emotion'] = label_model_emotion
                result_data['confidence'] = max(conf_model_gender[idx_model_gender], 
                                               conf_model_emotion[idx_model_emotion])
                
                # Cập nhật bảng kết quả
                root.after(100, update_result_panel)

            # Chuẩn bị label hiển thị
            label = f"Face #{face_idx+1}: {label_model_gender}, {label_model_emotion}"
            
            # Chọn màu dựa vào emotion
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

            Y = endY + 25  # Vị trí văn bản dưới khuôn mặt
            cv2.putText(frame, label, (startX+5, Y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.65, text_color, 2)

        # Thêm watermark
        cv2.putText(frame, "HUIT - AI Face Analysis", 
                    (frame_width-250, frame_height-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Convert the image from OpenCV BGR format to PIL Image
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Scale to fit the display area while maintaining aspect ratio
            display_width = camera_frame.winfo_width()
            display_height = camera_frame.winfo_height()
            
            if display_width > 0 and display_height > 0:
                # Calculate scaling ratio
                img_ratio = frame_width / frame_height
                display_ratio = display_width / display_height
                
                if display_ratio > img_ratio:
                    # Height constrained
                    new_height = display_height
                    new_width = int(new_height * img_ratio)
                else:
                    # Width constrained
                    new_width = display_width
                    new_height = int(new_width / img_ratio)
                    
                # Sử dụng LANCZOS nếu có, nếu không thì dùng ANTIALIAS
                resize_method = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS
                image = image.resize((new_width, new_height), resize_method)

            # Convert the PIL Image to ImageTk to display on Tkinter label
            imgtk = ImageTk.PhotoImage(image=image)

            # Update the image on the label
            camera_label.configure(image=imgtk)
            camera_label.image = imgtk
        except Exception as e:
            print(f"Error displaying image: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if is_running:  # Nếu thoát vòng lặp không phải do cancel_feed
        is_running = False
        start_button.config(state="normal")
        stop_button.config(state="disabled")

# Tạo layout chính
main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.pack(fill=BOTH, expand=True)

# Tạo header
header_frame = tk.Frame(main_frame, bg=SECONDARY_COLOR, height=80)
header_frame.pack(fill=X)

# Logo và tiêu đề
try:
    logo_img = Image.open('img/logo_huit.png')
    logo_size = min(60, logo_img.width, logo_img.height)
    logo_img = logo_img.resize((logo_size, logo_size))
    resize_method = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS
    logo_img = logo_img.resize((logo_size, logo_size), resize_method)
    logo_photo = ImageTk.PhotoImage(logo_img)
    logo_label = tk.Label(header_frame, image=logo_photo, bg=SECONDARY_COLOR)
    logo_label.image = logo_photo
    logo_label.pack(side=LEFT, padx=20)
except Exception as e:
    print(f"Could not load logo: {e}")

title_frame = tk.Frame(header_frame, bg=SECONDARY_COLOR)
title_frame.pack(side=LEFT, padx=10)

main_title = tk.Label(title_frame, 
                    text="HỆ THỐNG NHẬN DIỆN KHUÔN MẶT VÀ CẢM XÚC", 
                    font=("Arial", 18, "bold"),
                    fg="white",
                    bg=SECONDARY_COLOR)
main_title.pack(anchor=W)

sub_title = tk.Label(title_frame, 
                    text="Deep Learning - Nhóm 5", 
                    font=("Arial", 12),
                    fg="white",
                    bg=SECONDARY_COLOR)
sub_title.pack(anchor=W)

# Tạo khung nội dung chính
content_frame = tk.Frame(main_frame, bg=BG_COLOR)
content_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

# Khung camera bên trái
camera_frame = tk.Frame(content_frame, bg=BG_COLOR, bd=2, relief=RIDGE)
camera_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
# Đặt kích thước tối thiểu và tối đa cho camera_frame
camera_frame.update()
screen_height = root.winfo_screenheight()
max_height = int(screen_height * 0.6)  # Giới hạn chiều cao tối đa là 60% màn hình
camera_frame.configure(height=max_height)
camera_frame.pack_propagate(False)  # Ngăn frame tự động thay đổi kích thước

camera_title = tk.Label(camera_frame, 
                       text="Camera Feed", 
                       font=("Arial", 14, "bold"),
                       bg=PRIMARY_COLOR,
                       fg="white",
                       padx=10,
                       pady=5)
camera_title.pack(fill=X)

camera_label = tk.Label(camera_frame, bg='black')
camera_label.pack(fill=BOTH, expand=True, padx=5, pady=5)

status_label = tk.Label(camera_frame, 
                       text="Chờ kết nối camera...", 
                       font=("Arial", 10),
                       bg=BG_COLOR,
                       fg=TEXT_COLOR)
status_label.pack(pady=5)

# Khung kết quả phân tích bên phải
result_frame = tk.Frame(content_frame, bg=BG_COLOR, width=300, bd=2, relief=RIDGE)
result_frame.pack(side=RIGHT, fill=Y, padx=(10, 0))
result_frame.pack_propagate(False)

result_title = tk.Label(result_frame, 
                       text="Kết Quả Phân Tích", 
                       font=("Arial", 14, "bold"),
                       bg=PRIMARY_COLOR,
                       fg="white",
                       padx=10,
                       pady=5)
result_title.pack(fill=X)

result_content = tk.Frame(result_frame, bg=BG_COLOR, padx=20, pady=20)
result_content.pack(fill=BOTH, expand=True)

# Thông tin giới tính
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

# Thông tin cảm xúc
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

# Độ tin cậy
confidence_frame = tk.Frame(result_content, bg=BG_COLOR, pady=10)
confidence_frame.pack(fill=X)

confidence_label = tk.Label(confidence_frame, 
                           text="Độ tin cậy:", 
                           font=("Arial", 12, "bold"),
                           bg=BG_COLOR,
                           fg=TEXT_COLOR)
confidence_label.pack(side=LEFT)

from tkinter import ttk
confidence_progress = ttk.Progressbar(result_content, orient="horizontal", 
                                     length=250, mode="determinate")
confidence_progress.pack(pady=10, fill=X)

# Thêm hiển thị thông tin
info_frame = tk.LabelFrame(result_content, text="Thông tin", bg=BG_COLOR, fg=TEXT_COLOR, padx=10, pady=10)
info_frame.pack(fill=X, pady=10)

face_count_label = tk.Label(info_frame, text="Số khuôn mặt: 0", bg=BG_COLOR, fg=TEXT_COLOR)
face_count_label.pack(anchor=W)

processing_time_label = tk.Label(info_frame, text="Thời gian xử lý: 0ms", bg=BG_COLOR, fg=TEXT_COLOR)
processing_time_label.pack(anchor=W)

# Tạo frame cho các nút điều khiển
button_container = tk.Frame(main_frame, bg=BG_COLOR)
button_container.pack(side=BOTTOM, fill=X, padx=20, pady=10)

button_frame = tk.Frame(button_container, bg=BG_COLOR)
button_frame.pack(fill=X)

# Nút điều khiển
start_button = tk.Button(button_frame, 
                       text="BẮT ĐẦU", 
                       font=('Arial', 12, 'bold'), 
                       fg='white', 
                       bg=HIGHLIGHT_COLOR,
                       padx=20,
                       pady=5,
                       command=use_camera)
start_button.pack(side=LEFT, padx=20)

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

exit_button = tk.Button(button_frame, 
                      text="THOÁT", 
                      font=('Arial', 12, 'bold'), 
                      fg='white', 
                      bg=SECONDARY_COLOR,
                      padx=20,
                      pady=5,
                      command=quit_program)
exit_button.pack(side=RIGHT, padx=20)

# Footer
footer_frame = tk.Frame(main_frame, bg=SECONDARY_COLOR, height=30)
footer_frame.pack(fill=X, side=BOTTOM)

footer_text = tk.Label(footer_frame, 
                     text="© 2025 Ho Chi Minh University Of Industry And Trade - AI Face Analysis System", 
                     font=("Arial", 9),
                     fg="white",
                     bg=SECONDARY_COLOR)
footer_text.pack(pady=5)

# Đặt các widget ở giữa
for widget in [button_frame]:
    widget.pack_configure(anchor=CENTER)

# Bắt sự kiện đóng cửa sổ
root.protocol("WM_DELETE_WINDOW", quit_program)

# Bắt đầu loop chính
root.mainloop()