import cv2 #thư viện xử lý hình ảnh và video trong Python. Nó cung cấp các chức năng để đọc, ghi và xử lý các hình ảnh từ các nguồn đầu vào khác nhau
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cvlib #một thư viện xử lý hình ảnh dựa trên OpenCV, cung cấp các công cụ giúp phát hiện khuôn mặt, đồng thời cung cấp chức năng nhận biết giới tính và cảm xúc từ khuôn mặt
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
import threading
import requests
import json
from datetime import datetime
import webbrowser
import urllib.parse
import io
import pygame

# Load model
# face_classifier = cv2.CascadeClassifier('face_detection.xml')
gender_model = load_model('Gender1.h5')
emotion_model = load_model('Emotion1.h5')

gender_labels = ['Male', 'Female']
emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprised', 'Angry']

# Emoji mapping cho cảm xúc
emotion_emojis = {
    'Neutral': '😐',
    'Happy': '😊',
    'Sad': '😢',
    'Surprised': '😲',
    'Angry': '😠'
}



# Cấu hình Telegram Bot (bạn cần tạo bot và lấy token)
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"  # Thay bằng token thật
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"     # Thay bằng chat ID thật

# Biến lưu trữ frame cuối cùng để lưu ảnh
last_frame = None
last_gender = ""
last_emotion = ""


# �️ GAME ĐUA XE ĐIỀU KHIỂN BẰNG CẢM XÚC
game_active = False
avatar_emotion = "Neutral"
avatar_animation_frame = 0
last_emotion_change = 0
emotion_intensity = 0  # Độ mạnh của cảm xúc (0-100)
avatar_x = 110  # Vị trí x của nhân vật
avatar_y = 200  # Vị trí y của nhân vật
animation_speed = 0.2
emotion_colors = {
    'Happy': '#FFD700',    # Vàng
    'Sad': '#4169E1',      # Xanh dương
    'Angry': '#FF4500',    # Đỏ cam
    'Surprised': '#FF69B4', # Hồng
    'Neutral': '#808080'    # Xám
}

# Create tkinter window
root = tk.Tk()
root.geometry('1200x700')
root.resizable(False, False)
root.title('AI Face Detection System - Professional Edition')
root.configure(bg='#1a1a2e')

# Load and set window icon
try:
    icon_image = Image.open('Huit.png')
    icon_image = icon_image.resize((32, 32), Image.LANCZOS)
    icon = ImageTk.PhotoImage(icon_image)
    root.iconphoto(True, icon)
except:
    pass


is_running = False

# ===== CÁC HÀM GỬI TIN NHẮN =====

def send_telegram_message(message):
    """Gửi tin nhắn qua Telegram Bot"""
    try:
        if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE":
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=5)
            if response.status_code == 200:
                print("✅ Telegram message sent successfully!")
            else:
                print("❌ Failed to send Telegram message")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

def send_whatsapp_message(message):
    """Mở WhatsApp Web với tin nhắn đã soạn sẵn"""
    try:
        # Encode message cho URL
        encoded_message = urllib.parse.quote(message)
        whatsapp_url = f"https://web.whatsapp.com/send?text={encoded_message}"
        webbrowser.open(whatsapp_url)
        print("✅ WhatsApp opened with message!")
    except Exception as e:
        print(f"❌ WhatsApp error: {e}")





# ===== GAME TÍCH ĐIỂM CẢM XÚC =====

def start_avatar_game():
    """Bắt đầu game nhân vật cảm xúc"""
    global game_active, avatar_emotion, last_emotion_change, emotion_intensity

    if not is_running:
        messagebox.showwarning("⚠️ Cảnh báo", "Vui lòng bấm START camera trước khi chơi game!")
        return

    game_active = True
    avatar_emotion = "Neutral"
    last_emotion_change = datetime.now().timestamp()
    emotion_intensity = 0

    # Cập nhật UI
    game_button.config(text="� ĐANG CHƠI...", state="disabled", bg="#95a5a6")
    update_avatar_display()

    print(f"� Game nhân vật bắt đầu!")
    print(f"✨ Thể hiện cảm xúc để điều khiển nhân vật!")

def stop_avatar_game():
    """Dừng game nhân vật cảm xúc"""
    global game_active

    game_active = False
    game_button.config(text="� AVATAR", state="normal", bg="#e67e22")

    # Hiển thị kết quả
    result_message = f"""
� GAME NHÂN VẬT CẢM XÚC KẾT THÚC

✨ Cảm ơn bạn đã chơi!
🎭 Nhân vật đã phản ứng theo cảm xúc của bạn
😊 Hãy thử lại để xem những phản ứng khác!
    """

    messagebox.showinfo("� Kết thúc Game", result_message)

    # Reset hiển thị
    game_info_label.config(text="� Bấm AVATAR để bắt đầu chơi!")

def get_rank_by_score(score):
    """Xếp hạng theo điểm số"""
    if score >= 90:
        return "🥇 MASTER - Bậc thầy cảm xúc!"
    elif score >= 70:
        return "🥈 EXPERT - Chuyên gia!"
    elif score >= 50:
        return "🥉 GOOD - Khá tốt!"
    elif score >= 30:
        return "📈 BEGINNER - Mới bắt đầu"
    else:
        return "😅 PRACTICE MORE - Luyện tập thêm nhé!"

def update_avatar_emotion(detected_emotion):
    """Cập nhật cảm xúc của nhân vật"""
    global avatar_emotion, last_emotion_change, emotion_intensity

    if not game_active:
        return

    current_time = datetime.now().timestamp()

    # Cập nhật cảm xúc nhân vật
    if detected_emotion != avatar_emotion:
        avatar_emotion = detected_emotion
        last_emotion_change = current_time
        emotion_intensity = 100  # Cảm xúc mạnh khi mới thay đổi
        print(f"🎭 Nhân vật thay đổi cảm xúc: {avatar_emotion} {emotion_emojis.get(avatar_emotion, '😐')}")

    # Giảm dần cường độ cảm xúc theo thời gian
    time_since_change = current_time - last_emotion_change
    emotion_intensity = max(20, 100 - (time_since_change * 20))  # Giảm 20 mỗi giây, tối thiểu 20

    # Cập nhật hiển thị
    update_avatar_display()

def update_avatar_display():
    """Cập nhật hiển thị nhân vật và thông tin"""
    if not game_active:
        return

    # Tạo mô tả trạng thái nhân vật
    emotion_desc = get_avatar_description(avatar_emotion, emotion_intensity)

    game_text = f"""🎭 NHÂN VẬT CẢM XÚC

😊 Cảm xúc hiện tại: {avatar_emotion} {emotion_emojis.get(avatar_emotion, '😐')}
💪 Cường độ: {emotion_intensity:.0f}%

{emotion_desc}

✨ Thể hiện cảm xúc khác để thay đổi nhân vật!"""

    game_info_label.config(text=game_text)

    # Vẽ nhân vật hoạt hình trên canvas
    draw_animated_avatar()

def draw_animated_avatar():
    """Vẽ nhân vật hoạt hình trên canvas"""
    global avatar_animation_frame

    # Xóa canvas
    avatar_canvas.delete("all")

    # Lấy màu theo cảm xúc
    color = emotion_colors.get(avatar_emotion, '#808080')

    # Tính toán animation frame
    avatar_animation_frame += animation_speed
    if avatar_animation_frame > 2 * 3.14159:  # 2π
        avatar_animation_frame = 0

    # Vị trí trung tâm trong canvas
    center_x = 110
    center_y = 100

    # Vẽ nhân vật theo cảm xúc
    if avatar_emotion == 'Happy':
        draw_happy_avatar(center_x, center_y, color)
    elif avatar_emotion == 'Sad':
        draw_sad_avatar(center_x, center_y, color)
    elif avatar_emotion == 'Angry':
        draw_angry_avatar(center_x, center_y, color)
    elif avatar_emotion == 'Surprised':
        draw_surprised_avatar(center_x, center_y, color)
    else:  # Neutral
        draw_neutral_avatar(center_x, center_y, color)

    # Lặp lại animation
    if game_active:
        root.after(50, draw_animated_avatar)  # 20 FPS

def draw_happy_avatar(x, y, color):
    """Vẽ nhân vật vui vẻ - nhảy múa"""
    import math

    # Nhảy lên xuống
    bounce = math.sin(avatar_animation_frame * 4) * 10
    y_pos = y + bounce

    # Thân người (hình oval)
    avatar_canvas.create_oval(x-30, y_pos-40, x+30, y_pos+40,
                             fill=color, outline='black', width=2)

    # Đầu (hình tròn)
    avatar_canvas.create_oval(x-20, y_pos-70, x+20, y_pos-30,
                             fill='#FFDBAC', outline='black', width=2)

    # Mắt vui (hình cung)
    avatar_canvas.create_arc(x-15, y_pos-60, x-5, y_pos-50,
                            start=0, extent=180, fill='black')
    avatar_canvas.create_arc(x+5, y_pos-60, x+15, y_pos-50,
                            start=0, extent=180, fill='black')

    # Miệng cười
    avatar_canvas.create_arc(x-10, y_pos-50, x+10, y_pos-40,
                            start=0, extent=-180, outline='red', width=3)

    # Tay vẫy (chuyển động)
    arm_angle = math.sin(avatar_animation_frame * 6) * 30
    arm_x = x + 35 + math.cos(math.radians(arm_angle)) * 15
    arm_y = y_pos - 10 + math.sin(math.radians(arm_angle)) * 15

    avatar_canvas.create_line(x+30, y_pos-10, arm_x, arm_y,
                             fill='black', width=4)
    avatar_canvas.create_oval(arm_x-5, arm_y-5, arm_x+5, arm_y+5,
                             fill='#FFDBAC', outline='black')

def draw_sad_avatar(x, y, color):
    """Vẽ nhân vật buồn - cúi đầu"""
    # Thân người
    avatar_canvas.create_oval(x-30, y-40, x+30, y+40,
                             fill=color, outline='black', width=2)

    # Đầu cúi xuống
    head_y = y - 45
    avatar_canvas.create_oval(x-20, head_y-25, x+20, head_y+15,
                             fill='#FFDBAC', outline='black', width=2)

    # Mắt buồn (đường thẳng)
    avatar_canvas.create_line(x-15, head_y-10, x-5, head_y-5,
                             fill='black', width=2)
    avatar_canvas.create_line(x+5, head_y-10, x+15, head_y-5,
                             fill='black', width=2)

    # Miệng buồn
    avatar_canvas.create_arc(x-8, head_y, x+8, head_y+10,
                            start=0, extent=180, outline='blue', width=3)

    # Nước mắt
    avatar_canvas.create_oval(x-18, head_y-5, x-16, head_y+5,
                             fill='lightblue', outline='blue')
    avatar_canvas.create_oval(x+16, head_y-5, x+18, head_y+5,
                             fill='lightblue', outline='blue')

def draw_angry_avatar(x, y, color):
    """Vẽ nhân vật tức giận - rung lắc"""
    import math

    # Rung lắc
    shake = math.sin(avatar_animation_frame * 10) * 3
    x_pos = x + shake

    # Thân người (đỏ)
    avatar_canvas.create_oval(x_pos-30, y-40, x_pos+30, y+40,
                             fill='#FF4500', outline='darkred', width=3)

    # Đầu
    avatar_canvas.create_oval(x_pos-20, y-70, x_pos+20, y-30,
                             fill='#FFDBAC', outline='darkred', width=2)

    # Mắt giận (chữ X)
    avatar_canvas.create_line(x_pos-15, y-60, x_pos-5, y-50,
                             fill='red', width=3)
    avatar_canvas.create_line(x_pos-15, y-50, x_pos-5, y-60,
                             fill='red', width=3)
    avatar_canvas.create_line(x_pos+5, y-60, x_pos+15, y-50,
                             fill='red', width=3)
    avatar_canvas.create_line(x_pos+5, y-50, x_pos+15, y-60,
                             fill='red', width=3)

    # Miệng giận
    avatar_canvas.create_arc(x_pos-8, y-45, x_pos+8, y-35,
                            start=0, extent=180, outline='darkred', width=3)

    # Khói từ đầu
    for i in range(3):
        smoke_y = y - 80 - i*10
        avatar_canvas.create_oval(x_pos-5+i*2, smoke_y-5, x_pos+5+i*2, smoke_y+5,
                                 fill='gray', outline='darkgray')

def draw_surprised_avatar(x, y, color):
    """Vẽ nhân vật ngạc nhiên - giật mình"""
    import math

    # Giật mình (scale lớn hơn)
    scale = 1 + math.sin(avatar_animation_frame * 8) * 0.1

    # Thân người
    avatar_canvas.create_oval(x-30*scale, y-40*scale, x+30*scale, y+40*scale,
                             fill=color, outline='black', width=2)

    # Đầu
    avatar_canvas.create_oval(x-20*scale, y-70*scale, x+20*scale, y-30*scale,
                             fill='#FFDBAC', outline='black', width=2)

    # Mắt to (hình tròn)
    avatar_canvas.create_oval(x-18, y-65, x-8, y-55,
                             fill='white', outline='black', width=2)
    avatar_canvas.create_oval(x+8, y-65, x+18, y-55,
                             fill='white', outline='black', width=2)
    avatar_canvas.create_oval(x-15, y-62, x-11, y-58, fill='black')
    avatar_canvas.create_oval(x+11, y-62, x+15, y-58, fill='black')

    # Miệng há hốc
    avatar_canvas.create_oval(x-6, y-48, x+6, y-42,
                             fill='black', outline='black')

    # Dấu chấm than
    avatar_canvas.create_text(x+35, y-80, text="!",
                             font=('Arial', 20, 'bold'), fill='red')

def draw_neutral_avatar(x, y, color):
    """Vẽ nhân vật bình thường"""
    # Thân người
    avatar_canvas.create_oval(x-30, y-40, x+30, y+40,
                             fill=color, outline='black', width=2)

    # Đầu
    avatar_canvas.create_oval(x-20, y-70, x+20, y-30,
                             fill='#FFDBAC', outline='black', width=2)

    # Mắt bình thường
    avatar_canvas.create_oval(x-15, y-60, x-10, y-55, fill='black')
    avatar_canvas.create_oval(x+10, y-60, x+15, y-55, fill='black')

    # Miệng thẳng
    avatar_canvas.create_line(x-8, y-45, x+8, y-45,
                             fill='black', width=2)

def get_avatar_description(emotion, intensity):
    """Mô tả trạng thái nhân vật theo cảm xúc"""
    descriptions = {
        'Happy': [
            "🎉 Nhân vật đang nhảy múa vui vẻ!",
            "😄 Mặt rạng rỡ, tay vẫy chào!",
            "🌟 Ánh mắt sáng ngời hạnh phúc!"
        ],
        'Sad': [
            "😢 Nhân vật cúi đầu buồn bã...",
            "💧 Nước mắt rơi, vai run rẩy...",
            "🌧️ Bầu không khí u ám quanh nhân vật..."
        ],
        'Angry': [
            "😡 Nhân vật nổi giận, mặt đỏ gay!",
            "🔥 Tay nắm chặt, chân dậm mạnh!",
            "⚡ Khói bốc lên từ đầu nhân vật!"
        ],
        'Surprised': [
            "😲 Nhân vật giật mình, mắt mở to!",
            "❗ Miệng há hốc, tay che miệng!",
            "✨ Ánh mắt ngạc nhiên thú vị!"
        ],
        'Neutral': [
            "😐 Nhân vật đứng bình thường...",
            "🤔 Biểu cảm trung tính, thư giãn...",
            "💭 Đang chờ đợi cảm xúc mới..."
        ]
    }

    emotion_list = descriptions.get(emotion, descriptions['Neutral'])

    # Chọn mô tả dựa trên cường độ
    if intensity > 80:
        return emotion_list[0]  # Mô tả mạnh nhất
    elif intensity > 50:
        return emotion_list[1]  # Mô tả trung bình
    else:
        return emotion_list[2]  # Mô tả nhẹ nhất



def save_detection_image(frame, gender, emotion):
    """Lưu ảnh và copy vào clipboard"""
    try:
        # Tạo thư mục nếu chưa có
        if not os.path.exists("saved_images"):
            os.makedirs("saved_images")

        # Tạo tên file với thời gian và kết quả
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_images/detection_{timestamp}_{gender}_{emotion}.jpg"

        # Lưu ảnh vào file
        cv2.imwrite(filename, frame)
        print(f"📸 Đã lưu ảnh: {filename}")

        # 📋 COPY ẢNH VÀO CLIPBOARD (Windows)
        try:
            # Chuyển đổi từ BGR (OpenCV) sang RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Tạo file tạm để copy
            temp_path = f"temp_clipboard_{timestamp}.png"
            pil_image.save(temp_path, format='PNG')

            # Sử dụng Windows command để copy ảnh vào clipboard
            import subprocess
            try:
                # PowerShell command để copy ảnh vào clipboard
                ps_command = f'Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::SetImage([System.Drawing.Image]::FromFile("{os.path.abspath(temp_path)}"))'
                subprocess.run(['powershell', '-Command', ps_command], check=True, capture_output=True)
                print("📋 Ảnh đã được copy vào clipboard!")

                # Xóa file tạm sau 3 giây
                def cleanup_temp():
                    try:
                        import time
                        time.sleep(3)
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except:
                        pass

                threading.Thread(target=cleanup_temp, daemon=True).start()

            except subprocess.CalledProcessError:
                print("⚠️ Không thể copy vào clipboard bằng PowerShell")
                # Fallback: Chỉ thông báo file đã lưu

        except Exception as clipboard_error:
            print(f"⚠️ Không thể copy vào clipboard: {clipboard_error}")

        return filename
    except Exception as e:
        print(f"❌ Lỗi lưu ảnh: {e}")
        return None

def use_camera():
    global is_running
    is_running = True
    start_button.config(state="disabled")
    stop_button.config(state="normal")
    exit_button.config(state="normal")

    worker_thread = threading.Thread(target=camera_worker)
    worker_thread.start()
    
def quit_program():
    answer = messagebox.askyesno("Quit", "Do you want to exit?")
    if answer:
        root.destroy()

def cancel_feed():
    global is_running, last_frame, last_gender, last_emotion
    is_running = False
    start_button.config(state="normal")
    stop_button.config(state="disabled")

    # 📸 TỰ ĐỘNG LƯU ẢNH KHI STOP
    if last_frame is not None and last_gender and last_emotion:
        filename = save_detection_image(last_frame, last_gender, last_emotion)
        if filename:
            messagebox.showinfo("📸 Đã lưu ảnh!",
                               f"✅ Ảnh đã được lưu thành công!\n\n"
                               f"📁 File: {filename}\n"
                               f"👤 Giới tính: {last_gender}\n"
                               f"😊 Cảm xúc: {last_emotion}\n\n"
                               f"� Ảnh đã copy vào clipboard!\n"
                               f"💬 Mở ứng dụng chat → Ctrl+V để gửi ảnh!\n\n"
                               f"�📂 Xem trong thư mục 'saved_images'")
        else:
            messagebox.showwarning("⚠️ Lỗi", "Không thể lưu ảnh!")
    else:
        messagebox.showinfo("ℹ️ Thông báo", "Chưa phát hiện khuôn mặt nào để lưu!")

def camera_worker():
    global last_frame, last_gender, last_emotion
    capture = cv2.VideoCapture(0)

    while is_running:
        ret, frame = capture.read()

        # Face detection
        faces, confidences = cvlib.detect_face(frame)

        for face, confidence in zip(faces, confidences):
            # Get the coordinates of the face rectangle
            (startX, startY) = face[0], face[1]
            (endX, endY) = face[2], face[3]

            # Draw rectangle around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2) #BGR

            # Crop the detected face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Preprocess the face for gender prediction
            face_crop = cv2.resize(face_crop, (150, 150))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Predict gender
            conf_model_gender = gender_model.predict(face_crop)[0]
            idx_model_gender = np.argmax(conf_model_gender)
            label_model_gender = gender_labels[idx_model_gender]

            # Predict emotion
            conf_model_emotion = emotion_model.predict(face_crop)[0]
            idx_model_emotion = np.argmax(conf_model_emotion)
            label_model_emotion = emotion_labels[idx_model_emotion]

            label = "{},{}".format(label_model_gender, label_model_emotion)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # Write the predicted gender label on the image
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # � LƯU FRAME CUỐI CÙNG ĐỂ IN ẢNH KHI STOP
            last_frame = frame.copy()
            last_gender = label_model_gender
            last_emotion = label_model_emotion

            # � CẬP NHẬT NHÂN VẬT CẢM XÚC
            if game_active:
                update_avatar_emotion(label_model_emotion)

            # Không tự động phát nhạc nữa - chỉ phát khi bấm nút

        # Convert the image from OpenCV BGR format to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((640, 480), Image.LANCZOS)

        # Convert the PIL Image to ImageTk to display on Tkinter label
        imgtk = ImageTk.PhotoImage(image=image)

        # Update the image on the label
        image_label.configure(image=imgtk)
        image_label.image = imgtk

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    capture.release()
    cv2.destroyAllWindows()

# Main frame với gradient background
main_frame = tk.Frame(root, bg='#1a1a2e')
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Header frame với thiết kế hiện đại
header_frame = tk.Frame(main_frame, bg='#16213e', relief=tk.RAISED, bd=2)
header_frame.pack(fill=tk.X, pady=(0, 20))

# Logo và title container
logo_title_frame = tk.Frame(header_frame, bg='#16213e')
logo_title_frame.pack(fill=tk.X, padx=20, pady=15)

# Load and display HUIT logo với viền đẹp
try:
    logo_image = Image.open('Huit.png')
    logo_image = logo_image.resize((70, 70), Image.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)

    logo_container = tk.Frame(logo_title_frame, bg='#0f3460', relief=tk.RAISED, bd=3)
    logo_container.pack(side=tk.LEFT, padx=(0, 20))

    logo_label = tk.Label(logo_container, image=logo_photo, bg='#0f3460')
    logo_label.image = logo_photo  # Keep a reference
    logo_label.pack(padx=5, pady=5)
except:
    pass

# Titles frame với typography đẹp
titles_frame = tk.Frame(logo_title_frame, bg='#16213e')
titles_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)

# Title 1 - Main title
label_title = tk.Label(titles_frame,
                       text='AI FACE DETECTION SYSTEM',
                       font=("Segoe UI", 24, "bold"),
                       fg="#00d4ff",
                       bg='#16213e')
label_title.pack(anchor='w', pady=(0, 5))

# Title 2 - Subtitle
label_title2 = tk.Label(titles_frame,
                        text='Gender & Emotion Recognition | Professional Edition',
                        font=("Segoe UI", 12),
                        fg="#a8b2d1",
                        bg='#16213e')
label_title2.pack(anchor='w', pady=(0, 5))

# Title 3 - Team info
label_title3 = tk.Label(titles_frame,
                        text='NHÓM 5 - LÀNG XÌ TRUM | Ho Chi Minh City University of Industry and Trade',
                        font=("Segoe UI", 10, "bold"),
                        fg="#ff6b6b",
                        bg='#16213e')
label_title3.pack(anchor='w')

# Camera frame - điều chỉnh để có chỗ cho game panel
image_label = tk.Label(main_frame, bg='#D9EAF4')
image_label.place(x=50, y=150, width=640, height=430)

# Avatar canvas - để vẽ nhân vật hoạt hình
avatar_canvas = tk.Canvas(main_frame, bg='#2c3e50', width=220, height=200)
avatar_canvas.place(x=710, y=350, width=220, height=200)

# Buttons frame - mở rộng để có thêm nút game
buttons_frame = tk.Frame(main_frame, bg='#f0f8ff')
buttons_frame.place(x=50, y=590, width=850, height=60)

# Start button
start_button = tk.Button(buttons_frame,
                         text="🎥 START",
                         font=('Arial', 12, 'bold'),
                         fg='white',
                         bg='#27ae60',
                         activebackground='#2ecc71',
                         relief=tk.RAISED,
                         bd=3,
                         cursor='hand2',
                         command=use_camera)
start_button.place(x=50, y=15, width=100, height=40)

# Stop button
stop_button = tk.Button(buttons_frame, text="⏹ STOP",
                        font=('Arial', 12, 'bold'),
                        fg='white',
                        bg='#e74c3c',
                        activebackground='#c0392b',
                        relief=tk.RAISED,
                        bd=3,
                        cursor='hand2',
                        command=cancel_feed,
                        state="disabled")
stop_button.place(x=170, y=15, width=100, height=40)

# Exit button
exit_button = tk.Button(buttons_frame, text="❌ EXIT",
                        font=('Arial', 12, 'bold'),
                        fg='white',
                        bg='#8e44ad',
                        activebackground='#9b59b6',
                        relief=tk.RAISED,
                        bd=3,
                        cursor='hand2',
                        command=quit_program,
                        state="normal")
exit_button.place(x=290, y=15, width=100, height=40)





# Game button
game_button = tk.Button(buttons_frame, text="� AVATAR",
                       font=('Arial', 12, 'bold'),
                       fg='white',
                       bg='#e67e22',
                       activebackground='#d35400',
                       relief=tk.RAISED,
                       bd=3,
                       cursor='hand2',
                       command=start_avatar_game)
game_button.place(x=420, y=15, width=100, height=40)

# Game info panel - hiển thị thông tin game (thu nhỏ để có chỗ cho canvas)
game_info_frame = tk.Frame(main_frame, bg='#2c3e50', relief=tk.RAISED, bd=2)
game_info_frame.place(x=710, y=150, width=220, height=190)

game_info_title = tk.Label(game_info_frame,
                          text="� AVATAR ZONE",
                          font=("Segoe UI", 14, "bold"),
                          fg="#f39c12",
                          bg='#2c3e50')
game_info_title.pack(pady=10)

game_info_label = tk.Label(game_info_frame,
                          text="� Bấm AVATAR để bắt đầu chơi!",
                          font=("Segoe UI", 10),
                          fg="#ecf0f1",
                          bg='#2c3e50',
                          wraplength=180,
                          justify='left')
game_info_label.pack(pady=10, padx=10, fill='both', expand=True)

# Hướng dẫn game
game_rules = tk.Label(game_info_frame,
                     text="""🎭 CÁCH CHƠI:
• Thể hiện cảm xúc trước camera
• Nhân vật sẽ phản ứng theo
• Cảm xúc mạnh → phản ứng mạnh
• Thử các cảm xúc khác nhau!

😊 CẢM XÚC:
😊 Happy → Vui vẻ nhảy múa
😢 Sad → Buồn bã cúi đầu
😠 Angry → Nổi giận đỏ mặt
😲 Surprised → Giật mình
😐 Neutral → Bình thường""",
                     font=("Segoe UI", 8),
                     fg="#bdc3c7",
                     bg='#2c3e50',
                     justify='left')
game_rules.pack(pady=10, padx=10)


root.mainloop()

# Note: Models are already loaded from existing .h5 files
# No need to save them again here
print("Application finished!")