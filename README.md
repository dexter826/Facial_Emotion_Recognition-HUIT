# Hệ thống Nhận diện Khuôn mặt và Cảm xúc - HUIT

![GUI Demo](img/GUI.png)

## Giới thiệu

Hệ thống Nhận diện Khuôn mặt và Cảm xúc là một ứng dụng sử dụng trí tuệ nhân tạo và học sâu (Deep Learning) để phát hiện khuôn mặt, nhận diện giới tính và phân tích cảm xúc theo thời gian thực. Dự án này được phát triển bởi nhóm 5 gồm 4 sinh viên khoa Công nghệ Thông tin, trường Đại học Công Thương (HUIT).

## Tính năng chính

- Phát hiện và nhận diện khuôn mặt theo thời gian thực qua camera
- Phân loại giới tính (Nam/Nữ)
- Nhận diện cảm xúc cơ bản (Bình thường, Vui vẻ, Buồn, Ngạc nhiên, Tức giận)
- Hiển thị kết quả phân tích trực quan
- Đo thời gian xử lý và độ tin cậy của kết quả

## Công nghệ sử dụng

- Python
- TensorFlow/Keras (Deep Learning frameworks)
- OpenCV (Xử lý ảnh)
- Tkinter (Giao diện người dùng đồ họa)
- Convolutional Neural Networks (CNN) cho phân loại giới tính và cảm xúc

## Thành viên nhóm

- Trần Công Minh
- Nguyễn Chí Tài
- Tạ Nguyên Vũ
- Lê Đức Trung

## Cấu trúc dự án

- **emotion-training.ipynb**: Notebook huấn luyện mô hình nhận diện cảm xúc
- **Emotion1.h5**: Mô hình học sâu đã được huấn luyện cho nhận diện cảm xúc
- **gender-training.ipynb**: Notebook huấn luyện mô hình phân loại giới tính
- **Gender1.h5**: Mô hình học sâu đã được huấn luyện cho phân loại giới tính
- **realtime.py**: Mã nguồn chính của ứng dụng thời gian thực
- **img/**: Thư mục chứa hình ảnh và tài nguyên đồ họa
- **Report/**: Tài liệu báo cáo và tài liệu dự án

## Hướng dẫn cài đặt

1. Cài đặt các thư viện cần thiết:

```bash
pip install tensorflow opencv-python cvlib pillow
```

2. Chạy ứng dụng:

```bash
python realtime.py
```

## Quy trình hoạt động

1. Hệ thống thu nhận hình ảnh từ camera theo thời gian thực
2. Phát hiện và định vị khuôn mặt trong khung hình
3. Tiền xử lý khuôn mặt để chuẩn hóa
4. Thực hiện phân loại giới tính và cảm xúc sử dụng các mô hình CNN đã huấn luyện
5. Hiển thị kết quả lên giao diện với các thông tin phụ trợ

## Quá trình phát triển

Các mô hình học sâu được huấn luyện trên bộ dữ liệu lớn với nhiều khuôn mặt đa dạng về giới tính và cảm xúc. Quá trình huấn luyện sử dụng kỹ thuật tăng cường dữ liệu (data augmentation) và kiến trúc mạng nơ-ron tích chập nhiều tầng để đạt độ chính xác cao trong việc phân loại.

## Liên hệ

- Trần Công Minh: tcongminh1604@gmail.com

---

© 2025 Ho Chi Minh University Of Industry And Trade - AI Facial Emotion Recognition System
