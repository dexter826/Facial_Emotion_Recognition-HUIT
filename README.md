# 🎭 Hệ thống Nhận diện Khuôn mặt và Cảm xúc

Dự án Deep Learning sử dụng CNN để nhận diện giới tính và cảm xúc từ khuôn mặt trong thời gian thực.

![GUI](img/GUI.png)

## 📋 Tổng quan

Hệ thống sử dụng hai mô hình CNN riêng biệt:

- **Phân loại giới tính**: Nam/Nữ (Gender Classification)
- **Nhận diện cảm xúc**: 5 cảm xúc chính (Angry, Happy, Neutral, Sad, Surprise)

## 🚀 Tính năng

- ✅ **Real-time detection**: Nhận diện trực tiếp từ webcam
- ✅ **Multi-face support**: Xử lý nhiều khuôn mặt cùng lúc
- ✅ **High accuracy**: Độ chính xác cao với dataset chất lượng
- ✅ **User-friendly GUI**: Giao diện đơn giản, dễ sử dụng
- ✅ **Balanced dataset**: Dataset đã được cân bằng để tối ưu hiệu suất

## 📊 Kết quả hiệu suất

### Mô hình Gender:

- **Training Accuracy**: 92.85%
- **Validation Accuracy**: 95.38%
- **Test Accuracy**: 95.38%

### Mô hình Emotion:

- **Training Accuracy**: 65.41%
- **Validation Accuracy**: 71.36%
- **Test Accuracy**: 71.36%

## 🛠️ Cài đặt

### Yêu cầu hệ thống:

- Python 3.8+
- Webcam
- GPU (khuyến nghị) hoặc CPU

### Cài đặt dependencies:

```bash
pip install tensorflow
pip install opencv-python
pip install cvlib
pip install pillow
pip install numpy
pip install tkinter
```

## 📁 Cấu trúc dự án

```
IdentifyFace/
├── dataset/
│   ├── emotion/
│   │   ├── train/          # Dataset emotion training
│   │   ├── val/            # Dataset emotion validation
│   │   └── backup_original/ # Backup dataset gốc
│   └── gender/
│       ├── train_folders/   # Dataset gender training
│       └── valid_folders/   # Dataset gender validation
├── models/
│   ├── Emotion1.h5         # Mô hình emotion đã train
│   └── Gender1.h5          # Mô hình gender đã train
├── scripts/
│   ├── emotion-training.py # Script training emotion
│   ├── gender-training.py  # Script training gender
│   └── balance_emotion_dataset.py # Script cân bằng dataset
├── realtime.py             # Ứng dụng chính
├── README.md
└── workflow.md
```

## 🎯 Hướng dẫn sử dụng

### 1. Chạy ứng dụng:

```bash
python realtime.py
```

### 2. Sử dụng giao diện:

- Nhấn **"Bắt đầu"** để khởi động camera
- Nhấn **"Dừng"** để tạm dừng
- Nhấn **"Thoát"** để đóng ứng dụng

## 📚 Dataset

### 1. Emotion Dataset

- **Nguồn**: [Emotion Recognition Dataset](https://www.kaggle.com/datasets/karthickmcw/emotion-recognition-dataset)
- **Mô tả**: Dataset chứa ảnh khuôn mặt với 5 cảm xúc chính
- **Kích thước**: ~12,825 ảnh (đã cân bằng)
- **Chất lượng**: Ảnh chất lượng cao (48-80KB/file)

### 2. Gender Dataset

- **Nguồn**: [Gender Classification Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset)
- **Mô tả**: Dataset phân loại giới tính Female/Male (đã được tối ưu)
- **Kích thước**: ~57,000 ảnh (đã cân bằng hoàn hảo)
- **Cấu trúc**:
  - Train: ~23,000 ảnh/class (female, male)
  - Val: ~5,500 ảnh/class (female, male)
- **Tỷ lệ**: 80% train / 20% validation (tỷ lệ lý tưởng)
- **Đặc điểm**: Dataset đã được tối ưu, không cần xử lý thêm

## 🔧 Training mô hình

### 1. Training mô hình emotion:

```bash
python emotion-training.py
```

### 2. Training mô hình gender:

```bash
python gender-training.py
```

## ⚙️ Cấu hình mô hình

### Kiến trúc CNN:

- **Input size**: 150x150x3 (RGB)
- **Convolutional layers**: 5 lớp với filters tăng dần
- **Pooling**: MaxPooling2D (2x2)
- **Dense layers**: 2 lớp với Dropout
- **Output**: Softmax activation

### Hyperparameters:

- **Optimizer**: RMSprop (lr=0.001)
- **Loss**: Categorical crossentropy
- **Batch size**: 32
- **Epochs**: 20
- **Data augmentation**: Rotation, shift, zoom, flip

## 📈 Cải thiện hiệu suất

### Để tăng độ chính xác:

1. **Tăng epochs**: 30-50 epochs
2. **Fine-tuning learning rate**: 0.0001
3. **Transfer learning**: Sử dụng pre-trained models
4. **Data augmentation mạnh hơn**
5. **Ensemble methods**: Kết hợp nhiều mô hình

### Để tăng tốc độ:

1. **Giảm input size**: 96x96 thay vì 150x150
2. **Model quantization**: Giảm kích thước mô hình
3. **GPU acceleration**: Sử dụng CUDA

## 👥 Đóng góp

Nhóm 5 - Môn Deep Learning:

- **Trần Công Minh**
- **Nguyễn Chí Tài**
- **Tạ Nguyên Vũ**
- **Lê Đức Trung**

_Trường Đại học Công Thương (HUIT)_

## 📄 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

## 🔗 Links

- [Emotion Dataset](https://www.kaggle.com/datasets/karthickmcw/emotion-recognition-dataset)
- [Gender Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset)
- [Workflow chi tiết](workflow.md)
