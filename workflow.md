# Workflow - Cách thức hoạt động của mô hình CNN trong Hệ thống Nhận diện Khuôn mặt và Cảm xúc

## Giới thiệu tổng quan

Dự án này sử dụng hai mô hình CNN (Convolutional Neural Network) riêng biệt để thực hiện hai nhiệm vụ chính:

1. **Phân loại giới tính** (Female/Male) - `Gender1.h5`
2. **Nhận diện cảm xúc** (Angry, Happy, Neutral, Sad, Surprise) - `Emotion1.h5`

## Bộ dữ liệu (Dataset) sử dụng

### 1. Dataset cho Nhận diện Cảm xúc

**Nguồn**: Emotion Recognition Dataset từ Kaggle

- **Link**: https://www.kaggle.com/datasets/karthickmcw/emotion-recognition-dataset
- **Mô tả**: Bộ dữ liệu chứa ảnh khuôn mặt chất lượng cao với 5 cảm xúc chính
- **Phân loại**: 5 loại cảm xúc (Angry, Happy, Neutral, Sad, Surprise)
- **Cấu trúc**:
  - Training set: dataset/emotion/train/ (đã cân bằng ~4,800 ảnh)
  - Validation set: dataset/emotion/val/ (đã cân bằng ~1,200 ảnh)
  - Backup: dataset/emotion/backup_original/ (dataset gốc)
- **Đặc điểm**:
  - Ảnh chất lượng cao (48-80KB/file)
  - Đã được cân bằng dữ liệu (~1,200 ảnh/emotion)
  - Tỷ lệ train/val: 80%/20%

### 2. Dataset cho Phân loại Giới tính

**Nguồn**: Gender Classification Dataset từ Kaggle

- **Link**: https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset
- **Mô tả**: Bộ dữ liệu chứa hình ảnh khuôn mặt đã được cắt và làm sạch
- **Phân loại**: 2 lớp (Female, Male)
- **Cấu trúc**:
  - Training set: dataset/gender/train_folders/
    - female/: ~23,000 ảnh khuôn mặt nữ
    - male/: ~23,000 ảnh khuôn mặt nam
  - Validation set: dataset/gender/valid_folders/
    - female/: ~5,500 ảnh khuôn mặt nữ
    - male/: ~5,500 ảnh khuôn mặt nam
- **Tổng số**: ~57,000 ảnh (đã được cân bằng và tối ưu)
- **Tỷ lệ Train/Val**: ~80%/20% (tỷ lệ lý tưởng)
- **Đặc điểm**:
  - Ảnh đã được tiền xử lý, cắt và căn chỉnh khuôn mặt
  - Dataset hoàn toàn cân bằng giữa nam và nữ
  - Chất lượng ảnh tốt, phù hợp cho deep learning
  - Đã được tối ưu hóa số lượng để training hiệu quả

### 3. Tiền xử lý dữ liệu

#### Cho mô hình Emotion:

- **Input size**: 150x150x3 (RGB) - tối ưu cho độ chính xác
- **Normalization**: Pixel values từ [0-255] → [0-1]
- **Data Augmentation**: Rotation (20°), shift (0.2), shear (0.2), zoom (0.2), horizontal flip
- **Dataset balancing**: Sử dụng script balance_emotion_dataset.py

#### Cho mô hình Gender:

- **Input size**: 150x150x3 (RGB) - tương thích với emotion model
- **Normalization**: Pixel values từ [0-255] → [0-1]
- **Data Augmentation**: Rotation (20°), shift (0.2), zoom (0.2), horizontal flip
- **Balanced dataset**: Dataset đã cân bằng sẵn

## Kiến trúc mô hình CNN

### 1. Mô hình Phân loại Giới tính (Gender Classification)

#### Cấu trúc mạng:

```
Input Layer: (150, 150, 3) - Ảnh RGB kích thước 150x150
├── Conv2D(32 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(64 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(128 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(256 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Flatten
├── Dense(256) + ReLU + Dropout(0.5)
├── Dense(128) + ReLU + Dropout(0.3)
└── Output: Dense(2) + Softmax → [Female, Male] - **Class indices: female=0, male=1**
```

### 2. Mô hình Nhận diện Cảm xúc (Emotion Recognition)

#### Cấu trúc mạng:

```
Input Layer: (150, 150, 3) - Ảnh RGB kích thước 150x150
├── Conv2D(32 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(64 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(128 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(256 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Flatten
├── Dense(256) + ReLU + Dropout(0.5)
├── Dense(128) + ReLU + Dropout(0.3)
└── Output: Dense(5) + Softmax → [Angry, Happy, Neutral, Sad, Surprise] - **Class indices: Angry=0, Happy=1, Neutral=2, Sad=3, Surprise=4**
```

## Quy trình hoạt động của hệ thống

### Giai đoạn 1: Huấn luyện mô hình (Training Phase)

#### 1.0 Cân bằng dataset (khuyến nghị cho emotion)

```python
# Chạy script cân bằng dataset emotion
python balance_emotion_dataset.py

# Kết quả:
# - Tạo backup dataset gốc
# - Cân bằng ~1,200 files/emotion
# - Tỷ lệ train/val: 80%/20%
# - Tổng: ~6,000 files (thay vì 12,825)
```

#### 1.1 Chuẩn bị dữ liệu

- **Data Augmentation**: Sử dụng `ImageDataGenerator` với các kỹ thuật:
  - Rescaling: Chuẩn hóa pixel values (0-1)
  - Rotation: Xoay ảnh ±20°
  - Width/Height shift: Dịch chuyển ±20%
  - Shear transformation: Biến dạng nghiêng
  - Zoom: Phóng to/thu nhỏ ±20%
  - Horizontal flip: Lật ngang ngẫu nhiên

#### 1.2 Cấu hình huấn luyện

- **Optimizer**: RMSprop với learning rate = 0.001
- **Loss function**: Categorical crossentropy
- **Metrics**: Accuracy
- **Batch size**: 32
- **Epochs**: 20 (với Early Stopping)
- **Callbacks**: EarlyStopping (patience=3) để tránh overfitting

#### 1.3 Quá trình học

1. **Forward Pass**: Dữ liệu đi qua các lớp CNN
2. **Feature Extraction**: Các lớp Conv2D trích xuất đặc trưng
3. **Pooling**: Giảm kích thước và tăng tính bất biến
4. **Classification**: Lớp Dense cuối phân loại
5. **Backpropagation**: Cập nhật trọng số dựa trên loss

### Giai đoạn 2: Ứng dụng thời gian thực (Inference Phase)

#### 2.1 Khởi tạo hệ thống

```python
# Load pre-trained models
gender_model = load_model('Gender1.h5', compile=False)
emotion_model = load_model('Emotion1.h5', compile=False)

# Define class labels (phải khớp với class indices của dataset)
gender_labels = ['Female', 'Male']  # female=0, male=1
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']  # theo thứ tự alphabet
```

#### 2.2 Quy trình xử lý frame từ camera

```
1. Capture frame từ camera
2. Face Detection (sử dụng cvlib)
3. Crop và resize face region → (150, 150, 3)
4. Preprocessing:
   - Normalize pixel values (0-1)
   - Expand dimensions cho batch processing
5. Inference:
   - Gender prediction
   - Emotion prediction
6. Post-processing:
   - Áp dụng softmax để có xác suất
   - Lấy class có xác suất cao nhất
7. Display results trên GUI
```

### Giai đoạn 3: Xử lý kết quả (Results Processing)

#### 3.1 Face Detection

- Sử dụng thư viện `cvlib` để phát hiện khuôn mặt
- Trả về bounding box coordinates và confidence score

#### 3.2 Prediction Pipeline

```python
def predict_gender_emotion(face_crop):
    # Preprocessing
    face_array = img_to_array(face_crop)
    face_array = np.expand_dims(face_array, axis=0)
    face_array = face_array / 255.0

    # Gender prediction
    gender_prediction = gender_model.predict(face_array)
    gender_label = gender_labels[np.argmax(gender_prediction)]
    gender_confidence = np.max(gender_prediction)

    # Emotion prediction
    emotion_prediction = emotion_model.predict(face_array)
    emotion_label = emotion_labels[np.argmax(emotion_prediction)]
    emotion_confidence = np.max(emotion_prediction)

    return gender_label, gender_confidence, emotion_label, emotion_confidence
```

## Đặc điểm kỹ thuật của CNN

### 1. Convolutional Layers

- **Chức năng**: Trích xuất đặc trưng từ ảnh (edges, patterns, textures)
- **Kernel size**: 3x3 (tối ưu cho việc phát hiện đặc trưng cục bộ)
- **Activation**: ReLU (tính toán nhanh, tránh vanishing gradient)

### 2. MaxPooling Layers

- **Chức năng**: Giảm chiều, tăng tính bất biến vị trí
- **Pool size**: 2x2 (giảm kích thước ảnh xuống 1/4)
- **Lợi ích**: Giảm overfitting, tăng tốc độ tính toán

### 3. Dense Layers

- **Fully Connected**: Kết hợp tất cả đặc trưng đã trích xuất
- **Dropout**: 0.5 để regularization, tránh overfitting
- **Output layer**: Softmax cho phân loại multi-class

## Hiệu suất và Tối ưu hóa

### 1. Kết quả Training thực tế

#### Mô hình Gender (Gender1.h5):

- **Training Accuracy**: 92.85%
- **Validation Accuracy**: 95.38%
- **Test Accuracy**: 95.38%
- **Training Loss**: 0.2023
- **Validation Loss**: 0.1493
- **Dataset**: ~57,000 ảnh (23k train + 5.5k val mỗi class)
- **Đánh giá**: ✅ Rất tốt, sẵn sàng production
- **Lý do thành công**: Dataset cân bằng hoàn hảo và đã được tối ưu số lượng

#### Mô hình Emotion (Emotion1.h5):

- **Training Accuracy**: 65.41%
- **Validation Accuracy**: 71.36%
- **Test Accuracy**: 71.36%
- **Training Loss**: 0.8909
- **Validation Loss**: 0.7254
- **Dataset**: ~12,825 ảnh (chưa cân bằng) hoặc ~6,000 ảnh (đã cân bằng)
- **Đánh giá**: ✅ Khá tốt, có thể cải thiện bằng cách cân bằng dataset

### 3. Techniques sử dụng

- **Data Augmentation**: Tăng đa dạng dữ liệu huấn luyện
- **Early Stopping**: Dừng huấn luyện khi không còn cải thiện
- **Dropout**: Regularization để tránh overfitting
- **Dataset Balancing**: Cân bằng dữ liệu để tránh bias (quan trọng cho emotion)
- **Quality over Quantity**: Sử dụng ảnh chất lượng cao
- **Optimal Dataset Size**: Gender đã tối ưu, Emotion cần cân bằng

### 4. Real-time Performance

- **Input size**: 150x150 (cân bằng giữa accuracy và speed)
- **Model size**: Compact cho inference nhanh
- **Threading**: Xử lý đa luồng cho GUI responsiveness
- **Face detection**: Sử dụng cvlib cho tốc độ tối ưu

## Ưu điểm của kiến trúc này

1. **Modularity**: Hai mô hình riêng biệt cho hai tác vụ khác nhau
2. **Efficiency**: Kích thước mô hình phù hợp cho real-time processing
3. **Scalability**: Có thể mở rộng thêm các tác vụ khác
4. **Accuracy**: Kiến trúc CNN phù hợp cho computer vision tasks

## Kết luận

Hệ thống sử dụng hai mô hình CNN được thiết kế tối ưu cho việc phân loại giới tính và nhận diện cảm xúc trong thời gian thực. Kiến trúc đơn giản nhưng hiệu quả, với việc sử dụng các kỹ thuật hiện đại như data augmentation, dropout, và early stopping để đạt được hiệu suất tốt nhất.

---

**Nhóm 5 - Môn Deep Learning**

- Trần Công Minh
- Nguyễn Chí Tài
- Tạ Nguyên Vũ
- Lê Đức Trung

_Trường Đại học Công Thương (HUIT)_
