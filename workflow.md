# Workflow - Cách thức hoạt động của mô hình CNN trong Hệ thống Nhận diện Khuôn mặt và Cảm xúc

## Giới thiệu tổng quan

Dự án này sử dụng hai mô hình CNN (Convolutional Neural Network) riêng biệt để thực hiện hai nhiệm vụ chính:

1. **Phân loại giới tính** (Nam/Nữ) - `Gender2.h5`
2. **Nhận diện cảm xúc** (Tức giận, Ghê tởm, Sợ hãi, Vui vẻ, Bình thường, Buồn, Ngạc nhiên) - `Emotion1.h5`

## Bộ dữ liệu (Dataset) sử dụng

### 1. Dataset cho Nhận diện Cảm xúc

**Nguồn**: FER-2013 (Facial Expression Recognition 2013) từ Kaggle

- **Link**: https://www.kaggle.com/datasets/msambare/fer2013
- **Mô tả**: Bộ dữ liệu chứa khoảng 30,000 ảnh khuôn mặt grayscale kích thước 48x48 pixel
- **Phân loại**: 7 loại cảm xúc (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) - **SỬ DỤNG ĐẦY ĐỦ**
- **Cấu trúc**:
  - Training set: ~28,000 ảnh
  - Test set: ~7,000 ảnh
- **Đặc điểm**: Ảnh được thu thập từ nhiều nguồn khác nhau, đa dạng về độ tuổi, giới tính, và điều kiện ánh sáng

### 2. Dataset cho Phân loại Giới tính

**Nguồn**: Gender Classification Dataset từ Kaggle

- **Link**: https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset
- **Mô tả**: Bộ dữ liệu chứa hình ảnh khuôn mặt đã được cắt và làm sạch
- **Phân loại**: 2 lớp (Male, Female)
- **Cấu trúc**:
  - Thư mục "man": Chứa ảnh khuôn mặt nam
  - Thư mục "woman": Chứa ảnh khuôn mặt nữ
- **Tổng số**: Khoảng 47,000+ ảnh
- **Đặc điểm**: Ảnh đã được tiền xử lý, cắt và căn chỉnh khuôn mặt

### 3. Tiền xử lý dữ liệu

#### Cho mô hình Emotion:

- **Resize**: Từ 48x48 → 150x150 pixels
- **Color conversion**: Grayscale → RGB (duplicate channels)
- **Normalization**: Pixel values từ [0-255] → [0-1]
- **Data Augmentation**: Rotation, shift, zoom, flip

#### Cho mô hình Gender:

- **Input size**: 150x150x3 (RGB)
- **Normalization**: Pixel values từ [0-255] → [0-1]
- **Data Augmentation**: Rotation, shift, zoom, flip

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
├── Dense(128) + ReLU + Dropout(0.5)
└── Output: Dense(2) + Softmax → [Male, Female] - **AUTO-DETECT CLASSES**
```

### 2. Mô hình Nhận diện Cảm xúc (Emotion Recognition)

#### Cấu trúc mạng:

```
Input Layer: (150, 150, 3) - Ảnh RGB kích thước 150x150
├── Conv2D(16 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(32 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(64 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(128 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(256 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Flatten
├── Dense(128) + ReLU + Dropout(0.5)
└── Output: Dense(7) + Softmax → [Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise] - **ĐẦY ĐỦ 7 EMOTIONS**
```

## Quy trình hoạt động của hệ thống

### Giai đoạn 1: Huấn luyện mô hình (Training Phase)

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

# Define class labels
gender_labels = ['Male', 'Female']
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']  # Full FER-2013 emotions
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

### 1. Techniques sử dụng

- **Data Augmentation**: Tăng đa dạng dữ liệu huấn luyện
- **Early Stopping**: Dừng huấn luyện khi không còn cải thiện
- **Dropout**: Regularization để tránh overfitting
- **Batch Normalization**: Ổn định quá trình huấn luyện

### 2. Real-time Performance

- **Input size**: 150x150 (cân bằng giữa accuracy và speed)
- **Model size**: Compact cho inference nhanh
- **Threading**: Xử lý đa luồng cho GUI responsiveness

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
