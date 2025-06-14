import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

def create_model():
    """
    Tạo mô hình CNN cho nhận diện cảm xúc
    """
    model = Sequential()
    
    # Các lớp Convolutional
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    
    # Flatten
    model.add(Flatten())
    
    # Fully-connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output layer cho 5 emotions
    model.add(Dense(5, activation='softmax'))
    
    return model

def plot_training_history(history, model_name="Emotion"):
    """
    Vẽ biểu đồ accuracy và loss từ lịch sử training
    """
    # Tạo figure cho accuracy
    plt.figure(figsize=(10, 6))

    # Vẽ biểu đồ accuracy
    plt.plot(history.history['accuracy'], 'b-', label='train', linewidth=2)
    plt.plot(history.history['val_accuracy'], 'orange', label='validation', linewidth=2)
    plt.title('model accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Lưu biểu đồ accuracy
    accuracy_filename = f'{model_name.lower()}_model_accuracy.png'
    plt.tight_layout()
    plt.savefig(accuracy_filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Tạo figure cho loss
    plt.figure(figsize=(10, 6))

    # Vẽ biểu đồ loss
    plt.plot(history.history['loss'], 'b-', label='train', linewidth=2)
    plt.plot(history.history['val_loss'], 'orange', label='validation', linewidth=2)
    plt.title('model loss', fontsize=16, fontweight='bold')
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Lưu biểu đồ loss
    loss_filename = f'{model_name.lower()}_model_loss.png'
    plt.tight_layout()
    plt.savefig(loss_filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Biểu đồ đã được lưu:")
    print(f"- {accuracy_filename}")
    print(f"- {loss_filename}")

def train_emotion_model():
    """
    Huấn luyện mô hình nhận diện cảm xúc
    """
    print("Kiểm tra thư mục dataset...")

    # Sử dụng thư mục dataset mới với nhãn viết hoa đầu
    train_dir = "dataset/emotion/train"
    valid_dir = "dataset/emotion/val"

    if not os.path.exists(train_dir):
        print(f"Lỗi: Không tìm thấy thư mục {train_dir}")
        print("Vui lòng kiểm tra dataset emotion")
        return None, None

    if not os.path.exists(valid_dir):
        print(f"Lỗi: Không tìm thấy thư mục {valid_dir}")
        print("Vui lòng kiểm tra dataset emotion")
        return None, None

    print("Tạo data generators...")
    
    # Tạo bộ tăng cường dữ liệu (data augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1.0/255, 
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Áp dụng data augmentation cho tập huấn luyện
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    # Tạo bộ tăng cường dữ liệu cho tập validation (không thay đổi dữ liệu)
    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    # Áp dụng data augmentation cho tập validation
    validation_generator = validation_datagen.flow_from_directory(
        directory=valid_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )
    
    print("\nTạo mô hình...")
    
    # Khởi tạo mô hình CNN
    model = create_model()
    
    # Compile mô hình
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Hiển thị cấu trúc mô hình
    model.summary()
    
    print("\nBắt đầu huấn luyện...")
    
    # Callback để dừng sớm nếu không cải thiện
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Huấn luyện mô hình
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("\nLưu mô hình...")
    
    # Lưu mô hình
    model.save('Emotion1.h5')
    print("Đã lưu mô hình vào Emotion1.h5")
    
    # Hiển thị kết quả cuối cùng
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\nKết quả cuối cùng:")
    print(f"Training Accuracy: {final_train_acc:.4f}")
    print(f"Validation Accuracy: {final_val_acc:.4f}")
    print(f"Training Loss: {final_train_loss:.4f}")
    print(f"Validation Loss: {final_val_loss:.4f}")
    
    return model, history

def test_model():
    """
    Kiểm tra mô hình đã huấn luyện
    """
    print("\nKiểm tra mô hình...")

    # Kiểm tra file mô hình
    if not os.path.exists('Emotion1.h5'):
        print("Lỗi: Không tìm thấy file Emotion1.h5")
        print("Vui lòng huấn luyện mô hình trước")
        return None, None

    # Load mô hình
    model = tf.keras.models.load_model('Emotion1.h5')

    # Tạo test generator
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_generator = test_datagen.flow_from_directory(
        directory="dataset/emotion/val",
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Đánh giá mô hình
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Hiển thị class indices
    print(f"\nClass indices: {test_generator.class_indices}")
    
    return test_accuracy, test_loss

if __name__ == "__main__":
    print("=== HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN CẢM XÚC ===")
    print("Sử dụng dataset Emotion với 5 cảm xúc:")
    print("- Angry, Happy, Neutral, Sad, Surprise")
    print("Tương thích với realtime.py\n")

    # Kiểm tra dataset có tồn tại không
    if not os.path.exists("dataset/emotion/train"):
        print("Lỗi: Dataset emotion/train không tồn tại!")
        print("Vui lòng kiểm tra thư mục dataset/emotion/")
        exit(1)

    if not os.path.exists("dataset/emotion/val"):
        print("Lỗi: Dataset emotion/val không tồn tại!")
        print("Vui lòng kiểm tra thư mục dataset/emotion/")
        exit(1)

    # Huấn luyện mô hình
    model, history = train_emotion_model()

    if model is not None and history is not None:
        # Vẽ biểu đồ training history
        print("\nTạo biểu đồ training history...")
        plot_training_history(history, "Emotion")

        # Kiểm tra mô hình
        test_accuracy, test_loss = test_model()

        if test_accuracy is not None:
            print("\n=== HOÀN THÀNH ===")
            print("Mô hình đã được lưu vào Emotion1.h5")
            print("Biểu đồ training đã được tạo")
            print("Có thể sử dụng với realtime.py")
        else:
            print("\nLỗi trong quá trình kiểm tra mô hình")
    else:
        print("\nLỗi trong quá trình huấn luyện mô hình")
