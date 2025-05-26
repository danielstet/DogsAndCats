# Task 4 – Cats vs Dogs Classification  
**Course:** Introduction to Models  
**University:** Ariel University  

## 📝 Overview

This project is part of Task 4 in the "Introduction to Models" course at Ariel University. The goal is to train a **Convolutional Neural Network (CNN)** to classify images of **cats** and **dogs**.

---

## 📁 Dataset

The original dataset link pointed to a closed competition, so I used an alternative dataset available here:

🔗 [Download Dataset](https://drive.google.com/drive/u/0/folders/1dZvL1gi5QLwOGrfdn9XEsi4EnXx535bD)

Make sure to place the extracted files in the `CNNKerasDataSet/` folder in your working directory.

---

## ⚙️ Setup Instructions

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
    ```
2. Launch Jupyter Notebook:
   ```bash
    jupyter notebook
    ```
## 📊 Data Preparation:

Load data:
   ```bash
   X_train = np.loadtxt('CNNKerasDataSet/input.csv', delimiter=',')
   Y_train = np.loadtxt('CNNKerasDataSet/labels.csv', delimiter=',')
   
   X_test = np.loadtxt('CNNKerasDataSet/input_test.csv', delimiter=',')
   Y_test = np.loadtxt('CNNKerasDataSet/labels_test.csv', delimiter=',')
   
   Reshape data to match CNN input
   X_train = X_train.reshape(len(X_train), 100, 100, 3)
   Y_train = Y_train.reshape(len(Y_train), 1)
   
   X_test = X_test.reshape(len(X_test), 100, 100, 3)
   Y_test = Y_test.reshape(len(Y_test), 1)
   ```
Normalize pixel values:
   ```bash
   X_train = X_train / 255.0
   X_test = X_test / 255.0
   
   Visualize a random training image
   idx = random.randint(0, len(X_train))
   plt.imshow(X_train[idx])
   plt.show()
   ```
## 🧠 Model Architecture:
   ```bash
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
   
   model = Sequential([
       Input(shape=(100, 100, 3)),
       Conv2D(32, (3, 3), activation='relu'),
       MaxPooling2D((2, 2)),
       Conv2D(32, (3, 3), activation='relu'),
       MaxPooling2D((2, 2)),
       Flatten(),
       Dense(64, activation='relu'),
       Dense(1, activation='sigmoid')  # Binary classification
   ])
   ```
## 🛠️ Compilation and Training:
   ```bash
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.fit(X_train, Y_train, epochs=15, batch_size=64)
   ```
## 📈 Evaluation:
   ```bash
   loss, accuracy = model.evaluate(X_test, Y_test)
   ```
## The model achieves approximately 67% accuracy on the test set.

## 📚 Insights from the Second Project
The second source takes a different approach to solving the same problem. 
Instead of training a model from scratch, the author uses a well-established technique: fine-tuning a pre-trained model. 
Specifically, he chose the VGG16 architecture and achieved a significantly higher accuracy of 96.5%. 
Additionally, he trained his model on a much larger dataset containing 25,000 images, compared to the 1,000 images I used.

In general, larger datasets tend to yield better model performance in the long run.

## 🔍 VGG16 Architecture Highlights (as described in the source):
1. Utilization of very small convolutional filters (e.g., 3×3 and 1×1).

2. Max pooling layers with 2×2 filters and matching stride.

3. Stacking multiple convolutional layers before pooling to form well-defined blocks.

4. Repetitive use of the convolution → pooling block pattern.

5. Construction of very deep models (e.g., 16 or 19 layers).

## 🔑 Key Differences Between Our Approaches

| Aspect         | My Approach             | His Approach                    |
|----------------|-------------------------|----------------------------------|
| **Model**      | Built from scratch       | Fine-tuned pre-trained VGG16     |
| **Dataset Size** | 1,000 images           | 25,000 images                    |
| **Epochs**     | 15                      | 50                               |
| **Accuracy**   | ~67%                    | 96.5%                            |

## 📚 What You Can Learn from the Second Project
1. The Power of Transfer Learning
Instead of building a model from scratch, the author used transfer learning — specifically by fine-tuning a pre-trained VGG16 model.
This approach leverages the knowledge learned from a large dataset (like ImageNet) and adapts it to your specific task. Benefits include:

- Faster training

- Better performance with less data

- Avoids the need to design complex architectures from scratch

2. Importance of Dataset Size
The second project used 25,000 images while you used only 1,000. Generally:

- More data leads to better generalization

- Reduces the risk of overfitting

- Helps the model learn more diverse features

3. Impact of Training Duration
He trained for 50 epochs, while you stopped at 15.
Training for more epochs (with proper validation to avoid overfitting) can:

- Help the model learn better feature representations

- Improve final accuracy

- Require more compute resources, so careful monitoring is essential
