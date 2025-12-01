## 🌟 Custom GPT-2 Model: Tiny Stories Edition 📚

### Welcome to the **Custom GPT-2 Model** repository! 🚀 This project showcases the training of a **custom GPT-2 model** on a subset of the **Tiny Stories dataset**. The model is built from scratch, incorporating key components like **multi-head attention mechanisms** and **transformer blocks**. Let's dive into the details! 💡

 ---

## 📖 **Overview**

## This project demonstrates:
### - 🧠 **Custom GPT-2 Architecture**: Built from scratch using PyTorch, including **multi-head attention** and **transformer blocks**.
### - 📚 **Dataset**: Trained on a subset of the **Tiny Stories dataset**, which contains short, engaging stories.
### - ⚡ **Inference**: Generate creative and coherent text outputs.

 ---

## 🛠️ **Key Components**

### 🔗 **Multi-Head Attention**
#### The model uses a custom implementation of **multi-head attention**, enabling it to focus on different parts of the input sequence simultaneously.

### 🧩 **Transformer Blocks**
### The architecture includes **transformer blocks** with:
### - Layer normalization
### - Feed-forward networks
### - Residual connections

### 📊 **Training Pipeline**
### - **Optimizer**: AdamW with weight decay for stable training.
### - **Loss Function**: Cross-entropy loss for token prediction.
### - **Evaluation**: Periodic validation to monitor performance.

 ---

## 📂 **Project Structure**

 ```
 src/
 ├── GPT.py                      # GPT model implementation
 ├── GPT_Tiny_Stories.py         # Model training
 ├── transformer.py              # Transformer block and layer normalization
 ├── attention_with_trainable.py # Multi-head attention implementation
 ├── DataLoader.py               # Data loading and preprocessing
 ├── inference.py                # Inference script for text generation
 ├── loss.py                     # Loss calculation and training utilities
 
 ```

---

## 🚀 **How to Use**

### 1️⃣ **Clone the Repository**
```bash
   git clone https://github.com/your-username/custom-gpt2-tiny-stories.git
   cd custom-gpt2-tiny-stories
 ```

### 2️⃣ **Install Dependencies**
# Ensure you have Python 3.8+ and install the required libraries:
```bash
   pip install -r Requirements.txt
 ```

### 3️⃣ **Train the Model**
Run the training script to train the model on the Tiny Stories dataset:
```bash
   python src/GPT_Tiny_Stories.py
```

### 4️⃣ **Generate Text**
Use the inference script to generate text:
```bash
   python src/inference.py
```

---

## 📊 **Results**

### - **Training Dataset**: Subset of Tiny Stories
### - **Model Size**: GPT-2 Medium (355M parameters)
### - **Performance**: Achieved coherent and creative text generation after fine-tuning.

---

## 🧠 **Model Architecture**

### **Transformer Block**
### - **Multi-Head Attention**: Focuses on multiple parts of the input.
### - **Feed-Forward Network**: Adds non-linearity and depth.
### - **Residual Connections**: Helps with gradient flow.
### - **Layer Normalization**: Stabilizes training.

### **Multi-Head Attention**
### - **Query, Key, Value**: Linear projections for attention computation.
### - **Attention Masking**: Ensures causal attention for autoregressive tasks.

---

## 📚 **Dataset**

### The **Tiny Stories dataset** is a collection of short, engaging stories. It is split into:
### - **Training Set**: 90% of the data
### - **Validation Set**: 10% of the data

---

## 🤖 **Future Work**

### - 🔍 **Experiment with Larger Models**: Train on GPT-2 Large or GPT-2 XL.
### - 📈 **Hyperparameter Tuning**: Optimize learning rate, batch size, and dropout.
### - 🌐 **Deploy the Model**: Create a web app for real-time text generation.

---

## ❤️ **Acknowledgments**

### This project is inspired by the book **"Build a Large Language Model From Scratch"** by Sebastian Raschka. Special thanks to the creators of the **Tiny Stories dataset**.

---

