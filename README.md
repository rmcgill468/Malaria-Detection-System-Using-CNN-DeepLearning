# 🦟 Malaria Detection System Using CNN Deep Learning

### Classification of Malaria Infection in Blood Smear Images Using CNN Deep Learning Models: A Computer Vision Approach

---

## 📌 About the Project

Malaria remains one of the most life-threatening parasitic diseases globally, particularly in developing regions where diagnostic resources are limited. Traditional microscopic examination is accurate but slow, labour-intensive, and prone to human error due to technician fatigue and varying expertise.

This project develops a **custom Convolutional Neural Network (CNN)** to automatically classify **malaria-infected (Parasitized)** vs **healthy (Uninfected)** blood smear images — providing a fast, reliable, and scalable alternative to manual microscopy. The trained model is deployed as an **interactive Streamlit web application** that gives instant predictions with confidence scores.

The Project is live and deployed on: https://huggingface.co/spaces/rmcgill468/malaria-detection-cnn 

## 🎯 Key Highlights

- ✅ **96.77% Validation Accuracy** on 27,558 NIH Malaria images
- ✅ **94.55% Training Accuracy**
- ✅ Classifies a blood smear image in **milliseconds**
- ✅ **No GPU required** — trained on a standard Windows 10 CPU
- ✅ Deployed as a **Streamlit Web App** for real-time predictions

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.x | Primary programming language |
| TensorFlow 2.x / Keras | CNN model building and training |
| OpenCV | Image reading, resizing, preprocessing |
| NumPy | Numerical operations and array handling |
| Matplotlib | Plotting training/validation accuracy graphs |
| Pillow | Image handling in the Streamlit app |
| Streamlit | Interactive web application deployment |
| Anaconda | Python environment and package manager |
| VS Code | Code editor |

---

## 📊 Dataset

- **Name:** NIH Malaria Cell Image Dataset
- **Source:** Kaggle (publicly available)
- **Total Images:** 27,558 labeled microscopic blood smear images
- **Classes:**
  - 🔴 **Parasitized** — 13,780 images (infected red blood cells)
  - 🟢 **Uninfected** — 13,778 images (healthy red blood cells)
- **Original Resolution:** 148×148 px | **Resized to:** 64×64 px
- **Train/Validation Split:** 80% training (20,813) / 20% validation (5,202)

---

## 🧠 CNN Model Architecture

```
Input (64×64×3)
    ↓
Conv2D (32 filters, ReLU) → BatchNorm → MaxPooling → Dropout(0.25)
    ↓
Conv2D (64 filters, ReLU) → BatchNorm → MaxPooling → Dropout(0.25)
    ↓
Conv2D (128 filters, ReLU) → BatchNorm → MaxPooling → Dropout(0.25)
    ↓
Flatten
    ↓
Dense (256 neurons, ReLU) → Dropout(0.5)
    ↓
Output (Sigmoid — Binary Classification)
```

**Training Configuration:**
- Optimizer: Adam (learning rate = 0.0001)
- Loss Function: Binary Cross-Entropy
- Epochs: 20 with Early Stopping
- Batch Size: 32
- Class weights applied for balanced training

---

## 🔄 Data Augmentation

To improve generalization and prevent overfitting:
- Rotation: ±15°
- Horizontal Flipping: Enabled
- Width & Height Shifting: ±10%
- Zoom Range: 10%

---

## ⚙️ System Workflow

1. **User uploads** a blood smear image via the Streamlit web app
2. **Image is preprocessed** — resized to 64×64, normalized to 0–1, reshaped for CNN input
3. **CNN extracts features** — parasite shape, color intensity, cell morphology
4. **Classification output** — probability score between 0 and 1
5. **Result displayed** with color-coded box:
   - 🔴 **RED box** = Parasitized (Infected)
   - 🟢 **GREEN box** = Uninfected (Healthy)
   - Confidence score shown as a percentage

---

## 📈 Results

| Metric | Score |
|---|---|
| Training Accuracy | 94.55% |
| Validation Accuracy | 96.77% |
| Loss Function | Binary Cross-Entropy |
| Optimizer | Adam |

The model demonstrates strong classification performance and generalization capability comparable to expert-level diagnosis.

---

## 🚀 How to Run

1. **Clone the repository**
```bash
   git clone https://github.com/YOUR-USERNAME/malaria-detection-cnn.git
   cd malaria-detection-cnn
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Run the Streamlit web app**
```bash
   streamlit run app.py
```

4. **Upload** a blood smear image and get an instant prediction!

---

## 💻 Hardware Requirements

- OS: Windows 10
- RAM: 8 GB minimum
- Processor: CPU-based (No GPU required)
- Environment: Anaconda Python Distribution

---

## 👤 Author

**Russell McGill**
- 🎓 M.Tech Scholar — Computer Engineering
- 🏫 RK University, Rajkot, Gujarat — April 2026
- 📋 Enrollment No: 24SOECE21601
- 👨‍🏫 Under the Guidance of: Dr. Chetan Shingadiya (HOD, Computer Engineering)

---

## 💼 Why This Project Matters

- 🌍 Supports healthcare in **low-resource and rural regions**
- ⚡ Provides **instant diagnosis** — milliseconds vs hours for manual microscopy
- 🤖 Demonstrates how **AI can augment medical professionals**
- 📱 Potential for integration into **mobile diagnostic apps**
- 🏥 Contributes to **WHO global malaria eradication goals**

---

## 📄 License

This project is open source and available for academic and research use.
