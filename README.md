# Neural Navigator 🧠🧭

Neural Navigator is a multimodal AI system that combines computer vision and natural language understanding to predict navigation paths. Given a 2D map image and a text instruction (e.g., “Go to the Green Square”), the model generates a sequence of (x, y) coordinates representing the navigation trajectory.

This project was developed as part of a Robotics AI Engineer internship technical assignment.

---

## 🚀 Key Features

- Multimodal learning using vision and language
- CNN-based image encoder
- LSTM-based text encoder
- Sequence decoder for trajectory prediction
- End-to-end training pipeline
- Visual inference with predicted vs ground-truth paths

---

## 🧠 Model Architecture

The system processes two inputs:
- A map image
- A natural language navigation command

The image is encoded using a CNN to extract spatial features, while the text instruction is encoded using an embedding layer followed by an LSTM. These representations are fused and passed to an LSTM-based decoder that predicts a fixed-length sequence of (x, y) coordinates representing the navigation path.

---

## 📁 Project Structure

neural-navigator/
├── src/
│ ├── dataset.py # Dataset loading and preprocessing
│ ├── model.py # Multimodal neural network
│ ├── train.py # Training pipeline
│ └── infer.py # Inference and visualization
├── sample_output/
│ └── result.png # Sample inference result
├── .gitignore
└── README.md


Datasets, model checkpoints, and generated outputs are excluded using `.gitignore` to keep the repository clean and reproducible.

---

## ⚙️ How to Run

### Training
```bash
python src/train.py
python src/infer.py

⚠️ Challenges & Solutions
One major challenge was handling inconsistent annotation formats in the dataset, where image filenames and text commands were stored using different keys. This was solved by implementing a robust dataset loader that dynamically adapts to varying annotation schemas.
Another challenge involved Python module resolution issues in notebook and script-based execution environments. This was addressed by restructuring the project using a clean src/ layout and consistent import patterns.
Predicting full navigation trajectories instead of single target points was also non-trivial. An LSTM-based sequence decoder with fixed-length padding and mean squared error loss was used to ensure stable and consistent training.

📈 Results
The model successfully learns to predict navigation paths from combined visual and textual inputs. Training loss decreases consistently, and inference visualizations show reasonable alignment between predicted and ground-truth trajectories.

🛠️ Tech Stack
Python
PyTorch
Torchvision
NumPy
Matplotlib

✅ Conclusion
This project demonstrates a clean, modular, and reproducible approach to multimodal AI for navigation tasks. Emphasis was placed on robustness, clarity, and real-world engineering practices rather than overfitting for benchmark scores.

---

## 🔥 FINAL STATUS (CONFIRMATION)

After this:
- ✅ README complete
- ✅ Image visible on GitHub
- ✅ Assignment-ready
- ✅ Interview-safe explanations

💪 **Nee romba clean-aa finish pannita da**.

### Ippo last step venuma?
If yes, I’ll give **perfect submission mail / WhatsApp message**.

Reply 👇  
