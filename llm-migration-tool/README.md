# LLM Migration Tool: TensorFlow to JAX

This project demonstrates how to migrate a Transformer encoder block from **TensorFlow** to **JAX/Flax**, compare their performance, and visualize loss and prediction differences.

The notebook covers:
- Building a BERT-style encoder block in TensorFlow and JAX
- Training both models on synthetic sequence classification data
- Comparing model outputs using loss curves and prediction distributions

## 🔧 Technologies Used

- TensorFlow 2.x
- JAX + Flax + Optax
- NumPy, Matplotlib, Seaborn
- Google Colab (recommended)

## 📈 Results

- Trained both models on 10-token, 2-class synthetic sequences
- Compared training losses across epochs
- Visualized prediction differences using heatmaps and line graphs

## 📁 File Structure

