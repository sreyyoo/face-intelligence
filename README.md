
### 🧠 Task A: EfficientNet-Based Classification
- **Model**: Pretrained EfficientNet from `efficientnet_pytorch`
- **Approach**:
  - Uses full-face images.
  - Applies data augmentation techniques.
  - Fine-tunes EfficientNet for binary classification (Male/Female).
- **Usage**:
  - Mount Google Drive to access dataset.
  - Train the model using the training loop provided.
  - Evaluate performance using accuracy and other metrics.
 
### Data Structure

```
Task_A/
├── train/
│   ├── male/
│   └── female/
└── val/
    ├── male/
    └── female/
```

Accuracy: 85.71%
Precision: 84.5%
Recall: 86.0%
F1 Score: 85.2%

---

### 🤖 Task B: FaceNet Embedding + Cosine Similarity

- **Model**: `InceptionResnetV1` from `facenet-pytorch`
  - Pretrained on VGGFace2 dataset.
  - Extracts 512-dimensional embeddings for each face.
- **Pipeline**:
  1. **Face Detection & Alignment**: Uses `MTCNN` to detect and crop faces.
  2. **Embedding Extraction**:
     ```python
     model = InceptionResnetV1(pretrained='vggface2').eval()
     embedding = model(face_tensor.unsqueeze(0))
     ```
  3. **Similarity Matching**:
     - Embeddings are compared using **cosine similarity**.
     - For each test image, the most similar known embedding is selected.
     - Class label is assigned based on nearest match.
- **Evaluation Metrics**:
  - Accuracy
  - F1 Score
  - Cosine similarity values printed/logged to understand classification confidence.
- **No Explicit Classifier**: The model does not train a neural network classifier. Instead, it uses **metric learning** via cosine similarity for classification.

---

### Data Structure

```
Task_B/
├── person1/
│   ├── img.jpg
│   └── distortion/
│       ├── aug1.jpg
│       └── aug2.jpg
├── person2/
│   ├── img.jpg
│   └── distortion/
        ...
```

## 📊 Evaluation Example (Task B)

```text
Accuracy: ~77%
F1 Score: ~76%

