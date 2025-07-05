
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

### 🤖 Task B: FaceNet Embedding + Classifier
- **Model**: FaceNet for embedding + Custom classifier
- **Approach**:
  - Uses `facenet-pytorch` to extract 512-dimensional facial embeddings.
  - Trains a simple classifier (e.g., MLP or shallow CNN) on these embeddings.
- **Usage**:
  - Detects and aligns faces from the dataset.
  - Extracts embeddings and stores them.
  - Trains classifier on the embeddings for final prediction.
