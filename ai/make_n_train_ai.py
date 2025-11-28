# CUDA версия pytorch для 13.0.x СКАЧИВАТЬ С VPN
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np
import re
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE

import nltk
from nltk.stem.snowball import SnowballStemmer

nltk.download('averaged_perceptron_tagger_rus', 'C:\\nltk_data')
nltk.download('stopwords', 'C:\\nltk_data')


def upper_case_rate(string):
    """Returns percentage of uppercase letters in the string"""
    return np.array(list(map(str.isupper, string))).mean()


def clean_text(string: str) -> str:
    string = string.lower()
    string = re.sub(r"http\S+", "", string)
    string = string.replace('ё', 'е')

    words = re.findall(r'[а-яa-z]+', string)
    stopwords = set(nltk.corpus.stopwords.words('russian'))
    words = [w for w in words if w not in stopwords]

    functionalPos = {'CONJ', 'PRCL'}
    words = [w for w, pos in nltk.pos_tag(words, lang='rus') if pos not in functionalPos]

    stemmer = SnowballStemmer('russian')
    stemmed_words = [stemmer.stem(word) for word in words]

    return ' '.join(stemmed_words)


def clean_text_batch(texts):
    """Очистка батча текстов для многопроцессорной обработки"""
    return [clean_text(text) for text in texts]


def clean_texts_parallel(texts, num_processes=None):
    """Многопроцессорная очистка текстов"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)

    print(f"Using {num_processes} processes for text cleaning...")
    batch_size = max(1, len(texts) // num_processes)
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(
            executor.map(clean_text_batch, batches),
            total=len(batches),
            desc="Cleaning texts"
        ))

    cleaned_texts = []
    for result in results:
        cleaned_texts.extend(result)

    return cleaned_texts


class SimpleNN(nn.Module):
    """Улучшенная нейронная сеть для классификации"""

    def __init__(self, input_size, hidden_size=256, num_classes=1):
        super(SimpleNN, self).__init__()
        # Увеличиваем capacity сети
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # BatchNorm для стабилизации обучения
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


class TextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def plot_training_history(history):
    """Улучшенная визуализация обучения"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # График потерь
    ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График точности
    ax2.plot(history['train_accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График F1-score
    ax3.plot(history['train_f1'], label='Training F1', linewidth=2)
    ax3.plot(history['val_f1'], label='Validation F1', linewidth=2)
    ax3.set_title('Training and Validation F1-Score', fontsize=14)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1-Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # График learning rate
    ax4.plot(history['learning_rate'], label='Learning Rate', linewidth=2, color='red')
    ax4.set_title('Learning Rate Schedule', fontsize=14)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history_detailed.png', dpi=150, bbox_inches='tight')
    plt.show()


def calculate_f1(predictions, labels):
    """Вычисление F1-score"""
    predictions_binary = (predictions > 0.5).float()
    f1 = metrics.f1_score(labels.cpu().numpy(), predictions_binary.cpu().numpy())
    return f1


def train_nn(data: pd.DataFrame):
    start_time = time.time()

    print("Calculating uppercase rates...")
    data['upcase_rate'] = list(map(upper_case_rate, data.comment.values))
    text = np.array(data.comment.values)
    target = data.toxic.astype(int).values

    print("Starting parallel text cleaning...")
    text = clean_texts_parallel(text.tolist())

    X_temp, X_test, y_temp, y_test = train_test_split(
        text, target, test_size=0.2, stratify=target, random_state=0
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=0
    )

    print('Data split:')
    print(f'Train: {len(X_train)} \tTarget rate: {y_train.mean() * 100:.2f}%')
    print(f'Val: {len(X_val)} \tTarget rate: {y_val.mean() * 100:.2f}%')
    print(f'Test: {len(X_test)} \tTarget rate: {y_test.mean() * 100:.2f}%')

    # Векторизация ТОЛЬКО на train данных
    print("Vectorizing texts...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train).toarray()

    # SMOTE ТОЛЬКО на тренировочных данных
    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)

    # Валидация и тест - БЕЗ SMOTE
    X_val_vec = vectorizer.transform(X_val).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    print(f"After SMOTE - Train: {len(X_train_vec)}, Target rate: {y_train.mean() * 100:.2f}%")
    print(f"Feature dimension: {X_train_vec.shape[1]}")

    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    # DataLoader
    train_dataset = TextDataset(torch.FloatTensor(X_train_vec), torch.FloatTensor(y_train))
    val_dataset = TextDataset(torch.FloatTensor(X_val_vec), torch.FloatTensor(y_val))
    test_dataset = TextDataset(torch.FloatTensor(X_test_vec), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # Увеличиваем batch size
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Модель и оптимизатор
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SimpleNN(input_size=X_train_vec.shape[1], hidden_size=512)  # Увеличиваем hidden size
    model.to(device)

    # Взвешенная функция потерь для борьбы с дисбалансом классов
    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)]).to(device)
    criterion = nn.BCELoss(weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # AdamW + регуляризация
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    history = {
        'train_loss': [], 'train_accuracy': [], 'train_f1': [],
        'val_loss': [], 'val_accuracy': [], 'val_f1': [],
        'learning_rate': []
    }

    num_epochs = 20
    best_val_f1 = 0
    patience = 5
    patience_counter = 0

    print("Starting training...")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for features, labels in train_pbar:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            all_train_preds.extend(outputs.detach())
            all_train_labels.extend(labels.detach())

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / train_total:.2f}%'
            })

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
        with torch.no_grad():
            for features, labels in val_pbar:
                features, labels = features.to(device), labels.to(device)

                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_val_preds.extend(outputs)
                all_val_labels.extend(labels)

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * val_correct / val_total:.2f}%'
                })

        # Метрики
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100. * train_correct / train_total
        val_accuracy = 100. * val_correct / val_total

        # Вычисляем F1 для обоих наборов
        train_f1 = calculate_f1(torch.stack(all_train_preds), torch.stack(all_train_labels))
        val_f1 = calculate_f1(torch.stack(all_val_preds), torch.stack(all_val_labels))

        # Обновляем историю
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Early Stopping по F1 на валидации
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Сохраняем лучшую модель
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
            }, 'best_model.pth')
            print(f"New best model saved with Val F1: {val_f1:.4f}")
        else:
            patience_counter += 1

        scheduler.step(avg_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Train F1: {train_f1:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val F1: {val_f1:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        print()

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch'] + 1} with Val F1: {checkpoint['val_f1']:.4f}")

    plot_training_history(history)

    print("\n" + "=" * 60)
    print("FINAL TESTING ON TEST SET")
    print("=" * 60)

    model.eval()
    all_test_predictions = []
    all_test_labels = []
    test_probabilities = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features).squeeze()
            predictions = (outputs > 0.5).float().cpu().numpy()
            all_test_predictions.extend(predictions)
            all_test_labels.extend(labels.numpy())
            test_probabilities.extend(outputs.cpu().numpy())

    total_time = time.time() - start_time
    print(f"Total training and testing time: {total_time:.2f} seconds")

    # Детальная оценка на TEST SET
    print("\nTEST SET Classification Report:")
    print(classification_report(all_test_labels, all_test_predictions))

    test_f1 = metrics.f1_score(all_test_labels, all_test_predictions)
    test_accuracy = metrics.accuracy_score(all_test_labels, all_test_predictions)
    test_precision = metrics.precision_score(all_test_labels, all_test_predictions)
    test_recall = metrics.recall_score(all_test_labels, all_test_predictions)

    print(f"TEST SET Metrics:")
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")

    # Матрица ошибок
    cm = confusion_matrix(all_test_labels, all_test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = metrics.roc_curve(all_test_labels, test_probabilities)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Test Set')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nROC AUC Score: {roc_auc:.4f}")

    torch.save(model.state_dict(), 'final_toxicity_model.pth')
    print("Final model saved as 'final_toxicity_model.pth'")


def test_gpu_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Available CPU cores: {mp.cpu_count()}")


if __name__ == "__main__":
    print("Loading data...")
    df1 = pd.read_csv(r'dataset/labeled_2ch_pikabu.csv')
    df2 = pd.concat([pd.read_json('dataset/okru_part1.jsonl', lines=True),
                     pd.read_json('dataset/okru_part2.jsonl', lines=True)], axis=0, ignore_index=True)
    df2['comment'] = df2['text']
    df2['toxic'] = df2['label'].apply(lambda x: float(x))
    df2.drop('text', axis=1, inplace=True)
    df2.drop('label', axis=1, inplace=True)

    total_df = pd.concat([df1, df2], axis=0, ignore_index=True)

    test_gpu_cuda()
    train_nn(total_df)