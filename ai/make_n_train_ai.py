# CUDA версия pytorch для 13.0.x СКАЧИВАТЬ С VPN
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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
    def __init__(self, input_size, hidden_size=256, num_classes=3):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)  # 3 выходных нейрона

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.softmax = nn.Softmax(dim=1)  # Softmax для многоклассовой классификации

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
        x = self.softmax(x)
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
    """Улучшенная визуализация обучения для многоклассовой классификации"""
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

    # График F1-score (macro average)
    ax3.plot(history['train_f1_macro'], label='Training F1 Macro', linewidth=2)
    ax3.plot(history['val_f1_macro'], label='Validation F1 Macro', linewidth=2)
    ax3.set_title('Training and Validation F1-Score (Macro)', fontsize=14)
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
    plt.savefig('training_history_reviews.png', dpi=150, bbox_inches='tight')
    plt.show()


def calculate_multiclass_metrics(predictions, labels):
    """Вычисление метрик для многоклассовой классификации"""
    predictions_classes = torch.argmax(predictions, dim=1)
    accuracy = metrics.accuracy_score(labels.cpu().numpy(), predictions_classes.cpu().numpy())
    f1_macro = metrics.f1_score(labels.cpu().numpy(), predictions_classes.cpu().numpy(), average='macro')
    f1_weighted = metrics.f1_score(labels.cpu().numpy(), predictions_classes.cpu().numpy(), average='weighted')

    return accuracy, f1_macro, f1_weighted


def train_nn(data: pd.DataFrame):
    start_time = time.time()

    print("Calculating uppercase rates...")
    data['upcase_rate'] = list(map(upper_case_rate, data.text.values))
    text = np.array(data.text.values)
    target = data.label.astype(int).values  # Используем колонку label

    print("Starting parallel text cleaning...")
    text = clean_texts_parallel(text.tolist())

    X_temp, X_test, y_temp, y_test = train_test_split(
        text, target, test_size=0.2, stratify=target, random_state=0
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=0
    )

    print('Data split:')
    print(f'Train: {len(X_train)}')
    print(f'Val: {len(X_val)}')
    print(f'Test: {len(X_test)}')

    # Распределение классов
    print("\nClass distribution:")
    for split_name, y_data in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y_data, return_counts=True)
        print(f"{split_name}: {dict(zip(unique, counts))}")

    # Векторизация ТОЛЬКО на train данных
    print("Vectorizing texts...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train).toarray()

    # SMOTE для многоклассовой классификации
    smote = SMOTE(random_state=42, sampling_strategy='auto')  # auto балансирует все классы
    X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)

    # Валидация и тест - БЕЗ SMOTE
    X_val_vec = vectorizer.transform(X_val).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    print(f"After SMOTE - Train: {len(X_train_vec)}")
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Class distribution after SMOTE: {dict(zip(unique, counts))}")
    print(f"Feature dimension: {X_train_vec.shape[1]}")

    joblib.dump(vectorizer, 'tfidf_vectorizer_reviews.pkl')

    # DataLoader
    train_dataset = TextDataset(torch.FloatTensor(X_train_vec),
                                torch.LongTensor(y_train))  # LongTensor для многоклассовой
    val_dataset = TextDataset(torch.FloatTensor(X_val_vec), torch.LongTensor(y_val))
    test_dataset = TextDataset(torch.FloatTensor(X_test_vec), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Модель и оптимизатор
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SimpleNN(input_size=X_train_vec.shape[1], hidden_size=512, num_classes=3)  # 3 класса
    model.to(device)

    # CrossEntropyLoss для многоклассовой классификации
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    history = {
        'train_loss': [], 'train_accuracy': [], 'train_f1_macro': [], 'train_f1_weighted': [],
        'val_loss': [], 'val_accuracy': [], 'val_f1_macro': [], 'val_f1_weighted': [],
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
        all_train_preds = []
        all_train_labels = []

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for features, labels in train_pbar:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            all_train_preds.append(outputs.detach())
            all_train_labels.append(labels.detach())

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}'
            })

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
        with torch.no_grad():
            for features, labels in val_pbar:
                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                all_val_preds.append(outputs)
                all_val_labels.append(labels)

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}'
                })

        # Метрики
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Вычисляем метрики для train и val
        train_preds = torch.cat(all_train_preds)
        train_labels = torch.cat(all_train_labels)
        train_accuracy, train_f1_macro, train_f1_weighted = calculate_multiclass_metrics(train_preds, train_labels)

        val_preds = torch.cat(all_val_preds)
        val_labels = torch.cat(all_val_labels)
        val_accuracy, val_f1_macro, val_f1_weighted = calculate_multiclass_metrics(val_preds, val_labels)

        # Обновляем историю
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['train_f1_macro'].append(train_f1_macro)
        history['val_f1_macro'].append(val_f1_macro)
        history['train_f1_weighted'].append(train_f1_weighted)
        history['val_f1_weighted'].append(val_f1_weighted)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Early Stopping по F1 macro на валидации
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            patience_counter = 0
            # Сохраняем лучшую модель
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1_macro': val_f1_macro,
            }, 'best_model_reviews.pth')
            print(f"New best model saved with Val F1 Macro: {val_f1_macro:.4f}")
        else:
            patience_counter += 1

        scheduler.step(avg_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(
            f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2%}, Train F1 Macro: {train_f1_macro:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2%}, Val F1 Macro: {val_f1_macro:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        print()

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Загружаем лучшую модель для тестирования
    checkpoint = torch.load('best_model_reviews.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch'] + 1} with Val F1 Macro: {checkpoint['val_f1_macro']:.4f}")

    plot_training_history(history)

    print("\n" + "=" * 60)
    print("FINAL TESTING ON TEST SET")
    print("=" * 60)

    model.eval()
    all_test_predictions = []
    all_test_labels = []
    all_test_probabilities = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            probabilities = outputs.cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            all_test_predictions.extend(predictions)
            all_test_labels.extend(labels.numpy())
            all_test_probabilities.extend(probabilities)

    total_time = time.time() - start_time
    print(f"Total training and testing time: {total_time:.2f} seconds")

    # Детальная оценка на TEST SET
    print("\nTEST SET Classification Report:")
    print(classification_report(all_test_labels, all_test_predictions,
                                target_names=['Positive (0)', 'Neutral (1)', 'Negative (2)']))

    test_accuracy = metrics.accuracy_score(all_test_labels, all_test_predictions)
    test_f1_macro = metrics.f1_score(all_test_labels, all_test_predictions, average='macro')
    test_f1_weighted = metrics.f1_score(all_test_labels, all_test_predictions, average='weighted')

    print(f"TEST SET Metrics:")
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"F1 Macro:  {test_f1_macro:.4f}")
    print(f"F1 Weighted: {test_f1_weighted:.4f}")

    # Матрица ошибок
    cm = confusion_matrix(all_test_labels, all_test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive', 'Neutral', 'Negative'],
                yticklabels=['Positive', 'Neutral', 'Negative'])
    plt.title('Confusion Matrix - Reviews Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_reviews.png', dpi=150, bbox_inches='tight')
    plt.show()

    torch.save(model.state_dict(), 'final_reviews_model.pth')
    print("Final model saved as 'final_reviews_model.pth'")


def test_gpu_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Available CPU cores: {mp.cpu_count()}")


if __name__ == "__main__":
    print("Loading reviews data...")

    # Загрузка данных с отзывами
    # Предполагается, что данные в формате: id,text,src,label
    reviews_data = pd.read_csv('dataset/train.csv')
    news_data = pd.read_csv('dataset/news_train.csv', delimiter='\t')
    women_clothing_data = pd.read_csv('dataset/women-clothing-accessories.3-class.balanced.csv', delimiter='\t')

    total_df = pd.concat([reviews_data, news_data, women_clothing_data], ignore_index=True)
    print(total_df.columns.tolist())
    print(len(total_df))

    # reviews_data = pd.read_csv('your_reviews_dataset.csv', names=['id', 'text', 'src', 'label'])

    print(f"Loaded {len(total_df)} reviews")
    print("Class distribution:")
    print(total_df['label'].value_counts().sort_index())

    test_gpu_cuda()
    train_nn(total_df)