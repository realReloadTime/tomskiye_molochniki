# CUDA версия pytorch для 13.0.x СКАЧИВАТЬ С VPN
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
import numpy as np
import re
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os


import nltk
from nltk.stem.snowball import SnowballStemmer

'''
If you did not install the data to one of the above central locations, you will need to set the NLTK_DATA 
environment variable to specify the location of the data. (On a Windows machine, right click on 
"My Computer" then select Properties > Advanced > Environment Variables > User Variables > New...)
'''

nltk.download('averaged_perceptron_tagger_rus', 'C:\\nltk_data')
nltk.download('stopwords', 'C:\\nltk_data')


def upper_case_rate(string):
    """Returns percentage of uppercase letters in the string"""
    return np.array(list(map(str.isupper, string))).mean()


def clean_text(string: str) -> str:
    string = string.lower()
    string = re.sub(r"http\S+", "", string)  # deletion urls
    string = string.replace('ё', 'е')

    # cyrillic + latin
    words = re.findall(r'[а-яa-z]+', string)

    # deletion "и", "а", "на", "в", etc.
    stopwords = set(nltk.corpus.stopwords.words('russian'))
    words = [w for w in words if w not in stopwords]

    functionalPos = {'CONJ', 'PRCL'}
    words = [w for w, pos in nltk.pos_tag(words, lang='rus') if pos not in functionalPos]

    # "Я ходил в магазин и купил молоко, а потом бегал в парке! http://example.com" -> "ход магазин куп молок бега парк"
    stemmer = SnowballStemmer('russian')
    stemmed_words = [stemmer.stem(word) for word in words]

    return ' '.join(stemmed_words)


def clean_text_batch(texts):
    """Очистка батча текстов для многопроцессорной обработки"""
    return [clean_text(text) for text in texts]


def clean_texts_parallel(texts, num_processes=None):
    """Многопроцессорная очистка текстов"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)  # Ограничиваем максимум 8 процессами

    print(f"Using {num_processes} processes for text cleaning...")

    # Разбиваем тексты на батчи
    batch_size = max(1, len(texts) // num_processes)
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(
            executor.map(clean_text_batch, batches),
            total=len(batches),
            desc="Cleaning texts"
        ))

    # Объединяем результаты
    cleaned_texts = []
    for result in results:
        cleaned_texts.extend(result)

    return cleaned_texts


class SimpleNN(nn.Module):
    """Простая нейронная сеть для классификации"""

    def __init__(self, input_size, hidden_size=128, num_classes=1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class TextDataset(Dataset):
    """Dataset для текстовых данных"""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def plot_training_history(history):
    """Построение графиков обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # График потерь
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # График точности
    ax2.plot(history['train_accuracy'], label='Training Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def train_nn(data: pd.DataFrame):
    """Обучение нейронной сети с многопроцессорной предобработкой"""
    start_time = time.time()

    # Добавляем признак uppercase rate
    print("Calculating uppercase rates...")
    data['upcase_rate'] = list(map(upper_case_rate, data.comment.values))
    text = np.array(data.comment.values)
    target = data.toxic.astype(int).values

    # Многопроцессорная очистка текстов
    print("Starting parallel text cleaning...")
    text = clean_texts_parallel(text.tolist())

    preprocessing_time = time.time() - start_time
    print(f"Text preprocessing completed in {preprocessing_time:.2f} seconds")

    X_train, X_test, y_train, y_test = train_test_split(text, target, test_size=.3, stratify=target, shuffle=True,
                                                        random_state=0)
    print('Dim of train:', len(X_train), '\tTarget rate: {:.2f}%'.format(y_train.mean() * 100))
    print("Dim of test:", len(X_test), '\tTarget rate: {:.2f}%'.format(y_test.mean() * 100))

    # Векторизация текста
    print("Vectorizing texts...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    print(f"Feature dimension: {X_train_vec.shape[1]}")

    # Создание Dataset и DataLoader
    train_dataset = TextDataset(torch.FloatTensor(X_train_vec), torch.FloatTensor(y_train))
    test_dataset = TextDataset(torch.FloatTensor(X_test_vec), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Инициализация модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SimpleNN(input_size=X_train_vec.shape[1], hidden_size=256)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    num_epochs = 10
    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch_idx, (features, labels) in enumerate(train_pbar):
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

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / train_total:.2f}%'
            })

        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
        with torch.no_grad():
            for features, labels in val_pbar:
                features, labels = features.to(device), labels.to(device)

                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * val_correct / val_total:.2f}%'
                })

        # Сохранение метрик
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        train_accuracy = 100. * train_correct / train_total
        val_accuracy = 100. * val_correct / val_total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print()

    plot_training_history(history)

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features).squeeze()
            predictions = (outputs > 0.5).float().cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_predictions))
    f1_final = metrics.f1_score(all_labels, all_predictions)
    print(f"Final F1 Score: {f1_final:.4f}")


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