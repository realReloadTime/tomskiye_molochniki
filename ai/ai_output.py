import joblib
import torch
import pandas as pd
import numpy as np
import multiprocessing as mp
from joblib.externals.loky import ProcessPoolExecutor

from ai.make_n_train_ai import SimpleNN, clean_text


def process_text_batch(texts_batch):
    """Обработка батча текстов для многопроцессорности"""
    return [clean_text(text) for text in texts_batch]


def clean_texts_fast(texts, num_processes=None):
    """Быстрая параллельная очистка текстов без tqdm"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)

    batch_size = max(1, len(texts) // num_processes)
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(process_text_batch, batches))

    cleaned_texts = []
    for result in results:
        cleaned_texts.extend(result)

    return cleaned_texts


def predict_batch(texts, model, vectorizer, device, batch_size=1000):
    """Предсказание для батча текстов"""
    # Векторизация
    text_vectors = vectorizer.transform(texts).toarray()

    # Предсказание батчами
    all_predictions = []
    all_probabilities = []

    for i in range(0, len(text_vectors), batch_size):
        batch_vectors = text_vectors[i:i + batch_size]
        batch_tensor = torch.FloatTensor(batch_vectors).to(device)

        with torch.no_grad():
            batch_probs = model(batch_tensor).squeeze().cpu().numpy()

        # Обработка случая, когда batch_probs - скаляр
        if batch_probs.ndim == 0:
            batch_probs = np.array([batch_probs])

        batch_preds = (batch_probs > 0.5).astype(int)

        all_predictions.extend(batch_preds)
        all_probabilities.extend(batch_probs)

    return all_predictions, all_probabilities


def load_toxicity_model(input_size=5000):
    """Загружает модель и векторизатор для предсказания
    :arg input_size: должен совпадать с размерностью векторизатора"""

    # Загружаем векторизатор
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Загружаем архитектуру модели
    model = SimpleNN(input_size=input_size, hidden_size=512)

    # Загружаем веса модели
    model.load_state_dict(torch.load('final_toxicity_model.pth', map_location='cpu'))
    model.eval()  # переводим модель в режим оценки

    return model, vectorizer


def predict_toxicity_with_probability(text: str, model, vectorizer, device='cpu') -> tuple:
    """
    Предсказывает токсичность с вероятностью

    Returns:
        tuple: (класс, вероятность_токсичности)
    """
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text]).toarray()
    text_tensor = torch.FloatTensor(text_vector).to(device)

    with torch.no_grad():
        probability = model(text_tensor).squeeze().item()

    class_label = 1 if probability > 0.5 else 0
    return class_label, probability


def process_toxicity_csv(csv_bytes: bytes, model, vectorizer, device='cpu') -> pd.DataFrame:
    """
    Обрабатывает CSV-файл с комментариями и возвращает результат с классификацией

    Args:
        csv_bytes (bytes): Байтовый поток CSV-файла с колонкой 'comment'

    Returns:
        pd.DataFrame: Датафрейм с колонками ['comment', 'class', 'prob']
    """

    # Чтение CSV
    df = pd.read_csv(csv_bytes)

    if 'comment' not in df.columns:
        raise ValueError("CSV file must contain 'comment' column")

    texts = df['comment'].fillna('').astype(str).tolist()

    # Быстрая очистка текстов
    print("Cleaning texts...")
    cleaned_texts = clean_texts_fast(texts)

    # Предсказание
    print("Predicting toxicity...")
    predictions, probabilities = predict_batch(
        cleaned_texts, model, vectorizer, device
    )

    result_df = pd.DataFrame({
        'comment': df['comment'],
        'class': predictions,
        'prob': probabilities
    })

    return result_df


if __name__ == "__main__":
    # Инициализация (делается один раз)
    print("Loading model...")
    model, vectorizer = load_toxicity_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print("Model loaded successfully!")

    # Примеры предсказаний
    test_texts = [
        "Это нормальный комментарий без оскорблений",
        "Ты полный идиот и дебил!",
        "Спасибо за полезную информацию",
        "Иди нафиг, тупой мудак!",
        "НУ ДА КОНЕЧНО НУ ДА"
    ]

    for text in test_texts:
        prediction = predict_toxicity_with_probability(text, model, vectorizer, device)
        class_label, prob = predict_toxicity_with_probability(text, model, vectorizer, device)

        print(f"\nТекст: {text}")
        print(f"Класс: {prediction[0]} ({'токсичный' if prediction[0] == 1 else 'нетоксичный'})")
        print(f"Вероятность токсичности: {prob:.4f}")


    # ------------------------- Пример для большого байтового потока CSV-файла с одним единственным столбцом "comment"
    # csv_bytes = sample_data.encode('utf-8')
    #
    # # Обрабатываем
    # result = process_toxicity_csv(csv_bytes, model, vectorizer)
    # print("\nРезультаты:")
    # print(result.head(10))