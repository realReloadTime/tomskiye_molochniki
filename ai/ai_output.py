import joblib
import torch
import pandas as pd
import numpy as np
import multiprocessing as mp

from fontTools.misc.bezierTools import printSegments
from joblib.externals.loky import ProcessPoolExecutor
import os

from make_n_train_ai import SimpleNN, clean_text

current_dir = os.path.dirname(os.path.abspath(__file__))


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
    """Предсказание для батча текстов - многоклассовая версия"""
    # Векторизация
    text_vectors = vectorizer.transform(texts).toarray()

    # Предсказание батчами
    all_predictions = []
    all_probabilities = []

    for i in range(0, len(text_vectors), batch_size):
        batch_vectors = text_vectors[i:i + batch_size]
        batch_tensor = torch.FloatTensor(batch_vectors).to(device)

        with torch.no_grad():
            batch_outputs = model(batch_tensor).cpu().numpy()

        # Получаем предсказанные классы (argmax) и вероятности
        batch_preds = np.argmax(batch_outputs, axis=1)
        batch_probs = np.max(batch_outputs, axis=1)

        all_predictions.extend(batch_preds)
        all_probabilities.extend(batch_probs)

    return all_predictions, all_probabilities


def load_reviews_model(input_size=5000):
    """Загружает модель и векторизатор для предсказания тональности отзывов
    :arg input_size: должен совпадать с размерностью векторизатора"""

    # Загружаем векторизатор
    vectorizer = joblib.load(current_dir + '\\' + 'tfidf_vectorizer_reviews.pkl')

    # Загружаем архитектуру модели (3 класса)
    model = SimpleNN(input_size=input_size, hidden_size=512, num_classes=3)

    # Загружаем веса модели
    model.load_state_dict(torch.load(current_dir + '\\' + 'final_reviews_model.pth', map_location='cpu'))
    model.eval()  # переводим модель в режим оценки

    return model, vectorizer


def predict_sentiment_with_probability(text: str, model, vectorizer, device='cpu') -> tuple[str, int, float, list]:
    """
    Предсказывает тональность отзыва с вероятностью

    Returns:
        tuple: (текст, класс, вероятность, все_вероятности)
        Классы: 0 - положительный, 1 - нейтральный, 2 - отрицательный
    """
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text]).toarray()
    text_tensor = torch.FloatTensor(text_vector).to(device)

    with torch.no_grad():
        probabilities = model(text_tensor).squeeze().cpu().numpy()

    # Если probabilities - скаляр (маловероятно для 3 классов), преобразуем в массив
    if probabilities.ndim == 0:
        probabilities = np.array([probabilities])

    # Для многоклассовой классификации берем argmax
    class_label = np.argmax(probabilities)
    confidence = np.max(probabilities)

    return text, class_label, confidence, probabilities.tolist()


def get_sentiment_label(class_id: int) -> str:
    """Возвращает текстовое описание тональности"""
    sentiment_map = {
        0: "положительный",
        1: "нейтральный",
        2: "отрицательный"
    }
    return sentiment_map.get(class_id, "неизвестно")


def process_sentiment_csv(df: pd.DataFrame, model, vectorizer, device='cpu') -> pd.DataFrame:
    """
    Обрабатывает CSV-файл с отзывами и возвращает результат с классификацией

    Args:
        df (pandas DataFrame): DataFrame отзывов с колонкой 'text' и 'ID'

    Returns:
        pd.DataFrame: Датафрейм с колонками ['text', 'class_label', 'confidence', 'sentiment']
    """

    if 'text' not in df.columns or 'ID' not in df.columns:
        raise ValueError("CSV file must contain 'text' and 'ID' column")

    texts = df['text'].fillna('').astype(str).tolist()

    cleaned_texts = clean_texts_fast(texts)

    predictions, confidences = predict_batch(
        cleaned_texts, model, vectorizer, device
    )

    sentiments = [get_sentiment_label(pred) for pred in predictions]
    labels = []
    for sentiment in sentiments:
        if sentiment == 'нейтральный':
            labels.append(0)
        elif sentiment == 'положительный':
            labels.append(1)
        else:
           labels.append(2)

    result_df = pd.DataFrame({
        'ID': df['ID'],
        'label': labels
    })

    return result_df


if __name__ == "__main__":
    # Инициализация (делается один раз)
    print("Loading model...")
    model, vectorizer = load_reviews_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print("Model loaded successfully!")

    # Примеры предсказаний
    test_texts = [
        "Отличный товар! Очень доволен покупкой, качество на высоте.",
        "Товар немного не соответствует описанию, ткань не такая плотная как на фотографиях",
        "Платье сшито хорошо, но по размеру не подошло, по длине тоже..",
        "Заказ я не получил совсем, так ещё и возврат денег не сделали.",
        "Обычный товар, ничего особенного. Соответствует описанию.",
        "Прекрасный сервис! Быстрая доставка, вежливый персонал.",
        "Ужасное качество! Товар сломался через день использования.",
        "Нормально, но есть небольшие недочеты в работе."
    ]

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ КЛАССИФИКАЦИИ ОТЗЫВОВ")
    print("=" * 80)

    for text in test_texts:
        original_text, class_label, confidence, all_probs = predict_sentiment_with_probability(
            text, model, vectorizer, device
        )

        sentiment_label = get_sentiment_label(class_label)

        print(f"\nТекст: {text}")
        print(f"Тональность: {sentiment_label} (класс: {class_label})")
        print(f"Уверенность: {confidence:.4f}")
        print(
            f"Все вероятности: [положительный: {all_probs[0]:.4f}, нейтральный: {all_probs[1]:.4f}, отрицательный: {all_probs[2]:.4f}]")

    # Пример обработки CSV файла
    print("\n" + "=" * 80)
    print("ПРИМЕР ОБРАБОТКИ CSV ФАЙЛА")
    print("=" * 80)

    # Создаем тестовый DataFrame
    sample_data = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'text': [
            "Очень понравилось, рекомендую!",
            "Нормальный товар, но дорогой",
            "Ужасное качество, не советую покупать",
            "Отлично, все супер!",
            "Ничего особенного, обычная вещь"
        ]
    })

    print("Входные данные:")
    print(sample_data)

    # Обрабатываем
    result = process_sentiment_csv(sample_data, model, vectorizer, device)
    print("\nРезультаты классификации:")
    print(result)