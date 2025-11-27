import joblib
import torch
from ai.make_n_train_ai import SimpleNN, clean_text


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