import { useState } from 'react';
import './Home.css'; // вынесем стили отдельно для ясности

export default function Home() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);

    if (!formData.get('review') && !formData.get('csvFile')) {
      alert('Введите отзыв или загрузите файл');
      return;
    }

    setLoading(true);
    setResult(null);

    // Здесь будет реальный fetch...
    // Для демо — имитируем задержку и ответ
    setTimeout(() => {
      setResult({
        total: 7,
        positive: 3,
        neutral: 2,
        negative: 2
      });
      setLoading(false);
    }, 1500);
  };

  // Функция для демо — показать результат без отправки
  const handleDemo = () => {
    setResult({
      total: 8,
      positive: 5,
      neutral: 1,
      negative: 2
    });
  };

  return (
    <div className="page home">
      <h1>Анализ отзывов</h1>
      <p>Загрузите отзыв или CSV — сервер проанализирует тональность</p>

      <form onSubmit={handleSubmit} className="upload-form">
        <div className="form-group">
          <label>Один отзыв</label>
          <textarea
            name="review"
            placeholder="Введите отзыв..."
            rows="4"
          ></textarea>
        </div>

        <div className="form-group">
          <label>Или загрузите CSV-файл</label>
          <input type="file" name="csvFile" accept=".csv" />
        </div>

        <button type="submit" disabled={loading} className="submit-btn">
          {loading ? 'Отправляется...' : 'Отправить на анализ'}
        </button>
      </form>

      {/* Кнопка для просмотра примера */}
      <div className="demo-section">
        <button type="button" className="demo-btn" onClick={handleDemo}>
          результат
        </button>
      </div>

      {/* Отображение результата */}
      {result && (
        <div className="result-card">
          <h2>Результат анализа</h2>

          <div className="stats-grid">
            <div className="stat positive">
              <h3>Положительные</h3>
              <p>{result.positive}</p>
              <div className="bar">
                <span style={{ width: `${(result.positive / result.total) * 100}%` }}></span>
              </div>
            </div>

            <div className="stat neutral">
              <h3>Нейтральные</h3>
              <p>{result.neutral}</p>
              <div className="bar">
                <span style={{ width: `${(result.neutral / result.total) * 100}%` }}></span>
              </div>
            </div>

            <div className="stat negative">
              <h3>Отрицательные</h3>
              <p>{result.negative}</p>
              <div className="bar">
                <span style={{ width: `${(result.negative / result.total) * 100}%` }}></span>
              </div>
            </div>
          </div>

          <p className="total">
            <strong>Всего отзывов:</strong> {result.total}
          </p>
        </div>
      )}
    </div>
  );
}
