import { useState } from 'react';

export default function UploadForm({ onResult }) {
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

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });

      if (response.ok) {
        const data = await response.json();
        setResult(data);
        onResult?.(data);
      } else {
        const error = await response.text();
        alert(`Ошибка: ${error}`);
      }
    } catch (err) {
      alert('Не удалось подключиться к серверу');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-section">
      <form onSubmit={handleSubmit} className="upload-form">
        <div className="form-group">
          <label>Один отзыв</label>
          <textarea name="review" placeholder="Введите отзыв..." rows="4" />
        </div>

        <div className="form-group">
          <label>Или загрузите файл</label>
          <input type="file" name="File" accept=".txt" />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Анализ...' : 'Отправить'}
        </button>
      </form>

      {/* Результат */}
      {result && <AnalysisResult result={result} />}
    </div>
  );
}
