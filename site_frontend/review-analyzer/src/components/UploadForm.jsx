import { useState } from 'react';
import TextAnalysisResult from './TextAnalysisResult';
import FileAnalysisResult from './FileAnalysisResult';

const API_URL = 'http://localhost:5039/api';

export default function UploadForm({ onResult }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const review = formData.get('review');
    const file = formData.get('csvFile');

    // Проверяем что есть либо текст, либо файл
    const hasText = review && review.trim().length > 0;
    const hasFile = file && file.size > 0;

    if (!hasText && !hasFile) {
      alert('Введите отзыв или загрузите файл');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      let endpoint = '/Analysis/analyze';
      
      // Если есть файл, используем эндпоинт для файлов
      if (hasFile && !hasText) {
        endpoint = '/Analysis/analyze-file';
      }
      // Если есть и текст и файл, приоритет у текста
      
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });

      if (response.ok) {
        const data = await response.json();
        setResult({
          ...data,
          type: hasFile && !hasText ? 'file' : 'text' // определяем тип результата
        });
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

  // Закрытие модального окна
  const closeModal = () => setResult(null);

  return (
    <div className="upload-section">
      <form onSubmit={handleSubmit} className="upload-form">
        <div className="form-group">
          <label>Один отзыв</label>
          <textarea 
            name="review" 
            placeholder="Введите отзыв для анализа тональности..." 
            rows="4" 
          />
        </div>

        <div className="form-group">
          <label>Или загрузите файл</label>
          <input type="file" name="csvFile" accept=".csv,.txt" />
          <small className="file-hint">Поддерживаются CSV и TXT файлы</small>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? '🔄 Анализ...' : '📊 Проанализировать'}
        </button>
      </form>

      {/* Модальное окно с результатом для текста */}
      {result && result.type === 'text' && (
        <div onClick={closeModal}>
          <TextAnalysisResult result={result} />
        </div>
      )}

      {/* Временный вывод для файлов */}
      {result && result.type === 'file' && (
        <div onClick={closeModal}>
        <FileAnalysisResult result={result} />
        </div>
      )}
    </div>
  );
}