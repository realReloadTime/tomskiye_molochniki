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
    const review = formData.get('review')?.trim();
    const file = formData.get('csvFile');

    const hasText = review && review.length > 0;
    const hasFile = file && file.size > 0;

    if (!hasText && !hasFile) {
      alert('Введите отзыв или загрузите файл');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      if (hasText && !hasFile) {
        const response = await fetch(`${API_URL}/Analysis/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ comment: review }),
        });

        if (response.ok) {
          const data = await response.json();
          const normalizedData = {
            comment: data.comment || review,
            classLabel: data.classLabel || data.class_label,
            probability: data.probability,
            createdDate: data.createdDate || data.created_date,
            type: 'text'
          };
          setResult(normalizedData);
          onResult?.(normalizedData);
        } else {
          const error = await response.text();
          alert(`Ошибка анализа текста: ${error}`);
        }

      } else if (hasFile && !hasText) {
        const uploadData = new FormData();
        uploadData.append('csvFile', file);

        const response = await fetch(`${API_URL}/Analysis/analyze-file`, {
          method: 'POST',
          body: uploadData,
        });

        if (response.ok) {
          const data = await response.json();
          const normalizedData = {
            totalRecords: data.totalRecords,
            positiveCount: data.positiveCount,
            negativeCount: data.negativeCount,
            analysisDate: data.analysisDate,
            type: 'file'
          };
          setResult(normalizedData);
          onResult?.(normalizedData);
        } else {
          const error = await response.text();
          alert(`Ошибка анализа файла: ${error}`);
        }

      } else {
        alert('Нельзя одновременно отправить текст и файл');
      }
    } catch (err) {
      alert('Не удалось подключиться к серверу');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

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
          <input type="file" name="csvFile" accept=".csv" />
          <small className="file-hint">Поддерживаются только CSV файлы</small>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? '🔄 Анализ...' : '📊 Проанализировать'}
        </button>
      </form>

      {/* Модалки */}
      {result && result.type === 'text' && (
        <div className="modal-overlay" onClick={closeModal}>
          <TextAnalysisResult result={result} />
        </div>
      )}

      {result && result.type === 'file' && (
        <div className="modal-overlay" onClick={closeModal}>
          <FileAnalysisResult result={result} />
        </div>
      )}
    </div>
  );
}