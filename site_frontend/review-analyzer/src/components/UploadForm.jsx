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
    
    if (hasFile && !hasText) {
      endpoint = '/Analysis/analyze-file';
    }
    
    const response = await fetch(`${API_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      
 
      const normalizedData = {
        comment: data.comment,
        classLabel: data.class_label, 
        probability: data.probability,
        createdDate: data.created_date,
        type: hasFile && !hasText ? 'file' : 'text'
      };
      
      setResult(normalizedData);
      onResult?.(normalizedData);
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