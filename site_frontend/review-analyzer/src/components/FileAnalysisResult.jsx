export default function FileAnalysisResult({ result }) {
  if (!result) return null;

  console.log('FileAnalysisResult data:', result); // для отладки

  return (
    <div className="file-analysis-modal">
      <div className="modal-content">
        <div className="modal-header">
          <h2>📊 Результат анализа файла</h2>
        </div>
        
        <div className="modal-body">
          <div className="file-info">
            {/* УБРАТЬ fileName - его нет в ответе */}
            <p><strong>📈 Всего записей:</strong> {result.totalRecords || 0}</p>
            <p><strong>🟢 Позитивных:</strong> {result.positiveCount || 0}</p>
            <p><strong>🔴 Негативных:</strong> {result.negativeCount || 0}</p>
            <p><strong>📅 Дата анализа:</strong> {result.analysisDate ? new Date(result.analysisDate).toLocaleString() : 'Не указана'}</p>
          </div>

          {/* УБРАТЬ таблицу с records - этого поля нет в ответе */}
          <div className="statistics">
            <h4>Статистика анализа</h4>
            <p>Файл успешно обработан. Получена общая статистика по тональности комментариев.</p>
            
            {result.totalRecords > 0 && (
              <div className="percentages">
                <p><strong>🟢 Позитивные:</strong> {((result.positiveCount / result.totalRecords) * 100).toFixed(1)}%</p>
                <p><strong>🔴 Негативные:</strong> {((result.negativeCount / result.totalRecords) * 100).toFixed(1)}%</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}