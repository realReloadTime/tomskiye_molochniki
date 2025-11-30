export default function FileAnalysisResult({ result }) {
  if (!result) return null;

  const total = result.totalRecords;
  const positive = result.positiveCount;
  const neutral = result.neutralCount;
  const negative = result.negativeCount;

  const posPercent = total ? ((positive / total) * 100).toFixed(1) : 0;
  const neuPercent = total ? ((neutral / total) * 100).toFixed(1) : 0;
  const negPercent = total ? ((negative / total) * 100).toFixed(1) : 0;

  return (
    <div className="file-analysis-modal">
      <div className="modal-content">
        <div className="modal-header">
          <h2>📊 Результат анализа файла</h2>
        </div>

        <div className="modal-body">
          <div className="file-info">
            <p><strong>📈 Всего записей:</strong> {total}</p>
            <p><strong>🟢 Положительных:</strong> {positive}</p>
            <p><strong>🟡 Нейтральных:</strong> {neutral}</p>
            <p><strong>🔴 Отрицательных:</strong> {negative}</p>
            <p><strong>📅 Дата анализа:</strong> {result.analysisDate ? new Date(result.analysisDate).toLocaleString() : 'Не указана'}</p>
          </div>

          <div className="statistics">
            <h4>Статистика анализа</h4>
            <p>Распределение тональности комментариев:</p>

            {total > 0 && (
              <div className="percentages">
                <div className="bar-group">
                  <div className="bar-label">
                    <span>🟢 Положительные</span>
                    <span>{posPercent}%</span>
                  </div>
                  <div className="bar-bg">
                    <div className="bar-fill positive" style={{ width: `${posPercent}%` }}></div>
                  </div>
                </div>

                <div className="bar-group">
                  <div className="bar-label">
                    <span>🟡 Нейтральные</span>
                    <span>{neuPercent}%</span>
                  </div>
                  <div className="bar-bg">
                    <div className="bar-fill neutral" style={{ width: `${neuPercent}%` }}></div>
                  </div>
                </div>

                <div className="bar-group">
                  <div className="bar-label">
                    <span>🔴 Отрицательные</span>
                    <span>{negPercent}%</span>
                  </div>
                  <div className="bar-bg">
                    <div className="bar-fill negative" style={{ width: `${negPercent}%` }}></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
