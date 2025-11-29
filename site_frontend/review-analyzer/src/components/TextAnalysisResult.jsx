export default function FileAnalysisResult({ result }) {
  if (!result) return null;

  // Добавьте проверку на наличие records
  const records = result.records || [];
  
  return (
    <div className="file-analysis-modal">
      <div className="modal-content">
        <div className="modal-header">
          <h2>📊 Результат анализа файла</h2>
        </div>
        
        <div className="modal-body">
          <div className="file-info">
            {/* УБРАТЬ fileName - его нет в ответе бэкенда */}
            <p><strong>📈 Всего записей:</strong> {result.totalRecords || 0}</p>
            <p><strong>🟢 Позитивных:</strong> {result.positiveCount || 0}</p>
            <p><strong>🔴 Негативных:</strong> {result.negativeCount || 0}</p>
            <p><strong>📅 Дата анализа:</strong> {result.analysisDate ? new Date(result.analysisDate).toLocaleString() : 'Не указана'}</p>
          </div>

          {/* Показываем таблицу только если есть records */}
          {records.length > 0 && (
            <div className="records-list">
              <h4>Детали анализа:</h4>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Текст</th>
                      <th>Вердикт</th>
                      <th>Уверенность</th>
                    </tr>
                  </thead>
                  <tbody>
                    {records.slice(0, 10).map((record, index) => (
                      <tr key={index}>
                        <td className="text-cell">
                          {record.comment?.length > 100 
                            ? record.comment.substring(0, 100) + '...' 
                            : record.comment
                          }
                        </td>
                        <td className={record.classLabel === 1 ? 'toxic' : 'non-toxic'}>
                          {record.classLabel === 1 ? '🔴 Токсичный' : '🟢 Нетоксичный'}
                        </td>
                        <td className="confidence">
                          {((record.probability || 0) * 100).toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {records.length > 10 && (
                  <p className="more-records">
                    ... и еще {records.length - 10} записей
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Сообщение если records нет */}
          {records.length === 0 && (
            <div className="no-records">
              <p>Детальная информация по записям недоступна</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}