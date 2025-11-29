export default function FileAnalysisResult({ result }) {
  if (!result) return null;

  return (
    <div className="file-analysis-modal">
      <div className="modal-content">
        <div className="modal-header">
          <h2>📊 Результат анализа файла</h2>
        </div>
        
        <div className="modal-body">
          <div className="file-info">
            <p><strong>📁 Файл:</strong> {result.fileName}</p>
            <p><strong>📈 Всего записей:</strong> {result.totalRecords}</p>
            <p><strong>🟢 Нетоксичных:</strong> {result.nonToxicCount}</p>
            <p><strong>🔴 Токсичных:</strong> {result.toxicCount}</p>
          </div>

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
                  {result.records.slice(0, 10).map((record, index) => (
                    <tr key={index}>
                      <td className="text-cell">
                        {record.comment.length > 100 
                          ? record.comment.substring(0, 100) + '...' 
                          : record.comment
                        }
                      </td>
                      <td className={record.classLabel === 1 ? 'toxic' : 'non-toxic'}>
                        {record.classLabel === 1 ? '🔴 Токсичный' : '🟢 Нетоксичный'}
                      </td>
                      <td className="confidence">
                        {(record.probability * 100).toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {result.records.length > 10 && (
                <p className="more-records">
                  ... и еще {result.records.length - 10} записей
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}