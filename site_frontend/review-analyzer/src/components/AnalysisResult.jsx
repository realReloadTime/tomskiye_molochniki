export default function AnalysisResult({ result }) {
  return (
    <div className="result-card">
      <h3>Результат анализа</h3>
      <div className="stats-grid">
        <div className="stat positive">
          <h4>Положительные</h4>
          <p>{result.positive}</p>
          <div className="bar"><span style={{ width: `${(result.positive / result.total) * 100}%` }}></span></div>
        </div>
        <div className="stat neutral">
          <h4>Нейтральные</h4>
          <p>{result.neutral}</p>
          <div className="bar"><span style={{ width: `${(result.neutral / result.total) * 100}%` }}></span></div>
        </div>
        <div className="stat negative">
          <h4>Отрицательные</h4>
          <p>{result.negative}</p>
          <div className="bar"><span style={{ width: `${(result.negative / result.total) * 100}%` }}></span></div>
        </div>
      </div>
      <p><strong>Всего:</strong> {result.total}</p>
    </div>
  );
}
