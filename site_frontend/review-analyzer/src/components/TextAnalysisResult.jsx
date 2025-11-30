export default function TextAnalysisResult({ result }) {
  if (!result) return null;

  // Определяем цвета по classLabel
  const getVerdictConfig = (label) => {
    const map = {
      positive: { text: 'Положительный', color: 'positive' },
      neutral: { text: 'Нейтральный', color: 'neutral' },
      negative: { text: 'Отрицательный', color: 'negative' }
    };
    return map[label] || map.negative;
  };

  const verdict = getVerdictConfig(result.classLabel);

  const getConfidenceLevel = (prob) => {
    if (prob > 0.8) return 'high';
    if (prob > 0.5) return 'medium';
    return 'low';
  };

  const confidenceLevel = getConfidenceLevel(result.probability);

  return (
    <div className="analysis-modal" onClick={(e) => e.stopPropagation()}>
      <div className="modal-content">
        <div className="modal-header">
          <h2>💬 Анализ отзыва</h2>
        </div>

        <div className="modal-body">
          <div className="text-section">
            <div className="section-title">Ваш отзыв</div>
            <div className="text-bubble">
              {result.comment}
            </div>
          </div>

          <div className="result-section">
            <div className="verdict-item">
              <div className="label">Вердикт</div>
              <div className={`verdict ${verdict.color}`}>
                {verdict.text}
              </div>
            </div>

            <div className="confidence-item">
              <div className="label">Уверенность</div>
              <div className="confidence-container">
                <div className="confidence-bar">
                  <div
                    className={`confidence-fill ${verdict.color}`}
                    style={{ width: `${(result.probability * 100)}%` }}
                  ></div>
                </div>
                <div className="confidence-text">
                  {(result.probability * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
