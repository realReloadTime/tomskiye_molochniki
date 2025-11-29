import './TextAnalysisResult.css'; // стили для красивого отображения

export default function TextAnalysisResult({ result }) {
  if (!result) return null;

  const isToxic = result.classLabel === 1;
  const confidence = (result.probability * 100).toFixed(2);
  
  return (
    <div className="analysis-modal">
      <div className="modal-content">
        <div className="modal-header">
          <h2>🎯 Результат анализа тональности</h2>
        </div>
        
        <div className="modal-body">
          <div className="text-section">
            <div className="section-title">📝 Анализируемый текст:</div>
            <div className="text-bubble">"{result.comment}"</div>
          </div>
          
          <div className="result-section">
            <div className="verdict-item">
              <span className="label">🏷️ Вердикт:</span>
              <span className={`verdict ${isToxic ? 'toxic' : 'non-toxic'}`}>
                {isToxic ? '🔴 Токсичный' : '🟢 Нетоксичный'}
              </span>
            </div>
            
            <div className="confidence-item">
              <span className="label">📊 Уверенность:</span>
              <div className="confidence-container">
                <div className="confidence-bar">
                  <div 
                    className={`confidence-fill ${confidence > 70 ? 'high' : confidence > 40 ? 'medium' : 'low'}`}
                    style={{ width: `${confidence}%` }}
                  ></div>
                </div>
                <span className="confidence-text">{confidence}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}