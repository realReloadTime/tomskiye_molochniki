import './TextAnalysisResult.css';

export default function TextAnalysisResult({ result }) {
  if (!result) return null;


  console.log('TextAnalysisResult received:', result);

  
  const classLabel = result.class_label !== undefined ? result.class_label : result.classLabel;
  const probability = result.probability;

  console.log('Normalized classLabel:', classLabel);

  const getToneInfo = (classLabel) => {
    console.log('getToneInfo called with:', classLabel);
    switch(classLabel) {
      case 0: return { text: '🟢 Позитивный', className: 'positive' };
      case 1: return { text: '🟡 Нейтральный', className: 'neutral' };
      case 2: return { text: '🔴 Негативный', className: 'negative' };
      default: return { text: '❓ Неизвестно', className: 'unknown' };
    }
  };

  const toneInfo = getToneInfo(classLabel);
  const confidence = Math.round(probability);
  
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
              <span className="label">🏷️ Тональность:</span>
              <span className={`verdict ${toneInfo.className}`}>
                {toneInfo.text}
              </span>
            </div>
            
            <div className="confidence-item">
              <span className="label">📊 Уверенность:</span>
              <div className="confidence-container">
                <div className="confidence-bar">
                  <div 
                    className={`confidence-fill ${toneInfo.className}`}
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