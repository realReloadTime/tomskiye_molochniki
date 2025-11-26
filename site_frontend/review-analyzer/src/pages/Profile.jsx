export default function Profile() {
  return (
    <div className="page profile">
      <h1>Личный кабинет</h1>
      <div className="card">
        <h2>Ваш аккаунт</h2>
        <p><strong>Почта:</strong> user@example.com</p>
        <p><strong>Дата регистрации:</strong> 1 января 2025</p>
      </div>
      <div className="card">
        <h2>История анализов</h2>
        <p>Ваши данные синхронизируются с сервером.</p>
        <div className="placeholder-list">
          <div className="placeholder-item">
            <strong>Анализ #123</strong>
            <span>10 отзывов • 26 ноября 2025</span>
          </div>
          <div className="placeholder-item">
            <strong>Анализ #122</strong>
            <span>5 отзывов • 25 ноября 2025</span>
          </div>
          <div className="placeholder-item">
            <strong>Анализ #121</strong>
            <span>8 отзывов • 24 ноября 2025</span>
          </div>
        </div>
      </div>
    </div>
  );
}
