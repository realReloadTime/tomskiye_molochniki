export default function Navbar({ active, onChange }) {
  return (
    <nav className="navbar">
      <div className="nav-brand">ReviewAnalyzer</div>
      <div className="nav-links">
        <button
          className={active === 'home' ? 'active' : ''}
          onClick={() => onChange('home')}
        >
          Анализ
        </button>
        <button
          className={active === 'profile' ? 'active' : ''}
          onClick={() => onChange('profile')}
        >
          Кабинет
        </button>
      </div>
    </nav>
  );
}
