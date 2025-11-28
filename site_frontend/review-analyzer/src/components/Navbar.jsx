import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

export default function Navbar() {
  const { user } = useAuth();
  const navigate = useNavigate();

  return (
    <nav className="navbar">
      <div className="nav-brand" onClick={() => navigate('/')}>
        ReviewAnalyzer
      </div>

      <div className="nav-links">
        {user ? (
          <button onClick={() => navigate('/profile')}>Профиль</button>
        ) : (
          <button onClick={() => navigate('/login')}>Войти</button>
        )}
      </div>
    </nav>
  );
}
