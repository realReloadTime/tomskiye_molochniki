import { useState } from 'react';
import { login } from '../services/auth';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const form = e.target;
    const loginValue = form.login.value;
    const password = form.password.value;

    try {
      await login(loginValue, password);
      navigate('/profile');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="page auth">
      <h1>Вход в аккаунт</h1>
      {error && <p className="error">{error}</p>}
      <form onSubmit={handleSubmit}>
        <input name="login" placeholder="Логин" required />
        <input name="password" type="password" placeholder="Пароль" required />
        <button type="submit">Войти</button>
      </form>
      <p><a href="/register">Нет аккаунта? Зарегистрироваться</a></p>
    </div>
  );
}
