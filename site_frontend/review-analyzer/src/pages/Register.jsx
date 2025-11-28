import { useState } from 'react';
import { register } from '../services/auth';
import { useNavigate } from 'react-router-dom';
import './Register.css';

export default function Register() {
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const form = e.target;
    const login = form.login.value.trim();
    const password = form.password.value;
    const confirmPassword = form.confirmPassword.value;

    // Валидация на фронте
    if (login.length < 3) {
      setError('Логин должен быть не менее 3 символов');
      return;
    }
    if (password.length < 6) {
      setError('Пароль должен быть не менее 6 символов');
      return;
    }
    if (!/[A-Za-z]/.test(password) || !/\d/.test(password)) {
      setError('Пароль должен содержать хотя бы одну букву и одну цифру');
      return;
    }
    if (password !== confirmPassword) {
      setError('Пароли не совпадают');
      return;
    }

    try {
      await register(login, password, confirmPassword);
      alert('Регистрация успешна! Войдите в аккаунт');
      navigate('/login');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="page auth">
      <h1>Регистрация</h1>
      {error && <p className="error">{error}</p>}
      <form onSubmit={handleSubmit} className="auth-form">
        <input name="login" placeholder="Логин" required autoFocus />
        <input name="password" type="password" placeholder="Пароль" required />
        <input name="confirmPassword" type="password" placeholder="Подтвердите пароль" required />
        <button type="submit">Зарегистрироваться</button>
      </form>
      <p>
        Уже есть аккаунт? <a href="/login">Войти</a>
      </p>
    </div>
  );
}
