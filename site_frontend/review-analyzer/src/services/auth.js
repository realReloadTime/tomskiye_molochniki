const API_URL = '/api';

// Регистрация
export const register = async (login, password, confirmPassword) => {
  const response = await fetch(`${API_URL}/Registration`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ login, password, confirmPassword }),
    credentials: 'include' // ← отправляем куки
  });

  const data = await response.text();
  if (!response.ok) throw new Error(data);
  return data; // "Пользователь зарегистрирован"
};

// Вход
export const login = async (login, password) => {
  const response = await fetch(`${API_URL}/Auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ login, password }),
    credentials: 'include'
  });

  const data = await response.text();
  if (!response.ok) throw new Error(data);
  return data; // "Успешный вход"
};

// Выход
export const logout = async () => {
  const response = await fetch(`${API_URL}/Auth/logout`, {
    method: 'POST',
    credentials: 'include'
  });

  const data = await response.text();
  if (!response.ok) throw new Error(data);
  return data; // "Выход выполнен успешно"
};

// Проверка авторизации
export const checkAuth = async () => {
  const response = await fetch(`${API_URL}/Auth/check`, {
    method: 'GET',
    credentials: 'include'
  });

  if (!response.ok) return { isAuthenticated: false };
  return await response.json();
};
