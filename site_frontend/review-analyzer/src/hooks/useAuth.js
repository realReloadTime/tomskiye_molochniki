import { useState, useEffect } from 'react';
import { checkAuth, logout as apiLogout } from '../services/auth';

export const useAuth = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const check = async () => {
    try {
      const data = await checkAuth();
      if (data.isAuthenticated) {
        setUser({
          id: data.userId,
          login: data.login
        });
      } else {
        setUser(null);
      }
    } catch (err) {
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    try {
      await apiLogout();
    } catch (err) {
      console.error(err);
    } finally {
      setUser(null);
    }
  };

  useEffect(() => {
    check();
  }, []);

  return { user, loading, logout, refresh: check };
};
