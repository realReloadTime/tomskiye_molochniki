import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useAuth } from './hooks/useAuth';
import Home from './pages/Home';
import Profile from './pages/Profile';
import Login from './pages/Login';
import Register from './pages/Register';
import Navbar from './components/Navbar';
import './App.css';

// Простой компонент-обёртка для страниц с навбаром
function Layout({ children }) {
  return (
    <>
      <Navbar />
      {children}
    </>
  );
}

function AppContent() {
  const { loading } = useAuth();

  if (loading) {
    return <div className="loading">Загрузка...</div>;
  }

  return (
    <Routes>
      <Route
        path="/"
        element={
          <Layout>
            <Home />
          </Layout>
        }
      />
      <Route
        path="/profile"
        element={
          <Layout>
            <Profile />
          </Layout>
        }
      />
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}
