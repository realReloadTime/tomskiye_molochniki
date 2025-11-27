import { useState } from 'react';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Profile from './pages/Profile';
import './App.css';

export default function App() {
  const [page, setPage] = useState('home'); // 'home' или 'profile'

  return (
    <div className="app">
      <Navbar active={page} onChange={setPage} />
      <main className="main">
        {page === 'home' && <Home />}
        {page === 'profile' && <Profile />}
      </main>
    </div>
  );
}
