import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import Trade from './pages/Trade';
import History from './pages/History';

function App() {
  const [wallet, setWallet] = useState('');

  return (
    <Router>
      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        <Navbar wallet={wallet} setWallet={setWallet} />
        <main style={{ flex: 1, paddingBottom: '50px' }}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/dashboard" element={<Dashboard wallet={wallet} />} />
            <Route path="/trade" element={<Trade wallet={wallet} />} />
            <Route path="/history" element={<History wallet={wallet} />} />
          </Routes>
        </main>
        <footer style={{ 
          textAlign: 'center', 
          padding: '20px', 
          borderTop: '1px solid var(--border-color)',
          color: '#888',
          fontSize: '0.9rem'
        }}>
          &copy; {new Date().getFullYear()} CubeX Exchange. All rights reserved.
        </footer>
      </div>
    </Router>
  );
}

export default App;
