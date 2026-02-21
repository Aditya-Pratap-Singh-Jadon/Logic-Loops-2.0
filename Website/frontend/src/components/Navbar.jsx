import React from 'react';
import { Link } from 'react-router-dom';
import WalletConnectButton from './WalletConnectButton';

const Navbar = ({ wallet, setWallet }) => {
  return (
    <nav style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '20px 40px',
      background: 'rgba(0,0,0,0.5)',
      backdropFilter: 'blur(10px)',
      borderBottom: '1px solid var(--border-color)',
      position: 'sticky',
      top: 0,
      zIndex: 100
    }}>
      <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
        <Link to="/" className="text-gradient" style={{ textDecoration: 'none' }}>CubeX</Link>
      </div>
      <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
        <Link to="/" style={{ color: '#fff', fontSize: '1rem' }}>Home</Link>
        <Link to="/dashboard" style={{ color: '#fff', fontSize: '1rem' }}>Dashboard</Link>
        <Link to="/trade" style={{ color: '#fff', fontSize: '1rem' }}>Trade</Link>
        <Link to="/history" style={{ color: '#fff', fontSize: '1rem' }}>History</Link>
        <WalletConnectButton wallet={wallet} setWallet={setWallet} />
      </div>
    </nav>
  );
};

export default Navbar;
