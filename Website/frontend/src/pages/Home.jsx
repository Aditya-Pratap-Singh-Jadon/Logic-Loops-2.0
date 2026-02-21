import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div className="container" style={{ textAlign: 'center', paddingTop: '100px' }}>
      <h1 style={{ fontSize: '4rem', marginBottom: '20px' }}>
        <span className="text-gradient">CubeX</span> Exchange
      </h1>
      <p style={{ fontSize: '1.5rem', color: '#aaa', maxWidth: '800px', margin: '0 auto 40px' }}>
        The premier marketplace for Carbon Credit assets. secure, transparent, and decentralized trading on Sepolia.
      </p>
      
      <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', marginBottom: '80px' }}>
        <Link to="/trade" className="btn-primary" style={{ padding: '15px 40px', fontSize: '1.2rem', textDecoration: 'none' }}>
          Start Trading
        </Link>
        <Link to="/dashboard" className="glass-card" style={{ padding: '15px 40px', fontSize: '1.2rem', textDecoration: 'none', color: '#fff', border: '1px solid var(--border-color)' }}>
          View Dashboard
        </Link>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '30px', textAlign: 'left' }}>
        <div className="glass-card">
          <h3 style={{ color: 'var(--primary-color)' }}>Carbon Credits</h3>
          <p>Buy and sell CUBEX tokens backed by verified carbon offsets.</p>
        </div>
        <div className="glass-card">
          <h3 style={{ color: 'var(--primary-color)' }}>Fixed Pricing</h3>
          <p>Transparent pricing model with 1 CUBEX = ₹20,000 INR.</p>
        </div>
        <div className="glass-card">
          <h3 style={{ color: 'var(--primary-color)' }}>Instant Settlement</h3>
          <p>Powered by Ethereum Sepolia for secure and fast transactions.</p>
        </div>
      </div>
    </div>
  );
};

export default Home;
