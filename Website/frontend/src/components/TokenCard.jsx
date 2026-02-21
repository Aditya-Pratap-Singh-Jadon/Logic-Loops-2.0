import React from 'react';

const TokenCard = ({ title, value, subtext }) => {
  return (
    <div className="glass-card" style={{ flex: 1, minWidth: '250px', textAlign: 'center' }}>
      <h3 style={{ color: 'var(--text-secondary)', marginBottom: '10px' }}>{title}</h3>
      <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--primary-color)' }}>{value}</div>
      {subtext && <div style={{ fontSize: '0.9rem', marginTop: '5px', color: '#888' }}>{subtext}</div>}
    </div>
  );
};

export default TokenCard;
