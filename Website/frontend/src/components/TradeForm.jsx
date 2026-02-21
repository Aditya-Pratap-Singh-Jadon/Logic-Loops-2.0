import React, { useState } from 'react';

const TradeForm = ({ type, onTrade, balance, price }) => {
  const [amount, setAmount] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!amount) return;
    setLoading(true);
    await onTrade(type, amount);
    setLoading(false);
    setAmount('');
  };

  const estimatedCost = amount ? (parseFloat(amount) * price).toFixed(6) : '0';

  return (
    <div className="glass-card">
      <h2 style={{ marginBottom: '20px' }}>{type === 'BUY' ? 'Buy CUBEX' : 'Sell CUBEX'}</h2>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', color: '#888' }}>Amount (CUBEX)</label>
          <input 
            type="number" 
            step="0.000001" 
            className="input-field"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            placeholder="0.0"
          />
        </div>
        
        <div style={{ marginBottom: '20px', padding: '10px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
            <span>Price per Token:</span>
            <span>{price} ETH</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 'bold' }}>
            <span>Total {type === 'BUY' ? 'Cost' : 'Value'}:</span>
            <span style={{ color: 'var(--primary-color)' }}>{estimatedCost} ETH</span>
          </div>
        </div>

        <button 
          type="submit" 
          className="btn-primary" 
          style={{ width: '100%' }}
          disabled={loading}
        >
          {loading ? 'Processing...' : type}
        </button>
      </form>

      <div style={{ marginTop: '15px', textAlign: 'center', fontSize: '0.9rem', color: '#888' }}>
        Balance: {balance} CUBEX
      </div>
    </div>
  );
};

export default TradeForm;
