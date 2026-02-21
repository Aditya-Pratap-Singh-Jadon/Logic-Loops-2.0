import React from 'react';

const TransactionTable = ({ transactions }) => {
  if (!transactions || transactions.length === 0) {
    return <div style={{ textAlign: 'center', color: '#888', padding: '20px' }}>No transactions found</div>;
  }

  return (
    <div className="glass-card" style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: '600px' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left' }}>
            <th style={{ padding: '15px' }}>Type</th>
            <th style={{ padding: '15px' }}>Amount</th>
            <th style={{ padding: '15px' }}>ETH Value</th>
            <th style={{ padding: '15px' }}>Hash</th>
            <th style={{ padding: '15px' }}>Status</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map((tx, index) => (
            <tr key={index} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <td style={{ padding: '15px' }}>
                <span style={{ 
                  color: tx.type === 'BUY' ? 'var(--primary-color)' : '#ff4d4d',
                  fontWeight: 'bold'
                }}>
                  {tx.type}
                </span>
              </td>
              <td style={{ padding: '15px' }}>{tx.amount} CUBEX</td>
              <td style={{ padding: '15px' }}>{tx.ethValue} ETH</td>
              <td style={{ padding: '15px' }}>
                <a 
                  href={`https://sepolia.etherscan.io/tx/${tx.hash}`} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  style={{ color: 'var(--text-secondary)', textDecoration: 'underline' }}
                >
                  {tx.hash.substring(0, 10)}...
                </a>
              </td>
              <td style={{ padding: '15px', color: '#888' }}>Confirmed</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default TransactionTable;
