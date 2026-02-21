import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Balance = ({ wallet }) => {
  const [balance, setBalance] = useState(0);
  const [loading, setLoading] = useState(false);

  // Fetch balance whenever 'wallet' changes
  useEffect(() => {
    if (!wallet) return;

    const fetchBalance = async () => {
      setLoading(true);
      try {
        const res = await axios.get(`http://localhost:5000/api/users/${wallet}/balance`);
        setBalance(res.data.balance);
      } catch (error) {
        console.error("Error fetching balance:", error);
        setBalance(0);
      } finally {
        setLoading(false);
      }
    };

    fetchBalance();
    
    // Optional: Refresh every 5 seconds to keep it live
    const interval = setInterval(fetchBalance, 5000);
    return () => clearInterval(interval);

  }, [wallet]);

  if (!wallet) return <div>Please connect wallet</div>;

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Minted Token Balance</h3>
      <div style={styles.amount}>
        {loading ? '...' : balance.toLocaleString()} 
        <span style={styles.currency}> CBX</span>
      </div>
    </div>
  );
};

const styles = {
  container: {
    padding: '20px',
    background: '#1e293b',
    borderRadius: '12px',
    border: '1px solid #334155',
    color: 'white',
    display: 'inline-block',
    minWidth: '200px'
  },
  title: {
    margin: '0 0 10px 0',
    fontSize: '0.9rem',
    color: '#94a3b8'
  },
  amount: {
    fontSize: '2rem',
    fontWeight: 'bold',
    color: '#10b981'
  },
  currency: {
    fontSize: '1rem',
    color: '#64748b'
  }
};

export default Balance;