import React, { useState, useEffect } from 'react';
import TransactionTable from '../components/TransactionTable';

const History = ({ wallet }) => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (wallet) {
      fetchHistory();
    }
  }, [wallet]);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:5000/api/history/${wallet}`);
      const data = await res.json();
      
      if (Array.isArray(data)) {
          setHistory(data);
      } else {
          console.error("API Error or Invalid Data:", data);
          setHistory([]);
      }
    } catch (error) {
      console.error("Fetch error:", error);
      setHistory([]);
    }
    setLoading(false);
  };

  return (
    <div className="container" style={{ paddingTop: '50px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
        <h2>Transaction History</h2>
        <button onClick={fetchHistory} className="btn-primary" style={{ padding: '8px 16px', fontSize: '0.9rem' }}>
          Refresh
        </button>
      </div>
      
      {loading ? (
        <div style={{ textAlign: 'center', padding: '50px' }}>Loading...</div>
      ) : (
        <TransactionTable transactions={history} />
      )}
    </div>
  );
};

export default History;
