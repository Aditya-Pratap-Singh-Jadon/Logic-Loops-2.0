import React, { useState, useEffect } from 'react';
import TradeForm from '../components/TradeForm';
import TokenCard from '../components/TokenCard';
import { getContract, formatEther } from '../utils/contract';
import { ethers } from 'ethers';

const Trade = ({ wallet }) => {
  const [balance, setBalance] = useState('0');
  const [ethBalance, setEthBalance] = useState('0');
  const [tokenPrice, setTokenPrice] = useState('0'); // ETH per Token
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (wallet) {
      fetchData();
    }
  }, [wallet]);

  const fetchData = async () => {
    try {
      // Fetch price from backend or contract
      // Requirement says "GET /api/price returns fixed price"
      const res = await fetch('http://localhost:5000/api/price');
      const data = await res.json();
      setTokenPrice(data.eth); // 0.080321

      // Fetch balances
      const provider = new ethers.BrowserProvider(window.ethereum);
      const contract = await getContract(provider);
      
      try {
        const bal = await contract.balanceOf(wallet);
        setBalance(formatEther(bal));
      } catch (e) {
        console.warn("Contract read failed", e);
      }

      const ethBal = await provider.getBalance(wallet);
      setEthBalance(formatEther(ethBal));

    } catch (error) {
      console.error(error);
    }
  };

  const handleTrade = async (type, amount) => {
    if (!wallet) {
      alert("Please connect wallet");
      return;
    }
    
    setMessage(`Processing ${type}...`);
    try {
      const endpoint = type === 'BUY' ? '/api/buy' : '/api/sell';
      
      // If Selling, user needs to Approve/Transfer tokens to/via backend?
      // Requirement says "Calls backend -> sell endpoint".
      // If backend handles logic, it might expect User to sign a TX on frontend to transfer tokens?
      // Or backend has "custodial" access?
      // "Marketplace allows ... enterprises to buy and sell ... AI pipeline mints to enterprise wallets"
      // If Enterprise Wallet (User) holds tokens, Backend CANNOT take them without approval.
      
      // However, for this deliverable, if I follow instruction "Calls backend -> sell endpoint"
      // and "Security: Use private key from .env", maybe the backend acts as an *operator*?
      // But user wallet (MetaMask) is connected.
      
      // Implementation:
      // If BUY: Backend (Admin) sends tokens to User.
      // If SELL: User must specific send tokens to Admin.
      // So Frontend should probably handle SELL via Contract directly?
      // "Section 2: Sell Tokens ... Calls backend -> sell endpoint".
      
      // I will implement the API call as requested.
      // Note: If this fails because Backend can't move User's tokens, I will add a comment.
      // But maybe the "Wallet" is not the User's MetaMask, but a platform account?
      // "Show Connected wallet address"
      
      // Let's stick to API.
        
      const response = await fetch(`http://localhost:5000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ walletAddress: wallet, amount })
      });
      
      const result = await response.json();
      
      if (result.success) {
        setMessage(`Success! TX Hash: ${result.txHash}`);
        fetchData(); // Refresh balance
      } else {
        setMessage(`Error: ${result.error}`);
      }
    } catch (error) {
      setMessage(`Failed: ${error.message}`);
    }
  };

  return (
    <div className="container" style={{ paddingTop: '50px' }}>
      <h2 style={{ marginBottom: '30px', textAlign: 'center' }}>Trade CUBEX</h2>
      <div style={{ display: 'flex', justifyContent: 'center', gap: '30px', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: '300px' }}>
          <TradeForm type="BUY" onTrade={handleTrade} balance={balance} price={tokenPrice} />
        </div>
        <div style={{ flex: 1, minWidth: '300px' }}>
          <TradeForm type="SELL" onTrade={handleTrade} balance={balance} price={tokenPrice} />
        </div>
      </div>
      {message && (
        <div style={{ marginTop: '30px', padding: '15px', background: 'rgba(255,255,255,0.1)', borderRadius: '8px', textAlign: 'center' }}>
          {message}
        </div>
      )}
    </div>
  );
};

export default Trade;
