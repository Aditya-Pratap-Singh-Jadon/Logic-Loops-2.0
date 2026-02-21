import React, { useEffect, useState } from 'react';
import TokenCard from '../components/TokenCard';
import { getContract, formatEther } from '../utils/contract';
import { ethers } from 'ethers';
import Balance from '../components/Balance'
const Dashboard = ({ wallet }) => {
  const [balance, setBalance] = useState('0');
  const [ethBalance, setEthBalance] = useState('0');
  const [tokenPrice, setTokenPrice] = useState('0');

  useEffect(() => {
    if (wallet) {
      fetchData();
      // Live Sync: Poll every 5 seconds
      const interval = setInterval(() => {
        console.log("Syncing Dashboard...");
        fetchData();
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [wallet]);

  const fetchData = async () => {
    try {
      const provider = new ethers.BrowserProvider(window.ethereum);
      const contract = await getContract(provider);
      
      // Get CUBEX Balance
      // Note: contract might not be deployed yet, handle error
      try {
        const bal = await contract.balanceOf(wallet);
        setBalance(formatEther(bal));
      } catch (e) {
        console.warn("Contract not ready or network mismatch", e);
        setBalance("0.00");
      }

      // Get ETH Balance
      const ethBal = await provider.getBalance(wallet);
      setEthBalance(formatEther(ethBal));

      // Get Token Price
      try {
        const price = await contract.getTokenPrice();
        setTokenPrice(formatEther(price));
      } catch (e) {
         // Fallback hardcoded if contract call fails (e.g. not deployed)
         setTokenPrice("0.080321");
      }

    } catch (error) {
      console.error("Error fetching dashboard data", error);
    }
  };

  return (
    <div className="container" style={{ paddingTop: '50px' }}>
      <h2 style={{ marginBottom: '30px' }}>Your Dashboard</h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '30px', marginBottom: '50px' }}>
         <TokenCard title="CUBEX Balance" value={`${parseFloat(balance).toFixed(4)} CUBEX`} />
         <TokenCard title="ETH Balance" value={`${parseFloat(ethBalance).toFixed(4)} ETH`} />
         <TokenCard title="Current Price" value={`${tokenPrice} ETH`} subtext="≈ ₹20,000 INR" />
         <Balance/>
      </div>
    </div>
  );
};

export default Dashboard;
