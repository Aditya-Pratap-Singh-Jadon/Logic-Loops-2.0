import React, { useState, useEffect } from 'react';
import { ethers } from 'ethers';

const WalletConnectButton = ({ wallet, setWallet }) => {
  const connectWallet = async () => {
    if (window.ethereum) {
      try {
        const provider = new ethers.BrowserProvider(window.ethereum);
        const signer = await provider.getSigner();
        setWallet(await signer.getAddress());
      } catch (error) {
        console.error("Connection failed", error);
      }
    } else {
      alert("Please install MetaMask!");
    }
  };

  useEffect(() => {
    // Auto connect if permission tokens exist
    if (window.ethereum && window.ethereum.selectedAddress) {
       connectWallet();
    }
  }, []);

  return (
    <button 
      onClick={connectWallet} 
      className="btn-primary"
      style={{
        background: wallet ? 'transparent' : 'var(--primary-color)',
        border: wallet ? '1px solid var(--primary-color)' : 'none',
        color: wallet ? 'var(--primary-color)' : '#000'
      }}
    >
      {wallet ? `${wallet.substring(0, 6)}...${wallet.substring(wallet.length - 4)}` : 'Connect Wallet'}
    </button>
  );
};

export default WalletConnectButton;
