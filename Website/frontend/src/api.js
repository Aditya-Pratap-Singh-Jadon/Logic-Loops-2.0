import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

export const api = {
  // Stats
  getStats: () => axios.get(`${API_URL}/stats`),
  
  // User Balance
  getBalance: (userId) => axios.get(`${API_URL}/users/${userId}/balance`),
  
  // History
  getHistory: (userId) => axios.get(`${API_URL}/tokens/history`, { params: { userId } }),
  
  // Mint Tokens
  mintTokens: (data) => axios.post(`${API_URL}/tokens/mint`, data),
  
  // Update Balance (Add/Subtract)
  updateBalance: (data) => axios.post(`${API_URL}/users/update-balance`, data),
  
  // Blockchain Placeholders (if you implement the controller later)
  getPrice: () => axios.get(`${API_URL}/price`),
};