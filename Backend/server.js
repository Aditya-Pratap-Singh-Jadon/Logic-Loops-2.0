require('dotenv').config();
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const blockchainController = require('./controllers/blockchainController');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(bodyParser.json());

app.get('/', (req, res) => {
    res.send('CubeX Exchange API is running');
});

const userBalances = {};
const mintHistory = [];


app.post('/api/tokens/mint', async (req, res) => {
  try {
    const { userId, amount, transactionId, source, timestamp } = req.body;
    
    // Validate request
    if (!userId || !amount || !transactionId) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields'
      });
    }
    
    // Check if transaction already processed (idempotency)
    const existingTxn = mintHistory.find(t => t.transactionId === transactionId);
    if (existingTxn) {
      return res.status(200).json({
        success: true,
        message: 'Transaction already processed',
        transaction: existingTxn,
        newBalance: userBalances[userId] || 0
      });
    }
    
    // Initialize user balance if needed
    if (!userBalances[userId]) {
      userBalances[userId] = 0;
    }
    
    // Mint tokens (add to balance)
    userBalances[userId] += amount;
    
    // Record transaction
    const transaction = {
      transactionId,
      userId,
      amount,
      source,
      timestamp,
      mintedAt: new Date().toISOString(),
      status: 'completed'
    };
    
    mintHistory.push(transaction);
    
    console.log(`✅ Minted ${amount} tokens for ${userId} (Transaction: ${transactionId})`);
    
    res.json({
      success: true,
      message: `Successfully minted ${amount} tokens`,
      transaction,
      newBalance: userBalances[userId]
    });
    
  } catch (error) {
    console.error('Error minting tokens:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * POST /api/users/update-balance
 * 
 * Update user balance (add or subtract)
 * 
 * Request Body:
 * {
 *   "userId": "ENT_001",
 *   "tokens": 123.45,
 *   "value": 61.73,
 *   "operation": "add",  // or "subtract"
 *   "timestamp": "2024-01-01T12:00:00"
 * }
 */
app.post('/api/users/update-balance', async (req, res) => {
  try {
    const { userId, tokens, value, operation, timestamp } = req.body;
    
    // Validate request
    if (!userId || tokens === undefined || !operation) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields'
      });
    }
    
    // Initialize user balance if needed
    if (!userBalances[userId]) {
      userBalances[userId] = 0;
    }
    
    // Update balance
    const oldBalance = userBalances[userId];
    
    if (operation === 'add') {
      userBalances[userId] += tokens;
    } else if (operation === 'subtract') {
      userBalances[userId] -= tokens;
      // Prevent negative balance
      if (userBalances[userId] < 0) {
        userBalances[userId] = 0;
      }
    } else {
      return res.status(400).json({
        success: false,
        error: 'Invalid operation. Use "add" or "subtract"'
      });
    }
    
    console.log(`💰 Updated balance for ${userId}: ${oldBalance} → ${userBalances[userId]} (${operation} ${tokens})`);
    
    res.json({
      success: true,
      message: `Balance updated successfully`,
      userId,
      operation,
      amount: tokens,
      value,
      oldBalance,
      newBalance: userBalances[userId],
      timestamp
    });
    
  } catch (error) {
    console.error('Error updating balance:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * GET /api/users/:userId/balance
 * 
 * Get user's current balance
 */
app.get('/api/users/:userId/balance', (req, res) => {
  const { userId } = req.params;
  
  const balance = userBalances[userId] || 0;
  
  res.json({
    success: true,
    userId,
    balance,
    lastUpdated: new Date().toISOString()
  });
});

/**
 * GET /api/tokens/history
 * 
 * Get minting history
 */
app.get('/api/tokens/history', (req, res) => {
  const { userId } = req.query;
  
  let history = mintHistory;
  
  if (userId) {
    history = mintHistory.filter(t => t.userId === userId);
  }
  
  res.json({
    success: true,
    count: history.length,
    history
  });
});

/**
 * GET /api/stats
 * 
 * Get overall statistics
 */
app.get('/api/stats', (req, res) => {
  const totalMinted = mintHistory.reduce((sum, t) => sum + t.amount, 0);
  const totalUsers = Object.keys(userBalances).length;
  const totalBalance = Object.values(userBalances).reduce((sum, b) => sum + b, 0);
  
  res.json({
    success: true,
    stats: {
      totalUsers,
      totalMinted,
      totalBalance,
      totalTransactions: mintHistory.length
    }
  });
});

/**
 * Health check
 */
app.get('/health', (req, res) => {
  res.json({
    success: true,
    service: 'Express Backend',
    timestamp: new Date().toISOString()
  });
});
app.get('/api/price', blockchainController.getTokenPrice);
app.post('/api/buy', blockchainController.buyTokens);
app.post('/api/sell', blockchainController.sellTokens);
app.get('/api/history/:wallet', blockchainController.getTransactionHistory);

app.listen(PORT, () => {
  console.log('='.repeat(70));
  console.log('EXPRESS BACKEND - EXAMPLE IMPLEMENTATION');
  console.log('='.repeat(70));
  console.log(`Server running on http://localhost:${PORT}`);
  console.log('\nAvailable Endpoints:');
  console.log('  POST   /api/tokens/mint');
  console.log('  POST   /api/users/update-balance');
  console.log('  GET    /api/users/:userId/balance');
  console.log('  GET    /api/tokens/history');
  console.log('  GET    /api/stats');
  console.log('  GET    /health');
  console.log('='.repeat(70));
});




// Example usage for testing:
/*
// Test minting
curl -X POST http://localhost:5000/api/tokens/mint \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "ENT_001",
    "amount": 100,
    "transactionId": "TXN_TEST_001",
    "source": "carbon_capture",
    "timestamp": "2024-01-01T12:00:00"
  }'

// Test balance update
curl -X POST http://localhost:5000/api/users/update-balance \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "ENT_001",
    "tokens": 50,
    "value": 25,
    "operation": "add",
    "timestamp": "2024-01-01T12:00:00"
  }'

// Get balance
curl http://localhost:5000/api/users/ENT_001/balance

// Get history
curl http://localhost:5000/api/tokens/history?userId=ENT_001

// Get stats
curl http://localhost:5000/api/stats
*/
// Start Server
