const { ethers } = require('ethers');
require('dotenv').config();

// Contract Configuration
const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS;
const RPC_URL = process.env.SEPOLIA_RPC_URL;
let PRIVATE_KEY = process.env.PRIVATE_KEY;

// Sanitize Private Key
if (PRIVATE_KEY) {
    PRIVATE_KEY = PRIVATE_KEY.replace(/\s/g, '').replace(/^["']|["']$/g, '');
    if (!PRIVATE_KEY.startsWith("0x")) PRIVATE_KEY = "0x" + PRIVATE_KEY;
}

// Normalize Contract Address
let NORMALIZED_CONTRACT_ADDRESS = CONTRACT_ADDRESS;
try {
    if (CONTRACT_ADDRESS) {
        // Force lowercase to ensure we generate the correct checksum
        // ignoring any user-provided incorrect casing
        NORMALIZED_CONTRACT_ADDRESS = ethers.getAddress(CONTRACT_ADDRESS.toLowerCase());
    }
} catch (error) {
    console.error("CRITICAL ERROR: Invalid CONTRACT_ADDRESS format in .env");
    console.error(error.message);
}

console.log("--- Configuration Check ---");
console.log("RPC_URL:", RPC_URL ? "Set" : "Missing");
console.log("CONTRACT_ADDRESS:", NORMALIZED_CONTRACT_ADDRESS ? "Set (Normalized)" : "Missing");
console.log("PRIVATE_KEY:", PRIVATE_KEY ? `Set (Length: ${PRIVATE_KEY.length})` : "Missing");

if (PRIVATE_KEY) {
    // Safe Safe Print
    const start = PRIVATE_KEY.substring(0, 4);
    const end = PRIVATE_KEY.substring(PRIVATE_KEY.length - 4);
    console.log(`Key Preview: ${start}...${end}`);
}

if (!PRIVATE_KEY || PRIVATE_KEY.includes("YOUR_PRIVATE_KEY")) {
    console.error("ERROR: PRIVATE_KEY is not set or is still the default value.");
}
console.log("---------------------------");
// const CONTRACT_ABI = require('../path/to/abi.json'); // We will need to copy this after compilation
// For now, we can hardcode the parts we need or load dynamically if file exists.
// Ideally, we copy the artifacts to backend or import from smart-contract folder if locally monorepo.

// Since we are in a monorepo structure, we can try to require the JSON from the smart-contract folder
// But handling relative paths outside src might be tricky in some environments.
// Let's assume we will copy the ABI to `backend/abi/CubeXToken.json` during setup or deployment.
// Or we can define a minimal ABI here for simplicity if the artifact isn't ready.

// Minimal ABI for now to avoid runtime errors before compilation
const MINIMAL_ABI = [
    "function getTokenPrice() view returns (uint256)",
    "function buyTokens(uint256 amount) payable",
    "function sellTokens(uint256 amount) external",
    "function transfer(address to, uint256 amount) returns (bool)",
    "event Transfer(address indexed from, address indexed to, uint256 value)",
    "event TokenPurchased(address indexed buyer, uint256 amount, uint256 cost)",
    "event TokenSold(address indexed seller, uint256 amount, uint256 refund)"
];

// We need a provider and a signer (wallet)
// The signer acts as the unexpected 'operator' if we need to do admin things,
// BUT for buy/sell:
// - BUY: User calls function? No, if API calls it, API wallet (treasury) pays ETH?
//   Wait, the Requirement says "POST /api/buy ... calls smart contract buyTokens() returns tx hash".
//   If the API calls `buyTokens`, the API's wallet (server wallet) is sending the transaction.
//   Who calls? The USER calls from Frontend usually in Web3.
//   "Frontend ... MetaMask wallet connection ... Calls backend -> buy endpoint"
//   If logic is: User -> Backend -> Contract.
//   Then Backend needs to send the tx.
//   BUT `buyTokens` is `payable` and requires ETH. 
//   If Backend calls it, Backend wallet pays ETH.
//   Where does User's ETH go? User needs to send ETH to Backend wallet first?
//   This is a custody model.
//   
//   ALTERNATIVE: "Frontend ... calls backend -> buy endpoint" might just be to record DB entry?
//   OR User signs a transaction and sends it to backend to relay? (Meta-transaction)
//   
//   Let's re-read Requirement: "POST /api/buy ... body: { walletAddress, amount } ... calls smart contract buyTokens() ... returns tx hash"
//   And "buys transfers tokens from contract treasury".
//   
//   If the user interacts with MetaMask, they should call the contract DIRECTLY from Frontend.
//   Usually "Web3 Dashboard" implies Frontend connects to Contract.
//   
//   However, if requirements say "Calls backend -> buy endpoint -> calls smart contract",
//   Maybe the User sends ETH to backend separately? 
//   Or maybe the User invokes the transaction on Frontend, and Backend is just for History?
//   
//   But "Section 1: Buy Tokens ... Buy button ... Calls backend -> buy endpoint".
//   This strongly implies the API does the interaction.
//   IF SO, the User must have deposited funds or it's a simulated environment where User pays via other means?
//   
//   Wait, "1 CUBEX = ₹20,000 INR ... Fixed price".
//   Maybe the user doesn't pay ETH?
//   "Show: ETH required ... INR equivalent".
//   
//   IF User pays ETH, valid pattern is:
//   1. User calls `buyTokens` on contract via Frontend (MetaMask).
//   2. Backend monitors events.
//   
//   BUT Requirement says: "POST /api/buy ... calls smart contract buyTokens()".
//   This implies the Backend initiates the TX.
//   If so, the backend wallet pays gas + value (if payable).
//   If `buyTokens` expects `msg.value`, the backend wallet must send ETH.
//   This effectively means the Backend is buying tokens for itself, or for the user if it transfers them.
//   Our contract says: `_transfer(address(this), msg.sender, amount);`
//   If Backend calls it, `msg.sender` is Backend Wallet.
//   So Backend buys tokens from ... itself? (Treasury to Backend Wallet). Use `transfer`?
//   
//   Maybe the requirement implies a different flow:
//   "buyTokens() payable" - this expects ETH.
//   
//   If the User is supposed to pay ETH, they MUST sign the tx.
//   Backend CANNOT sign for the user's ETH.
//   
//   HYBRID APPROACH (Most likely intended for this hackathon/task):
//   The "Buy Endpoint" might just be a wrapper to helping the interaction or maybe I misunderstood.
//   "Frontend ... Ethers.js for blockchain interaction ... MetaMask wallet connection".
//   "Trade Page ... Buy button ... Calls backend -> buy endpoint".
//   
//   This is conflicting. Standard Web3 DApp: Buy button triggers MetaMask popup.
//   Backend API might be for *off-chain* logic or *privileged* logic.
//   
//   Let's assume the standard DApp flow for the actual trade:
//   Frontend -> Contract.
//   
//   BUT strict requirement: "POST /api/buy ... calls buyTokens()".
//   Okay, maybe the user *delegates* the purchase?
//   Or maybe this is a "Gasless" transaction (Meta-transaction) or simply a design flaw in requirements.
//   
//   OR: The User sends ETH to the address, and Backend detects it?
//   
//   Let's stick to the text: "POST /api/buy ... calls smart contract buyTokens()".
//   I will implement this in the controller.
//   NOTE: This will only work if the Backend Wallet has ETH and pays for it.
//   If the USER is supposed to pay, this endpoint won't work as expected (user's ETH won't move).
//   
//   However, I see "1 CUBEX = ₹20,000 INR". Maybe payment is assumed settled off-chain (fiat)?
//   But contract has `buyTokens` as `payable`.
//   
//   I will implement the controller to call the contract.
//   I'll also providing the `getHistory` which reads events.

const getProvider = () => {
    return new ethers.JsonRpcProvider(RPC_URL);
};

const getSigner = () => {
    const provider = getProvider();
    try {
        if (!PRIVATE_KEY) throw new Error("PRIVATE_KEY is undefined or empty");
        // Log length to verify it's still available
        console.log(`[getSigner] Using Key Length: ${PRIVATE_KEY.length}`);
        return new ethers.Wallet(PRIVATE_KEY, provider);
    } catch (error) {
        console.error("[getSigner] FAILED to create wallet!");
        console.error(`[getSigner] Key Value (First 4 chars): ${PRIVATE_KEY ? PRIVATE_KEY.substring(0, 4) : 'undefined'}`);
        throw error;
    }
};

const getContract = (signerOrProvider) => {
    return new ethers.Contract(NORMALIZED_CONTRACT_ADDRESS, MINIMAL_ABI, signerOrProvider);
};

exports.getTokenPrice = async (req, res) => {
    try {
        // Return fixed prices as per requirements
        // 1 CUBEX = ₹20,000 INR
        // 1 CUBEX = 0.080321 ETH
        // We can also fetch from contract to be sure
        // const contract = getContract(getProvider());
        // const priceWei = await contract.getTokenPrice();
        // const priceEth = ethers.formatEther(priceWei);
        
        res.json({
            inr: 20000,
            eth: 0.080321
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};

exports.buyTokens = async (req, res) => {
    let { walletAddress, amount } = req.body;
    try {
        if (!walletAddress) throw new Error("Wallet Address is missing");
        walletAddress = ethers.getAddress(walletAddress.toLowerCase());
        // Implementation of Admin-driven buy
        // WARNING: This assumes the Server Wallet is paying the ETH!
        // If the User is paying, this flow is incorrect for Web3, but we follow requirements.
        
        const signer = getSigner();
        const contract = getContract(signer);
        
        // Calculate cost
        // Price: 0.080321 ETH per token.
        // Amount is number of tokens (e.g., 1, 2, 0.5)
        // Contract expects wei amount for 'amount' param? 
        // My contract: `buyTokens(uint256 amount)` and `cost = (amount * price) / 1e18`.
        // So passing 1.0 (as 1e18) means 1 Token.
        
        const amountWei = ethers.parseEther(amount.toString());
        const pricePerToken = ethers.parseEther("0.080321");
        // Cost = amount * price
        // (Since parseEther handles decimal places, we can just multiply if we are careful, 
        // distinct from contract math which does (amountWei * priceWei) / 1e18)
        
        // JS math for ETH value:
        const costWei = (BigInt(amountWei) * BigInt(pricePerToken)) / BigInt("1000000000000000000"); // / 1e18
        
        // ROBUST FIX: Send 120% of the calculated cost.
        // The contract calculates exact cost and refunds ALL difference.
        // This guarantees we never hit "Insufficient ETH" due to minor math discrepancies.
        const valueToSend = (costWei * BigInt(120)) / BigInt(100);

        console.log(`[buyTokens] Amount: ${amount} CUBEX`);
        console.log(`[buyTokens] Calculated Cost: ${costWei.toString()} Wei`);
        console.log(`[buyTokens] Sending Value (120%): ${valueToSend.toString()} Wei`);

        // Call contract
        // Note: The function transfers tokens to `msg.sender` (the Server Wallet).
        // Then we might need to transfer to the User `walletAddress`.
        // The contract logic `_transfer(address(this), msg.sender, amount)` sends to caller.
        
        // If we want to send to User, we should 'buy' then 'transfer'.
        // OR we modify contract to `buyTo(address to)`.
        // Given constraints, we can use `mint` if we have separate mint, but buyTokens is fixed.
        
        // Let's assume we call buyTokens, then transfer to user.
        // OR simply `transfer` from Treasury to User if we treat "Buy" as "User paid Fiat".
        
        // IF we execute `buyTokens`, Server Wallet pays ETH -> Treasury. Server Wallet gets Tokens.
        // Then Server Wallet transfer -> User.
        
        const tx = await contract.buyTokens(amountWei, { value: valueToSend });
        await tx.wait();
        
        // Transfer to user
        const transferTx = await contract.transfer(walletAddress, amountWei);
        await transferTx.wait();

        res.json({ success: true, txHash: transferTx.hash, buyHash: tx.hash });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: error.message });
    }
};

exports.sellTokens = async (req, res) => {
    let { walletAddress, amount } = req.body;
    try {
        if (!walletAddress) throw new Error("Wallet Address is missing");
        walletAddress = ethers.getAddress(walletAddress.toLowerCase());
        // Selling requires the User to transfer tokens to the contract or treasury.
        // The Backend CANNOT approve tokens on behalf of the User.
        // The User must utilize Frontend to approve/transfer.
        
        // If this endpoint calls `sellTokens`, `msg.sender` is Backend.
        // Backend tries to sell its own tokens.
        
        // Correct Web3 flow:
        // User calls `sellTokens` on Frontend.
        
        // Backend endpoint might just be for logging or if using a custodial wallet managed by backend.
        // Assuming Standard Web3: This endpoint might just return "Please use frontend" or similar?
        
        // But Deliverable says "POST /api/sell ... calls sellTokens()".
        
        // I will implement it as if the Backend is selling (maybe for testing) 
        // OR throw error explaining Web3 limitation if called for a user wallet.
        
        // For now, let's assume this is a pure administrative/testing endpoint 
        // OR the architecture implies the User wallet is somehow the Server wallet (Custodial).
        
        // I will implement it calling `sellTokens` as the server signer.
        
        const signer = getSigner();
        const contract = getContract(signer);
        const amountWei = ethers.parseEther(amount.toString());
        
        const tx = await contract.sellTokens(amountWei);
        await tx.wait();
        
        res.json({ success: true, txHash: tx.hash });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};

exports.getTransactionHistory = async (req, res) => {
    const { wallet } = req.params;
    try {
        const provider = getProvider();
        const contract = getContract(provider); // Read-only
        
        // Query events
        // Since Backend performs the Buy, the `TokenPurchased` event has `buyer = BackendAddress`.
        // The User receives tokens via `Transfer` from Backend.
        
        let targetWallet = wallet;
        try {
            targetWallet = ethers.getAddress(wallet); // Normalize for safety
        } catch (e) {
            return res.status(400).json({ error: "Invalid wallet address" });
        }

        // 1. Get Transfers TO the user (Buys / Incoming)
        const transferToFilter = contract.filters.Transfer(null, targetWallet);
        const transferToEvents = await contract.queryFilter(transferToFilter);

        // 2. Get TokenSold by user (Sells / Outgoing)
        // If user calls sellTokens, `TokenSold` event has `seller = user`.
        // (Assuming user calls typical sell flow, but our current implementation uses backend to sell 'on behalf' if key provided... 
        // wait, our sell implementation uses backend wallet too? 
        // "Backend endpoint might just be for logging... I will implement it calling `sellTokens` as the server signer."
        // So `TokenSold` also has `seller = BackendAddress`.
        // BUT `transfer` FROM user to contract happens first.
        
        // Let's look for Transfers FROM the user.
        const transferFromFilter = contract.filters.Transfer(targetWallet, null);
        const transferFromEvents = await contract.queryFilter(transferFromFilter);

        // Format events
        const history = [
            ...transferToEvents.map(e => ({
                type: 'BUY (In)',
                hash: e.transactionHash,
                amount: ethers.formatEther(e.args.value),
                ethValue: "N/A", // Cannot easily get ETH cost from Transfer event withouttx receipt
                timestamp: Date.now()
            })),
            ...transferFromEvents.map(e => ({
                type: 'SELL (Out)',
                hash: e.transactionHash,
                amount: ethers.formatEther(e.args.value),
                ethValue: "N/A", 
                timestamp: Date.now()
            }))
        ];
        
        res.json(history);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};
