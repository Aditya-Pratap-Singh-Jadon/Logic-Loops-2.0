import { ethers } from 'ethers';

// Helper to get contract instance
// Note: You need to export the ABI from the artifacts after compilation.
// For now, we use a minimal ABI or assume it's available.
// Ideally, import CubeXToken from '../../../smart-contract/artifacts/contracts/CubeXToken.sol/CubeXToken.json';

// Placeholder ABI - Replace with actual ABI after compilation
const ABI = [
  "function name() view returns (string)",
  "function symbol() view returns (string)",
  "function balanceOf(address) view returns (uint256)",
  "function transfer(address to, uint256 amount) returns (bool)",
  "function buyTokens(uint256 amount) payable",
  "function sellTokens(uint256 amount)",
  "function getTokenPrice() view returns (uint256)",
  "event TokenPurchased(address indexed buyer, uint256 amount, uint256 cost)",
  "event TokenSold(address indexed seller, uint256 amount, uint256 refund)"
];

// Contract Address - Replace after deployment
export const CONTRACT_ADDRESS = "0x4D2C837A8d38fCFdb5d5C80bff939BFAAad79D7d";

export const getContract = async (providerOrSigner) => {
  return new ethers.Contract(CONTRACT_ADDRESS, ABI, providerOrSigner);
};

export const formatEther = (wei) => ethers.formatEther(wei);
export const parseEther = (eth) => ethers.parseEther(eth);
