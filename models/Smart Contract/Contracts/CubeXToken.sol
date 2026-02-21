// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract CubeXToken is ERC20, Ownable {
    // Fixed Price: 1 CUBEX = 0.080321 ETH
    uint256 public constant TOKEN_PRICE_ETH = 80321000000000000;

    event TokenPurchased(address indexed buyer, uint256 amount, uint256 cost);
    event TokenSold(address indexed seller, uint256 amount, uint256 refund);

    constructor() ERC20("CubeX Carbon Token", "CUBEX") Ownable(msg.sender) {
         _mint(address(this), 1000000 * 10 ** decimals());
    }

    // Function to mint tokens (simulating AI pipeline)
    function mint(address to, uint256 amount) external onlyOwner {
        _mint(to, amount);
    }

    
    receive() external payable {}

    
    function buyTokens(uint256 amount) external payable {
        require(amount > 0, "Amount must be greater than 0");
        
        uint256 cost = (amount * TOKEN_PRICE_ETH) / (10 ** decimals()); 
        
        
        require(msg.value >= cost, "Insufficient ETH sent");

        uint256 contractBalance = balanceOf(address(this));
        require(contractBalance >= amount, "Not enough tokens in treasury");

        _transfer(address(this), msg.sender, amount);

       
        if (msg.value > cost) {
            payable(msg.sender).transfer(msg.value - cost);
        }

        emit TokenPurchased(msg.sender, amount, cost);
    }


    function sellTokens(uint256 amount) external {
        require(amount > 0, "Amount must be greater than 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient token balance");

        uint256 refund = (amount * TOKEN_PRICE_ETH) / (10 ** decimals());

        require(address(this).balance >= refund, "Insufficient ETH in treasury");

        _transfer(msg.sender, address(this), amount);
        payable(msg.sender).transfer(refund);

        emit TokenSold(msg.sender, amount, refund);
    }

 
    function withdraw() external onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }
    
    
    function withdrawTokens(uint256 amount) external onlyOwner {
        _transfer(address(this), owner(), amount);
    }

    function getTokenPrice() external pure returns (uint256) {
        return TOKEN_PRICE_ETH;
    }
}
