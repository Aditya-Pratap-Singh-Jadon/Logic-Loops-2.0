// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/v4.9.0/contracts/token/ERC20/ERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/v4.9.0/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/v4.9.0/contracts/security/Pausable.sol";
import "./AggregatorV3Interface.sol";

contract CubexCarbonToken is ERC20, Ownable, Pausable {

    AggregatorV3Interface public priceFeed;

    uint256 public constant TOKEN_PRICE_INR = 20000; // ₹20,000 per token
    uint256 public constant INR_TO_USD = 83; // Fixed for hackathon demo

    address public oracle;

    mapping(address => uint256) public lastCarbonScore;

    event CarbonDataSubmitted(
        address indexed facility,
        uint256 carbonScore,
        bool malicious
    );

    event TokensMinted(
        address indexed facility,
        uint256 amount
    );

    event MaliciousDetected(address indexed facility);

    event TokensPurchased(
        address indexed buyer,
        uint256 ethPaid,
        uint256 tokensReceived
    );

    modifier onlyOracle() {
        require(msg.sender == oracle, "Not authorized oracle");
        _;
    }

    constructor(address _oracle)
        ERC20("Cubex Carbon Token", "CUBEX")
        
    {
        oracle = _oracle;

        // Sepolia ETH/USD Chainlink Price Feed
        priceFeed = AggregatorV3Interface(
            0x694AA1769357215DE4FAC081bf1f309aDC325306
        );
    }

    // ================= BUY TOKENS =================

    function buyTokens(uint256 tokenAmount)
        external
        payable
        whenNotPaused
    {
        require(tokenAmount > 0, "Invalid amount");

        uint256 ethRequired = getRequiredETH(tokenAmount);
        require(msg.value >= ethRequired, "Insufficient ETH sent");

        uint256 mintAmount = tokenAmount * 10 ** decimals();
        _mint(msg.sender, mintAmount);

        emit TokensPurchased(msg.sender, msg.value, mintAmount);
    }

    function getRequiredETH(uint256 tokenAmount)
        public
        view
        returns (uint256)
    {
        (, int price,,,) = priceFeed.latestRoundData();
        require(price > 0, "Invalid price feed");

        uint256 ethPriceUSD = uint256(price); // 8 decimals

        // Convert INR to USD (scaled to 8 decimals)
        uint256 tokenPriceUSD = (TOKEN_PRICE_INR * 1e8) / INR_TO_USD;

        // Calculate required ETH (18 decimals)
        uint256 ethRequired =
            (tokenAmount * tokenPriceUSD * 1e18) / ethPriceUSD;

        return ethRequired;
    }

    // ================= ORACLE MINTING =================

    function submitCarbonData(
        address facility,
        uint256 carbonScore,
        bool malicious
    )
        external
        onlyOracle
        whenNotPaused
    {
        emit CarbonDataSubmitted(facility, carbonScore, malicious);

        if (malicious) {
            lastCarbonScore[facility] = 0;
            emit MaliciousDetected(facility);
            return;
        }

        lastCarbonScore[facility] = carbonScore;

        uint256 mintAmount = carbonScore * 10 ** decimals();
        _mint(facility, mintAmount);

        emit TokensMinted(facility, mintAmount);
    }

    // ================= ADMIN =================

    function setOracle(address _newOracle)
        external
        onlyOwner
    {
        oracle = _newOracle;
    }

    function withdrawETH()
        external
        onlyOwner
    {
        uint256 balance = address(this).balance;

        (bool success, ) = payable(owner()).call{value: balance}("");
        require(success, "Transfer failed");
    }

    function pause()
        external
        onlyOwner
    {
        _pause();
    }

    function unpause()
        external
        onlyOwner
    {
        _unpause();
    }

    receive() external payable {}
}
