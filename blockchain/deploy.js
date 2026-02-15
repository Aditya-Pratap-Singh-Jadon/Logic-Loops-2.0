const hre = require("hardhat");

async function main() {

  const oracleAddress = "YOUR_WALLET_ADDRESS";

  const Contract = await hre.ethers.getContractFactory("CubexCarbonToken");
  const contract = await Contract.deploy(oracleAddress);

  await contract.waitForDeployment();

  console.log("Deployed at:", await contract.getAddress());
}

main();
