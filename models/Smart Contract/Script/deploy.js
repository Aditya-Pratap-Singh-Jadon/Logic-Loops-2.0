const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();

  console.log("Deploying contracts with the account:", deployer.address);

  const CubeXToken = await hre.ethers.getContractFactory("CubeXToken");
  const token = await CubeXToken.deploy();

  await token.waitForDeployment();

  console.log("CubeXToken deployed to:", await token.getAddress());
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
