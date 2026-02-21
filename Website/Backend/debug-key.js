require('dotenv').config();
const { ethers } = require('ethers');

async function checkKey() {
    let key = process.env.PRIVATE_KEY;
    console.log(`Original Key from env: '${key}'`);
    
    if (!key) {
        console.error("❌ Key is missing!");
        return;
    }

    try {
        // Apply same cleaning logic as server
        key = key.trim().replace(/\s/g, '').replace(/^["']|["']$/g, '');
        if (!key.startsWith("0x")) key = "0x" + key;

        console.log(`Cleaned Key: '${key}'`);
        console.log(`Length: ${key.length}`);

        const wallet = new ethers.Wallet(key);
        console.log("\n✅ SUCCESS! Private Key is valid.");
        console.log(`Wallet Address: ${wallet.address}`);
    } catch (error) {
        console.error("\n❌ ERROR: Key is invalid.");
        console.error(error.message);
    }
}

checkKey();
