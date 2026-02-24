import DLMM, { StrategyType } from "@meteora-ag/dlmm";
import bs58 from "bs58";
import BN from "bn.js";
import {
  Connection,
  Keypair,
  PublicKey,
  Transaction,
  VersionedTransaction,
  sendAndConfirmTransaction
} from "@solana/web3.js";

async function main() {
  const input = await readStdinJson();
  if (!input || input.command !== "apply-range") {
    throw new Error("Expected JSON stdin payload with command=apply-range");
  }

  const { rpc_url, private_key_b58, pool, target_range_sol_usdc, deposit, existing_position } = input;
  if (!rpc_url || !private_key_b58) throw new Error("rpc_url and private_key_b58 are required");
  if (!pool?.address) throw new Error("pool.address is required");
  if (!target_range_sol_usdc?.lower || !target_range_sol_usdc?.upper) {
    throw new Error("target_range_sol_usdc.lower and .upper are required");
  }

  const connection = new Connection(rpc_url, "confirmed");
  const secret = bs58.decode(private_key_b58);
  const wallet = Keypair.fromSecretKey(Uint8Array.from(secret));
  const dlmmPool = await DLMM.create(connection, new PublicKey(pool.address));

  const targetPoolOrientation = synthRangeToPoolOrientation(pool, target_range_sol_usdc.lower, target_range_sol_usdc.upper);

  const activeBin = await dlmmPool.getActiveBin();
  const activeBinPrice = numberOrNull(activeBin?.pricePerToken);
  const apiCurrentPrice = numberOrNull(pool.api_current_price);

  const { lowerForSdk, upperForSdk, sdkPriceOrientation } = mapPoolPriceToSdkOrientation({
    lowerPool: targetPoolOrientation.lower,
    upperPool: targetPoolOrientation.upper,
    activeBinPrice,
    apiCurrentPrice
  });

  const bins = await dlmmPool.getBinsBetweenMinAndMaxPrice(lowerForSdk, upperForSdk);
  if (!Array.isArray(bins) || bins.length === 0) {
    throw new Error("No bins returned for target price range; cannot initialize position");
  }

  const lowerBinId = extractBinId(bins[0]);
  const upperBinId = extractBinId(bins[bins.length - 1]);

  const txSignatures = [];

  if (existing_position?.position_pubkey && existing_position.lower_bin_id != null && existing_position.upper_bin_id != null) {
    const removeTxs = await dlmmPool.removeLiquidity({
      position: new PublicKey(existing_position.position_pubkey),
      fromBinId: Number(existing_position.lower_bin_id),
      toBinId: Number(existing_position.upper_bin_id),
      bps: new BN(10000),
      user: wallet.publicKey,
      shouldClaimAndClose: true
    });
    txSignatures.push(
      ...(await sendAllTransactions(connection, removeTxs, [wallet]))
    );
  }

  const { totalXAmount, totalYAmount } = mapDepositsToPoolSides(pool, deposit);
  const newPosition = Keypair.generate();
  const createTx = await dlmmPool.initializePositionAndAddLiquidityByStrategy({
    positionPubKey: newPosition.publicKey,
    user: wallet.publicKey,
    totalXAmount,
    totalYAmount,
    strategy: {
      minBinId: lowerBinId,
      maxBinId: upperBinId,
      strategyType: StrategyType.SpotBalanced
    }
  });
  txSignatures.push(
    ...(await sendAllTransactions(connection, createTx, [wallet, newPosition]))
  );

  const widthPct = Number(
    ((target_range_sol_usdc.upper - target_range_sol_usdc.lower) /
      ((target_range_sol_usdc.upper + target_range_sol_usdc.lower) / 2)) *
      100
  );

  writeJson({
    ok: true,
    changed: true,
    tx_signatures: txSignatures,
    active_position: {
      pool_address: pool.address,
      position_pubkey: newPosition.publicKey.toBase58(),
      lower_bin_id: lowerBinId,
      upper_bin_id: upperBinId,
      lower_price: Number(target_range_sol_usdc.lower),
      upper_price: Number(target_range_sol_usdc.upper),
      width_pct: widthPct,
      sdk_price_orientation,
      pool_price_orientation: targetPoolOrientation.orientation
    }
  });
}

function synthRangeToPoolOrientation(pool, lowerSolUsdc, upperSolUsdc) {
  const sx = String(pool.symbol_x || "").toUpperCase();
  const sy = String(pool.symbol_y || "").toUpperCase();
  if (sx === "SOL" && sy === "USDC") {
    return {
      lower: Number(lowerSolUsdc),
      upper: Number(upperSolUsdc),
      orientation: "USDC_per_SOL"
    };
  }
  if (sx === "USDC" && sy === "SOL") {
    return {
      lower: 1 / Number(upperSolUsdc),
      upper: 1 / Number(lowerSolUsdc),
      orientation: "SOL_per_USDC"
    };
  }
  throw new Error(`Unsupported pool symbols for MVP: ${sx}/${sy}. Expected SOL/USDC or USDC/SOL`);
}

function mapPoolPriceToSdkOrientation({ lowerPool, upperPool, activeBinPrice, apiCurrentPrice }) {
  if (!isFinite(activeBinPrice) || !isFinite(apiCurrentPrice) || apiCurrentPrice <= 0) {
    return {
      lowerForSdk: lowerPool,
      upperForSdk: upperPool,
      sdkPriceOrientation: "unknown_assume_api_orientation"
    };
  }
  const sameErr = relativeError(activeBinPrice, apiCurrentPrice);
  const invErr = relativeError(activeBinPrice, 1 / apiCurrentPrice);
  if (invErr < sameErr) {
    return {
      lowerForSdk: 1 / upperPool,
      upperForSdk: 1 / lowerPool,
      sdkPriceOrientation: "inverted_vs_api"
    };
  }
  return {
    lowerForSdk: lowerPool,
    upperForSdk: upperPool,
    sdkPriceOrientation: "same_as_api"
  };
}

function mapDepositsToPoolSides(pool, deposit) {
  const sx = String(pool.symbol_x || "").toUpperCase();
  const sy = String(pool.symbol_y || "").toUpperCase();
  let xUi;
  let yUi;

  if (sx === "SOL" && sy === "USDC") {
    xUi = Number(deposit.sol);
    yUi = Number(deposit.usdc);
  } else if (sx === "USDC" && sy === "SOL") {
    xUi = Number(deposit.usdc);
    yUi = Number(deposit.sol);
  } else {
    throw new Error(`Unsupported pool symbols for deposit mapping: ${sx}/${sy}`);
  }

  return {
    totalXAmount: uiAmountToBn(xUi, Number(pool.decimals_x)),
    totalYAmount: uiAmountToBn(yUi, Number(pool.decimals_y))
  };
}

function uiAmountToBn(value, decimals) {
  if (!isFinite(value) || value < 0) throw new Error(`Invalid UI amount: ${value}`);
  if (!Number.isInteger(decimals) || decimals < 0) throw new Error(`Invalid decimals: ${decimals}`);

  const [wholePart, fracPart = ""] = String(value).split(".");
  const safeFrac = fracPart.slice(0, decimals).padEnd(decimals, "0");
  const normalized = `${wholePart}${safeFrac}`.replace(/^(-?)0+(?=\d)/, "$1");
  return new BN(normalized || "0");
}

function extractBinId(bin) {
  const value = bin?.binId ?? bin?.bin_id;
  if (value == null) throw new Error(`Could not read binId from SDK response: ${JSON.stringify(bin)}`);
  return Number(value);
}

async function sendAllTransactions(connection, txOrTxs, signers) {
  const txs = Array.isArray(txOrTxs) ? txOrTxs : [txOrTxs];
  const signatures = [];
  for (const tx of txs) {
    signatures.push(await sendSingleTransaction(connection, tx, signers));
  }
  return signatures;
}

async function sendSingleTransaction(connection, tx, signers) {
  if (tx instanceof VersionedTransaction) {
    tx.sign(signers);
    const sig = await connection.sendTransaction(tx, { skipPreflight: false, maxRetries: 3 });
    await connection.confirmTransaction(sig, "confirmed");
    return sig;
  }
  if (tx instanceof Transaction || typeof tx?.partialSign === "function") {
    return sendAndConfirmTransaction(connection, tx, signers, {
      commitment: "confirmed"
    });
  }
  throw new Error(`Unsupported transaction type from SDK: ${tx?.constructor?.name || typeof tx}`);
}

function relativeError(a, b) {
  const denom = Math.max(Math.abs(b), 1e-12);
  return Math.abs(a - b) / denom;
}

function numberOrNull(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

async function readStdinJson() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const raw = Buffer.concat(chunks).toString("utf8").trim();
  if (!raw) return null;
  return JSON.parse(raw);
}

function writeJson(obj) {
  process.stdout.write(JSON.stringify(obj));
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(1);
});

