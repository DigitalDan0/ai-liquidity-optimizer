import { createRequire } from "node:module";
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

const require = createRequire(import.meta.url);
const dlmmModule = require("@meteora-ag/dlmm");
const DLMM = dlmmModule.default || dlmmModule;
const { StrategyType } = dlmmModule;

const RPC_FETCH_MAX_ATTEMPTS = parseEnvInt("RPC_FETCH_MAX_ATTEMPTS", 12, 1, 50);
const RPC_FETCH_BASE_DELAY_MS = parseEnvInt("RPC_FETCH_BASE_DELAY_MS", 500, 50, 30000);
const RPC_FETCH_MAX_DELAY_MS = parseEnvInt("RPC_FETCH_MAX_DELAY_MS", 15000, 100, 120000);
const RPC_OPERATION_MAX_ATTEMPTS = parseEnvInt("RPC_OPERATION_MAX_ATTEMPTS", 5, 1, 20);
const RPC_OPERATION_BASE_DELAY_MS = parseEnvInt("RPC_OPERATION_BASE_DELAY_MS", 600, 50, 30000);
const RPC_OPERATION_MAX_DELAY_MS = parseEnvInt("RPC_OPERATION_MAX_DELAY_MS", 10000, 100, 120000);

function createRpcConnection(rpcUrl) {
  const retryingFetch = makeRpcRetryingFetch({
    maxAttempts: RPC_FETCH_MAX_ATTEMPTS,
    baseDelayMs: RPC_FETCH_BASE_DELAY_MS,
    maxDelayMs: RPC_FETCH_MAX_DELAY_MS
  });
  return new Connection(rpcUrl, {
    commitment: "confirmed",
    disableRetryOnRateLimit: true,
    fetch: retryingFetch
  });
}

async function main() {
  const input = await readStdinJson();
  if (!input || !input.command) {
    throw new Error("Expected JSON stdin payload with a command");
  }
  if (input.command === "check-position") {
    await handleCheckPosition(input);
    return;
  }
  if (input.command === "quote-range-bins") {
    await handleQuoteRangeBins(input);
    return;
  }
  if (input.command === "close-position") {
    await handleClosePosition(input);
    return;
  }
  if (input.command === "wallet-snapshot") {
    await handleWalletSnapshot(input);
    return;
  }
  if (input.command !== "apply-range") {
    throw new Error(`Unsupported command: ${input.command}`);
  }

  const { rpc_url, private_key_b58, pool, target_range_sol_usdc, deposit, existing_position } = input;
  if (!rpc_url || !private_key_b58) throw new Error("rpc_url and private_key_b58 are required");
  if (!pool?.address) throw new Error("pool.address is required");
  if (!target_range_sol_usdc?.lower || !target_range_sol_usdc?.upper) {
    throw new Error("target_range_sol_usdc.lower and .upper are required");
  }
  const liquidityMode = String(input.liquidity_mode || "spot").toLowerCase();
  const maxCustomWeightPositionBins = Number.isFinite(Number(input.max_custom_weight_position_bins))
    ? Math.max(1, Number(input.max_custom_weight_position_bins))
    : 70;
  const synthWeightActiveBinFloorBps = clampInt(Number(input.synth_weight_active_bin_floor_bps ?? 1000), 0, 10000);
  const synthWeightMaxBinBpsPerSide = clampInt(Number(input.synth_weight_max_bin_bps_per_side ?? 7000), 0, 10000);

  const connection = createRpcConnection(rpc_url);
  const secret = bs58.decode(private_key_b58);
  const wallet = Keypair.fromSecretKey(Uint8Array.from(secret));
  const dlmmPool = await DLMM.create(connection, new PublicKey(pool.address));
  const rangeQuote = await quoteRangeBinsForSdk({
    dlmmPool,
    pool,
    targetRangeSolUsdc: target_range_sol_usdc
  });
  const {
    targetPoolOrientation,
    activeBin,
    lowerForSdk,
    upperForSdk,
    sdkPriceOrientation,
    bins
  } = rangeQuote;
  const lowerBinId = extractBinId(bins[0]);
  const upperBinId = extractBinId(bins[bins.length - 1]);

  const txSignatures = [];
  const { totalXAmount, totalYAmount } = mapDepositsToPoolSides(pool, deposit);

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

  const newPosition = Keypair.generate();
  const createBuild = await withRpcOperationRetries(
    async () => {
      if (liquidityMode === "synth_weights") {
        const activeBinId = extractBinId(activeBin);
        if (bins.length > maxCustomWeightPositionBins) {
          throw new Error(
            `Custom-weight position spans ${bins.length} bins, exceeding MAX_CUSTOM_WEIGHT_POSITION_BINS=${maxCustomWeightPositionBins}`
          );
        }
        const synthWeights = Array.isArray(input.target_bin_weights) ? input.target_bin_weights : null;
        const synthBinEdges = Array.isArray(input.target_bin_edges) ? input.target_bin_edges : null;
        const distributionBuild = buildXYAmountDistributionFromSynth({
          bins,
          targetBinIds: Array.isArray(input.target_bin_ids) ? input.target_bin_ids : null,
          targetBinWeights: synthWeights,
          targetBinEdges: synthBinEdges,
          pool,
          sdkPriceOrientation,
          lowerForSdk,
          upperForSdk,
          activeBinId,
          activeBinFloorBps: synthWeightActiveBinFloorBps,
          maxBinBpsPerSide: synthWeightMaxBinBpsPerSide,
          totalXAmount,
          totalYAmount
        });
        const createTx = await dlmmPool.initializePositionAndAddLiquidityByWeight({
          positionPubKey: newPosition.publicKey,
          user: wallet.publicKey,
          totalXAmount,
          totalYAmount,
          xYAmountDistribution: distributionBuild.xYAmountDistribution
        });
        return {
          createTx,
          executionMeta: {
            liquidity_mode: "synth_weights",
            sdk_method_used: "initializePositionAndAddLiquidityByWeight",
            custom_weight_num_bins: bins.length,
            active_bin_id: activeBinId,
            sdk_tx_count: Array.isArray(createTx) ? createTx.length : 1,
            custom_weight_validation: distributionBuild.validation,
            custom_weight_distribution_preview: distributionBuild.preview
          }
        };
      }
      if (liquidityMode === "spot") {
        const strategyParams = buildStrategyParams({
          lowerBinId,
          upperBinId,
          totalXAmount,
          totalYAmount
        });
        const createTx = await dlmmPool.initializePositionAndAddLiquidityByStrategy({
          positionPubKey: newPosition.publicKey,
          user: wallet.publicKey,
          totalXAmount,
          totalYAmount,
          strategy: strategyParams
        });
        return {
          createTx,
          executionMeta: {
            liquidity_mode: "spot",
            sdk_method_used: "initializePositionAndAddLiquidityByStrategy",
            sdk_tx_count: Array.isArray(createTx) ? createTx.length : 1
          }
        };
      }
      throw new Error(`Unsupported liquidity_mode: ${liquidityMode}`);
    },
    { label: `build-open-${liquidityMode}` }
  );
  const createTx = createBuild.value.createTx;
  const executionMeta = {
    ...createBuild.value.executionMeta,
    rpc_create_attempts: createBuild.attempts
  };
  const openSend = await withRpcOperationRetries(
    async () => sendAllTransactions(connection, createTx, [wallet, newPosition]),
    { label: "send-open-transactions" }
  );
  txSignatures.push(...openSend.value);
  executionMeta.rpc_send_attempts = openSend.attempts;

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
      sdk_price_orientation: sdkPriceOrientation,
      pool_price_orientation: targetPoolOrientation.orientation,
      liquidity_mode: executionMeta.liquidity_mode,
      sdk_method_used: executionMeta.sdk_method_used,
      custom_weight_num_bins: executionMeta.custom_weight_num_bins ?? null,
      custom_weight_validation: executionMeta.custom_weight_validation ?? null
    },
    execution: executionMeta
  });
}

async function handleCheckPosition(input) {
  const { rpc_url, position_pubkey } = input;
  if (!rpc_url) throw new Error("rpc_url is required");
  if (!position_pubkey) throw new Error("position_pubkey is required");

  const connection = createRpcConnection(rpc_url);
  const info = await connection.getAccountInfo(new PublicKey(position_pubkey), "confirmed");
  writeJson({
    ok: true,
    exists: Boolean(info),
    position_pubkey,
    owner: info?.owner?.toBase58?.() || null,
    lamports: typeof info?.lamports === "number" ? info.lamports : null
  });
}

async function handleClosePosition(input) {
  const { rpc_url, private_key_b58, existing_position } = input;
  if (!rpc_url || !private_key_b58) throw new Error("rpc_url and private_key_b58 are required");
  if (!existing_position?.pool_address) throw new Error("existing_position.pool_address is required");
  if (!existing_position?.position_pubkey) throw new Error("existing_position.position_pubkey is required");
  if (existing_position.lower_bin_id == null || existing_position.upper_bin_id == null) {
    throw new Error("existing_position.lower_bin_id and existing_position.upper_bin_id are required");
  }

  const connection = createRpcConnection(rpc_url);
  const secret = bs58.decode(private_key_b58);
  const wallet = Keypair.fromSecretKey(Uint8Array.from(secret));
  const dlmmPool = await DLMM.create(connection, new PublicKey(existing_position.pool_address));
  const removeTxs = await dlmmPool.removeLiquidity({
    position: new PublicKey(existing_position.position_pubkey),
    fromBinId: Number(existing_position.lower_bin_id),
    toBinId: Number(existing_position.upper_bin_id),
    bps: new BN(10000),
    user: wallet.publicKey,
    shouldClaimAndClose: true
  });
  const txSignatures = await sendAllTransactions(connection, removeTxs, [wallet]);
  writeJson({
    ok: true,
    changed: true,
    tx_signatures: txSignatures,
    active_position: null,
    execution: {
      action: "close-position",
      sdk_method_used: "removeLiquidity",
      sdk_tx_count: Array.isArray(removeTxs) ? removeTxs.length : 1
    }
  });
}

async function handleWalletSnapshot(input) {
  const { rpc_url, private_key_b58, pool, active_position } = input;
  if (!rpc_url || !private_key_b58) throw new Error("rpc_url and private_key_b58 are required");

  const connection = createRpcConnection(rpc_url);
  const secret = bs58.decode(private_key_b58);
  const wallet = Keypair.fromSecretKey(Uint8Array.from(secret));

  const walletPubkey = wallet.publicKey;
  const solLamports = await connection.getBalance(walletPubkey, "confirmed");
  const nativeSolBalance = Number(solLamports) / 1e9;

  const resolvedPoolAddress = pool?.address || active_position?.pool_address || null;
  let dlmmPool = null;
  if (resolvedPoolAddress) {
    dlmmPool = await DLMM.create(connection, new PublicKey(resolvedPoolAddress));
  }

  const symbolX = String(pool?.symbol_x || "").toUpperCase();
  const symbolY = String(pool?.symbol_y || "").toUpperCase();
  const mintX = pool?.mint_x || publicKeyLikeToString(dlmmPool?.tokenX?.mint);
  const mintY = pool?.mint_y || publicKeyLikeToString(dlmmPool?.tokenY?.mint);
  const decimalsX = Number.isFinite(Number(pool?.decimals_x))
    ? Number(pool.decimals_x)
    : Number(numberOrNull(dlmmPool?.tokenX?.decimals) ?? 0);
  const decimalsY = Number.isFinite(Number(pool?.decimals_y))
    ? Number(pool.decimals_y)
    : Number(numberOrNull(dlmmPool?.tokenY?.decimals) ?? 0);

  let spotSolUsdc = null;
  if (pool?.api_current_price != null && symbolX && symbolY) {
    spotSolUsdc = poolPriceToSynthSolUsdcPrice(
      { symbol_x: symbolX, symbol_y: symbolY },
      numberOrNull(pool.api_current_price)
    );
  }

  let tokenX = null;
  let tokenY = null;
  if (mintX && mintY) {
    tokenX = await getTokenUiBalanceByMint(connection, walletPubkey, String(mintX), decimalsX);
    tokenY = await getTokenUiBalanceByMint(connection, walletPubkey, String(mintY), decimalsY);
  }

  let poolUserPositions = [];
  let poolUserPositionsError = null;
  if (dlmmPool) {
    try {
      const byUserAndPair = await dlmmPool.getPositionsByUserAndLbPair(wallet.publicKey);
      const rawUserPositions = Array.isArray(byUserAndPair?.userPositions) ? byUserAndPair.userPositions : [];
      poolUserPositions = summarizePoolUserPositions(rawUserPositions, {
        symbolX,
        symbolY,
        decimalsX,
        decimalsY,
        spotSolUsdc
      });
    } catch (error) {
      poolUserPositionsError = String(error?.message || error);
    }
  }

  let walletSolTokenUi = null;
  let walletUsdcTokenUi = null;
  if (tokenX && tokenY) {
    if (symbolX === "SOL" && symbolY === "USDC") {
      walletSolTokenUi = tokenX.uiAmount;
      walletUsdcTokenUi = tokenY.uiAmount;
    } else if (symbolX === "USDC" && symbolY === "SOL") {
      walletSolTokenUi = tokenY.uiAmount;
      walletUsdcTokenUi = tokenX.uiAmount;
    }
  }
  const walletSolTotalUi = nativeSolBalance + (walletSolTokenUi || 0);
  const walletTotalUsdEst =
    spotSolUsdc != null && walletUsdcTokenUi != null
      ? walletUsdcTokenUi + walletSolTotalUi * spotSolUsdc
      : null;

  let positionSnapshot = null;
  let activePositionExists = null;
  if (active_position?.position_pubkey) {
    const info = await connection.getAccountInfo(new PublicKey(active_position.position_pubkey), "confirmed");
    activePositionExists = Boolean(info);
    if (info && dlmmPool) {
      try {
        const pos = await dlmmPool.getPosition(new PublicKey(active_position.position_pubkey));
        const data = pos?.positionData || null;
        const totalXRaw = bnLikeToBigInt(data?.totalXAmountExcludeTransferFee ?? data?.totalXAmount);
        const totalYRaw = bnLikeToBigInt(data?.totalYAmountExcludeTransferFee ?? data?.totalYAmount);
        const feeXRaw = bnLikeToBigInt(data?.feeXExcludeTransferFee ?? data?.feeX);
        const feeYRaw = bnLikeToBigInt(data?.feeYExcludeTransferFee ?? data?.feeY);
        const totalXUi = totalXRaw != null ? bigIntToUi(totalXRaw, decimalsX) : null;
        const totalYUi = totalYRaw != null ? bigIntToUi(totalYRaw, decimalsY) : null;
        const feeXUi = feeXRaw != null ? bigIntToUi(feeXRaw, decimalsX) : null;
        const feeYUi = feeYRaw != null ? bigIntToUi(feeYRaw, decimalsY) : null;

        let principalSolUi = null;
        let principalUsdcUi = null;
        let feeSolUi = null;
        let feeUsdcUi = null;
        if (symbolX === "SOL" && symbolY === "USDC") {
          principalSolUi = totalXUi;
          principalUsdcUi = totalYUi;
          feeSolUi = feeXUi;
          feeUsdcUi = feeYUi;
        } else if (symbolX === "USDC" && symbolY === "SOL") {
          principalSolUi = totalYUi;
          principalUsdcUi = totalXUi;
          feeSolUi = feeYUi;
          feeUsdcUi = feeXUi;
        }
        const positionSolUi = (principalSolUi || 0) + (feeSolUi || 0);
        const positionUsdcUi = (principalUsdcUi || 0) + (feeUsdcUi || 0);
        const positionTotalUsdEst =
          spotSolUsdc != null && principalUsdcUi != null
            ? positionUsdcUi + positionSolUi * spotSolUsdc
            : null;

        positionSnapshot = {
          position_pubkey: active_position.position_pubkey,
          lower_bin_id: data?.lowerBinId ?? active_position.lower_bin_id ?? null,
          upper_bin_id: data?.upperBinId ?? active_position.upper_bin_id ?? null,
          total_x_ui: totalXUi,
          total_y_ui: totalYUi,
          fee_x_ui: feeXUi,
          fee_y_ui: feeYUi,
          principal_sol_ui: principalSolUi,
          principal_usdc_ui: principalUsdcUi,
          fee_sol_ui: feeSolUi,
          fee_usdc_ui: feeUsdcUi,
          total_sol_ui: positionSolUi,
          total_usdc_ui: positionUsdcUi,
          total_usd_est: positionTotalUsdEst
        };
      } catch (error) {
        positionSnapshot = {
          position_pubkey: active_position.position_pubkey,
          error: String(error?.message || error)
        };
      }
    }
  }

  const trackedPositionPubkey = active_position?.position_pubkey ? String(active_position.position_pubkey) : null;
  const poolUserPositionPubkeys = poolUserPositions
    .map((position) => String(position?.position_pubkey || ""))
    .filter((value) => value.length > 0);
  const trackedPositionDetected =
    trackedPositionPubkey != null
      ? poolUserPositionPubkeys.includes(trackedPositionPubkey)
      : null;
  const poolOtherPositions =
    trackedPositionPubkey == null
      ? poolUserPositions
      : poolUserPositions.filter((position) => String(position?.position_pubkey || "") !== trackedPositionPubkey);
  const poolOtherPositionPubkeys = poolOtherPositions
    .map((position) => String(position?.position_pubkey || ""))
    .filter((value) => value.length > 0);

  const totalUsdEst =
    walletTotalUsdEst != null || numberOrNull(positionSnapshot?.total_usd_est) != null
      ? (walletTotalUsdEst || 0) + (numberOrNull(positionSnapshot?.total_usd_est) || 0)
      : null;

  const poolBalances = {
    symbol_x: symbolX || pool?.symbol_x || null,
    symbol_y: symbolY || pool?.symbol_y || null,
    mint_x: mintX || null,
    mint_y: mintY || null,
    decimals_x: decimalsX,
    decimals_y: decimalsY,
    token_x_ui: tokenX?.uiAmount ?? null,
    token_y_ui: tokenY?.uiAmount ?? null,
    wallet_native_sol_ui: nativeSolBalance,
    wallet_sol_token_ui: walletSolTokenUi,
    wallet_sol_total_ui: walletSolTotalUi,
    wallet_usdc_ui: walletUsdcTokenUi,
    wallet_total_usd_est: walletTotalUsdEst,
    spot_sol_usdc: spotSolUsdc
  };

  writeJson({
    ok: true,
    snapshot_at: new Date().toISOString(),
    wallet_pubkey: walletPubkey.toBase58(),
    sol_lamports: solLamports,
    sol_balance: walletSolTotalUi,
    usdc_balance: walletUsdcTokenUi,
    native_sol_balance: nativeSolBalance,
    wallet_sol_token_balance: walletSolTokenUi,
    wallet_sol_total_balance: walletSolTotalUi,
    wallet_usdc_total_balance: walletUsdcTokenUi,
    wallet_total_usd_est: walletTotalUsdEst,
    position_total_usd_est: numberOrNull(positionSnapshot?.total_usd_est),
    position_snapshot: positionSnapshot,
    spot_price_sol_usdc: spotSolUsdc,
    total_usd_est: totalUsdEst,
    pool_balances: poolBalances,
    active_position_exists: activePositionExists,
    pool_user_positions: poolUserPositions,
    pool_user_position_count: poolUserPositions.length,
    pool_user_position_pubkeys: poolUserPositionPubkeys,
    pool_other_position_count: poolOtherPositions.length,
    pool_other_position_pubkeys: poolOtherPositionPubkeys,
    tracked_position_pubkey: trackedPositionPubkey,
    tracked_position_detected: trackedPositionDetected,
    pool_user_positions_error: poolUserPositionsError
  });
}

async function getTokenUiBalanceByMint(connection, ownerPubkey, mintAddress, decimalsHint) {
  const mint = new PublicKey(mintAddress);
  const resp = await connection.getParsedTokenAccountsByOwner(ownerPubkey, { mint }, "confirmed");
  let totalRaw = 0n;
  let decimals = Number.isInteger(decimalsHint) && decimalsHint >= 0 ? decimalsHint : 0;
  for (const row of resp?.value || []) {
    const tokenAmount = row?.account?.data?.parsed?.info?.tokenAmount;
    if (!tokenAmount) continue;
    const raw = tokenAmount.amount;
    const rowDecimals = Number(tokenAmount.decimals);
    if (Number.isFinite(rowDecimals) && rowDecimals >= 0) {
      decimals = rowDecimals;
    }
    try {
      totalRaw += BigInt(raw);
    } catch {
      continue;
    }
  }
  return {
    mint: mintAddress,
    amountRaw: totalRaw.toString(),
    decimals,
    uiAmount: bigIntToUi(totalRaw, decimals)
  };
}

function bigIntToUi(amountRaw, decimals) {
  const d = Math.max(0, Number(decimals) || 0);
  const scale = 10n ** BigInt(d);
  const whole = amountRaw / scale;
  const frac = amountRaw % scale;
  return Number(whole) + Number(frac) / Number(scale);
}

function bnLikeToBigInt(value) {
  if (value == null) return null;
  try {
    if (typeof value === "bigint") return value;
    if (typeof value === "number") return BigInt(Math.trunc(value));
    if (typeof value === "string") return BigInt(value);
    if (typeof value.toString === "function") return BigInt(value.toString());
  } catch {
    return null;
  }
  return null;
}

function publicKeyLikeToString(value) {
  if (!value) return null;
  try {
    if (typeof value === "string") return value;
    if (typeof value.toBase58 === "function") return value.toBase58();
    if (typeof value.toString === "function") return value.toString();
  } catch {
    return null;
  }
  return null;
}

function summarizePoolUserPositions(rawUserPositions, { symbolX, symbolY, decimalsX, decimalsY, spotSolUsdc }) {
  if (!Array.isArray(rawUserPositions) || rawUserPositions.length === 0) return [];
  const out = [];
  for (const row of rawUserPositions) {
    if (!row || typeof row !== "object") continue;
    const positionPubkey = publicKeyLikeToString(row.publicKey);
    const positionData = row.positionData;
    if (!positionData || typeof positionData !== "object") continue;

    const totalXRaw = bnLikeToBigInt(positionData.totalXAmountExcludeTransferFee ?? positionData.totalXAmount) ?? 0n;
    const totalYRaw = bnLikeToBigInt(positionData.totalYAmountExcludeTransferFee ?? positionData.totalYAmount) ?? 0n;
    const feeXRaw = bnLikeToBigInt(positionData.feeXExcludeTransferFee ?? positionData.feeX) ?? 0n;
    const feeYRaw = bnLikeToBigInt(positionData.feeYExcludeTransferFee ?? positionData.feeY) ?? 0n;
    if (totalXRaw <= 0n && totalYRaw <= 0n && feeXRaw <= 0n && feeYRaw <= 0n) {
      continue;
    }

    const totalXUi = bigIntToUi(totalXRaw, decimalsX);
    const totalYUi = bigIntToUi(totalYRaw, decimalsY);
    const feeXUi = bigIntToUi(feeXRaw, decimalsX);
    const feeYUi = bigIntToUi(feeYRaw, decimalsY);

    let totalSolUi = null;
    let totalUsdcUi = null;
    if (symbolX === "SOL" && symbolY === "USDC") {
      totalSolUi = totalXUi + feeXUi;
      totalUsdcUi = totalYUi + feeYUi;
    } else if (symbolX === "USDC" && symbolY === "SOL") {
      totalSolUi = totalYUi + feeYUi;
      totalUsdcUi = totalXUi + feeXUi;
    }
    const totalUsdEst =
      spotSolUsdc != null && totalSolUi != null && totalUsdcUi != null
        ? totalUsdcUi + totalSolUi * spotSolUsdc
        : null;

    out.push({
      position_pubkey: positionPubkey,
      lower_bin_id: Number.isFinite(Number(positionData.lowerBinId)) ? Number(positionData.lowerBinId) : null,
      upper_bin_id: Number.isFinite(Number(positionData.upperBinId)) ? Number(positionData.upperBinId) : null,
      total_x_ui: totalXUi,
      total_y_ui: totalYUi,
      fee_x_ui: feeXUi,
      fee_y_ui: feeYUi,
      total_sol_ui: totalSolUi,
      total_usdc_ui: totalUsdcUi,
      total_usd_est: totalUsdEst
    });
  }
  return out;
}

async function handleQuoteRangeBins(input) {
  const { rpc_url, pool, target_range_sol_usdc } = input;
  if (!rpc_url) throw new Error("rpc_url is required");
  if (!pool?.address) throw new Error("pool.address is required");
  if (!target_range_sol_usdc?.lower || !target_range_sol_usdc?.upper) {
    throw new Error("target_range_sol_usdc.lower and .upper are required");
  }
  const connection = createRpcConnection(rpc_url);
  const dlmmPool = await DLMM.create(connection, new PublicKey(pool.address));
  const quote = await quoteRangeBinsForSdk({
    dlmmPool,
    pool,
    targetRangeSolUsdc: target_range_sol_usdc
  });

  const bins = quote.bins.map((bin) => {
    const binId = extractBinId(bin);
    const rawPricePerToken = numberOrNull(bin?.pricePerToken);
    const priceSdk = rawPricePerToken == null ? null : numberOrNull(dlmmPool.fromPricePerLamport(rawPricePerToken));
    const pricePool = sdkPriceToPoolPrice(priceSdk, quote.sdkPriceOrientation);
    const priceSolUsdc = poolPriceToSynthSolUsdcPrice(pool, pricePool);
    return {
      bin_id: binId,
      price_per_token_raw: rawPricePerToken,
      price_sdk: priceSdk,
      price_pool_orientation: pricePool,
      price_sol_usdc: priceSolUsdc
    };
  });

  writeJson({
    ok: true,
    pool_address: pool.address,
    active_bin_id: extractBinId(quote.activeBin),
    sdk_price_orientation: quote.sdkPriceOrientation,
    pool_price_orientation: quote.targetPoolOrientation.orientation,
    lower_for_sdk: quote.lowerForSdk,
    upper_for_sdk: quote.upperForSdk,
    bins
  });
}

async function quoteRangeBinsForSdk({ dlmmPool, pool, targetRangeSolUsdc }) {
  const targetPoolOrientation = synthRangeToPoolOrientation(pool, targetRangeSolUsdc.lower, targetRangeSolUsdc.upper);

  const activeBin = await dlmmPool.getActiveBin();
  const activeBinRawPricePerToken = numberOrNull(activeBin?.pricePerToken);
  const activeBinPrice = activeBinRawPricePerToken == null ? null : numberOrNull(dlmmPool.fromPricePerLamport(activeBinRawPricePerToken));
  const apiCurrentPrice = numberOrNull(pool.api_current_price);

  const { lowerForSdk, upperForSdk, sdkPriceOrientation } = mapPoolPriceToSdkOrientation({
    lowerPool: targetPoolOrientation.lower,
    upperPool: targetPoolOrientation.upper,
    activeBinPrice,
    apiCurrentPrice
  });

  // Meteora SDK bin-ID lookup expects the protocol price scale ("price per lamport"),
  // not the human "token per token" display price.
  const lowerForSdkPriceScale = Number(dlmmPool.toPricePerLamport(lowerForSdk));
  const upperForSdkPriceScale = Number(dlmmPool.toPricePerLamport(upperForSdk));

  const binsResp = await dlmmPool.getBinsBetweenMinAndMaxPrice(lowerForSdkPriceScale, upperForSdkPriceScale);
  const bins = extractBinsFromSdkResponse(binsResp);
  if (!Array.isArray(bins) || bins.length === 0) {
    throw new Error(
      [
        "No bins returned for target price range; cannot initialize position",
        `lowerForSdk=${lowerForSdk}`,
        `upperForSdk=${upperForSdk}`,
        `lowerForSdkPriceScale=${lowerForSdkPriceScale}`,
        `upperForSdkPriceScale=${upperForSdkPriceScale}`,
        `sdkPriceOrientation=${sdkPriceOrientation}`,
        `activeBinPrice=${activeBinPrice}`,
        `activeBinRawPricePerToken=${activeBinRawPricePerToken}`,
        `apiCurrentPrice=${apiCurrentPrice}`,
        `binsRespType=${binsResp && typeof binsResp}`,
        `binsRespKeys=${binsResp && typeof binsResp === "object" ? Object.keys(binsResp).join(",") : "n/a"}`
      ].join(" | ")
    );
  }
  return {
    targetPoolOrientation,
    activeBin,
    activeBinPrice,
    activeBinRawPricePerToken,
    apiCurrentPrice,
    lowerForSdk,
    upperForSdk,
    sdkPriceOrientation,
    lowerForSdkPriceScale,
    upperForSdkPriceScale,
    bins
  };
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

function extractBinsFromSdkResponse(resp) {
  if (Array.isArray(resp)) return resp;
  if (resp && Array.isArray(resp.bins)) return resp.bins;
  return null;
}

function buildXYAmountDistributionFromSynth({
  bins,
  targetBinIds,
  targetBinWeights,
  targetBinEdges,
  pool,
  sdkPriceOrientation,
  lowerForSdk,
  upperForSdk,
  activeBinId,
  activeBinFloorBps,
  maxBinBpsPerSide,
  totalXAmount,
  totalYAmount
}) {
  if (!Array.isArray(bins) || bins.length === 0) {
    throw new Error("buildXYAmountDistributionFromSynth requires non-empty bins");
  }
  if (!Array.isArray(targetBinWeights) || targetBinWeights.length === 0) {
    throw new Error("synth_weights mode requires target_bin_weights");
  }
  const explicitBinIds = Array.isArray(targetBinIds) && targetBinIds.length > 0;
  if (explicitBinIds) {
    if (targetBinIds.length !== bins.length) {
      throw new Error(`target_bin_ids length mismatch: ids=${targetBinIds.length} bins=${bins.length}`);
    }
    if (targetBinWeights.length !== bins.length) {
      throw new Error(`target_bin_weights length mismatch with explicit target_bin_ids: weights=${targetBinWeights.length} bins=${bins.length}`);
    }
    for (let i = 0; i < bins.length; i++) {
      const expected = Number(targetBinIds[i]);
      const got = extractBinId(bins[i]);
      if (!Number.isInteger(expected)) {
        throw new Error(`target_bin_ids[${i}] is not an integer: ${targetBinIds[i]}`);
      }
      if (expected !== got) {
        throw new Error(`target_bin_ids mismatch at index ${i}: expected=${expected} got=${got}`);
      }
    }
  }
  let weightsForSdkBins = targetBinWeights.map((w) => Number(w));
  let remapMeta = null;
  if (weightsForSdkBins.length !== bins.length) {
    if (explicitBinIds) {
      throw new Error(`target_bin_weights length mismatch: weights=${targetBinWeights.length} bins=${bins.length}`);
    }
    if (!Array.isArray(targetBinEdges) || targetBinEdges.length !== targetBinWeights.length + 1) {
      throw new Error(
        `target_bin_weights length mismatch: weights=${targetBinWeights.length} bins=${bins.length} (and target_bin_edges missing/invalid for remap)`
      );
    }
    remapMeta = remapSynthWeightsToSdkBins({
      bins,
      targetBinWeights: weightsForSdkBins,
      targetBinEdges,
      pool,
      sdkPriceOrientation,
      lowerForSdk,
      upperForSdk
    });
    weightsForSdkBins = remapMeta.weights;
  }

  const hasX = !totalXAmount.isZero();
  const hasY = !totalYAmount.isZero();
  const rawRows = [];
  let lastBinId = null;
  let sumInputWeights = 0;
  for (let i = 0; i < bins.length; i++) {
    const binId = extractBinId(bins[i]);
    if (lastBinId != null && binId !== lastBinId + 1) {
      throw new Error(`Discontinuous Bin ID at index ${i}: got ${binId}, expected ${lastBinId + 1}`);
    }
    lastBinId = binId;
    const w = Number(weightsForSdkBins[i]);
    if (!Number.isFinite(w) || w < 0) {
      throw new Error(`Invalid target_bin_weights[${i}] value after remap: ${weightsForSdkBins[i]}`);
    }
    sumInputWeights += w;
    let xRaw = 0;
    let yRaw = 0;
    let xEligible = false;
    let yEligible = false;
    if (hasX && hasY) {
      if (binId < activeBinId) {
        yEligible = true;
        yRaw = w;
      } else if (binId > activeBinId) {
        xEligible = true;
        xRaw = w;
      } else {
        xEligible = true;
        yEligible = true;
        xRaw = w;
        yRaw = w;
      }
    } else if (hasX) {
      if (binId >= activeBinId) {
        xEligible = true;
        xRaw = w;
      }
    } else if (hasY) {
      if (binId <= activeBinId) {
        yEligible = true;
        yRaw = w;
      }
    } else {
      throw new Error("Both totalXAmount and totalYAmount are zero; cannot add liquidity");
    }
    rawRows.push({ binId, w, xRaw, yRaw, xEligible, yEligible });
  }

  if (sumInputWeights <= 0) {
    throw new Error("target_bin_weights sum must be > 0");
  }

  const xRaw = rawRows.map((r) => r.xRaw);
  const yRaw = rawRows.map((r) => r.yRaw);
  const activeRowIndex = rawRows.findIndex((r) => r.binId === activeBinId);
  const xAlloc = normalizeRawWeightsToBpsWithConstraints({
    rawWeights: xRaw,
    eligibleMask: rawRows.map((r) => Boolean(r.xEligible)),
    sideEnabled: hasX,
    activeIndex: activeRowIndex,
    activeBinFloorBps,
    maxBinBpsPerSide
  });
  const yAlloc = normalizeRawWeightsToBpsWithConstraints({
    rawWeights: yRaw,
    eligibleMask: rawRows.map((r) => Boolean(r.yEligible)),
    sideEnabled: hasY,
    activeIndex: activeRowIndex,
    activeBinFloorBps,
    maxBinBpsPerSide
  });
  const xBps = xAlloc.bps;
  const yBps = yAlloc.bps;
  const xYAmountDistribution = rawRows.map((row, idx) => ({
    binId: row.binId,
    xAmountBpsOfTotal: new BN(String(xBps[idx])),
    yAmountBpsOfTotal: new BN(String(yBps[idx]))
  }));

  const preview = xYAmountDistribution
    .map((row) => ({
      binId: row.binId,
      x_bps: Number(row.xAmountBpsOfTotal.toString()),
      y_bps: Number(row.yAmountBpsOfTotal.toString()),
      total_bps: Number(row.xAmountBpsOfTotal.toString()) + Number(row.yAmountBpsOfTotal.toString())
    }))
    .sort((a, b) => b.total_bps - a.total_bps)
    .slice(0, 5);

  return {
    xYAmountDistribution,
    preview,
    validation: {
      bin_count: bins.length,
      active_bin_id: activeBinId,
      has_x: hasX,
      has_y: hasY,
      x_side_bins: xBps.filter((v) => v > 0).length,
      y_side_bins: yBps.filter((v) => v > 0).length,
      x_bps_total: xBps.reduce((a, b) => a + b, 0),
      y_bps_total: yBps.reduce((a, b) => a + b, 0),
      input_weight_sum: sumInputWeights,
      explicit_bin_ids: explicitBinIds,
      weights_remapped: Boolean(remapMeta),
      source_weight_count: Array.isArray(targetBinWeights) ? targetBinWeights.length : null,
      remap_coverage_ratio: remapMeta?.coverageRatio ?? null,
      synth_weight_active_bin_floor_bps: activeBinFloorBps,
      synth_weight_max_bin_bps_per_side: maxBinBpsPerSide,
      x_constraints: xAlloc.meta,
      y_constraints: yAlloc.meta
    }
  };
}

function remapSynthWeightsToSdkBins({
  bins,
  targetBinWeights,
  targetBinEdges,
  pool,
  sdkPriceOrientation,
  lowerForSdk,
  upperForSdk
}) {
  if (!Array.isArray(targetBinWeights) || !Array.isArray(targetBinEdges)) {
    throw new Error("remapSynthWeightsToSdkBins requires targetBinWeights and targetBinEdges arrays");
  }
  if (targetBinEdges.length !== targetBinWeights.length + 1) {
    throw new Error(
      `Cannot remap weights: target_bin_edges length=${targetBinEdges.length}, expected=${targetBinWeights.length + 1}`
    );
  }

  const srcEdgesSdk = [];
  for (let i = 0; i < targetBinEdges.length; i++) {
    const synthPrice = Number(targetBinEdges[i]);
    if (!Number.isFinite(synthPrice) || synthPrice <= 0) {
      throw new Error(`Invalid target_bin_edges[${i}] value: ${targetBinEdges[i]}`);
    }
    const poolPrice = synthPriceToPoolPrice(pool, synthPrice);
    const sdkPrice = poolPriceToSdkPrice(poolPrice, sdkPriceOrientation);
    if (!Number.isFinite(sdkPrice) || sdkPrice <= 0) {
      throw new Error(`Failed to convert target_bin_edges[${i}] to SDK orientation`);
    }
    srcEdgesSdk.push(sdkPrice);
  }

  let srcWeights = targetBinWeights.map((w) => Number(w));
  let srcEdges = srcEdgesSdk.slice();
  if (srcEdges[0] > srcEdges[srcEdges.length - 1]) {
    srcEdges = srcEdges.slice().reverse();
    srcWeights = srcWeights.slice().reverse();
  }
  ensureStrictlyIncreasing(srcEdges, "target_bin_edges (SDK orientation)");

  const sdkCenters = bins.map((b, i) => {
    const price = numberOrNull(b?.pricePerToken);
    if (!Number.isFinite(price) || price <= 0) {
      throw new Error(`SDK bin ${i} missing/invalid pricePerToken for weight remap`);
    }
    return Number(price);
  });
  ensureStrictlyIncreasing(sdkCenters, "SDK bin pricePerToken");
  const sdkBinEdges = deriveBinEdgesFromCenters({
    centers: sdkCenters,
    lowerBound: Number(lowerForSdk),
    upperBound: Number(upperForSdk)
  });

  const projected = Array.from({ length: bins.length }, () => 0);
  const srcTotal = srcWeights.reduce((acc, v) => acc + (Number.isFinite(v) && v > 0 ? v : 0), 0);
  if (srcTotal <= 0) {
    throw new Error("Cannot remap Synth weights: source weights sum <= 0");
  }

  for (let j = 0; j < srcWeights.length; j++) {
    const w = Number(srcWeights[j]);
    if (!Number.isFinite(w) || w <= 0) continue;
    const lo = srcEdges[j];
    const hi = srcEdges[j + 1];
    const width = hi - lo;
    if (!(width > 0)) continue;
    for (let i = 0; i < projected.length; i++) {
      const overlap = overlapLength(lo, hi, sdkBinEdges[i], sdkBinEdges[i + 1]);
      if (overlap > 0) {
        projected[i] += w * (overlap / width);
      }
    }
  }

  const projectedTotal = projected.reduce((a, b) => a + Math.max(0, b), 0);
  if (projectedTotal <= 0) {
    throw new Error("Remapped Synth weights produced zero overlap with SDK bins");
  }
  const scaled = projected.map((v) => (Math.max(0, v) / projectedTotal) * srcTotal);
  return {
    weights: scaled,
    coverageRatio: projectedTotal / srcTotal
  };
}

function synthPriceToPoolPrice(pool, synthPriceSolUsdc) {
  const sx = String(pool?.symbol_x || "").toUpperCase();
  const sy = String(pool?.symbol_y || "").toUpperCase();
  if (!Number.isFinite(Number(synthPriceSolUsdc)) || Number(synthPriceSolUsdc) <= 0) {
    throw new Error(`Invalid synth price for conversion: ${synthPriceSolUsdc}`);
  }
  const p = Number(synthPriceSolUsdc);
  if (sx === "SOL" && sy === "USDC") return p;
  if (sx === "USDC" && sy === "SOL") return 1 / p;
  throw new Error(`Unsupported pool symbols for Synth edge conversion: ${sx}/${sy}`);
}

function poolPriceToSdkPrice(poolPrice, sdkPriceOrientation) {
  if (!Number.isFinite(Number(poolPrice)) || Number(poolPrice) <= 0) {
    throw new Error(`Invalid pool price for SDK conversion: ${poolPrice}`);
  }
  const p = Number(poolPrice);
  if (sdkPriceOrientation === "inverted_vs_api") return 1 / p;
  return p;
}

function sdkPriceToPoolPrice(sdkPrice, sdkPriceOrientation) {
  if (!Number.isFinite(Number(sdkPrice)) || Number(sdkPrice) <= 0) {
    return null;
  }
  const p = Number(sdkPrice);
  if (sdkPriceOrientation === "inverted_vs_api") return 1 / p;
  return p;
}

function poolPriceToSynthSolUsdcPrice(pool, poolPrice) {
  if (!Number.isFinite(Number(poolPrice)) || Number(poolPrice) <= 0) {
    return null;
  }
  const sx = String(pool?.symbol_x || "").toUpperCase();
  const sy = String(pool?.symbol_y || "").toUpperCase();
  const p = Number(poolPrice);
  if (sx === "SOL" && sy === "USDC") return p;
  if (sx === "USDC" && sy === "SOL") return 1 / p;
  throw new Error(`Unsupported pool symbols for pool->SOL/USDC conversion: ${sx}/${sy}`);
}

function ensureStrictlyIncreasing(values, label) {
  for (let i = 0; i < values.length; i++) {
    const v = Number(values[i]);
    if (!Number.isFinite(v) || v <= 0) {
      throw new Error(`${label} contains invalid value at index ${i}: ${values[i]}`);
    }
    if (i > 0 && !(v > values[i - 1])) {
      throw new Error(`${label} must be strictly increasing (idx ${i - 1}=${values[i - 1]}, idx ${i}=${v})`);
    }
  }
}

function deriveBinEdgesFromCenters({ centers, lowerBound, upperBound }) {
  if (!Array.isArray(centers) || centers.length === 0) {
    throw new Error("deriveBinEdgesFromCenters requires centers");
  }
  if (centers.length === 1) {
    const lo = Math.min(lowerBound, upperBound);
    const hi = Math.max(lowerBound, upperBound);
    if (!(lo > 0 && hi > lo)) {
      throw new Error("Invalid bounds for single-bin edge derivation");
    }
    return [lo, hi];
  }
  const edges = [];
  const lo = Math.min(lowerBound, upperBound);
  const hi = Math.max(lowerBound, upperBound);
  if (!(lo > 0 && hi > lo)) {
    throw new Error(`Invalid SDK range bounds for bin-edge remap: ${lowerBound}, ${upperBound}`);
  }
  edges.push(lo);
  for (let i = 0; i < centers.length - 1; i++) {
    const c0 = Number(centers[i]);
    const c1 = Number(centers[i + 1]);
    const mid = c0 > 0 && c1 > 0 ? Math.sqrt(c0 * c1) : (c0 + c1) / 2;
    if (!Number.isFinite(mid) || mid <= 0) {
      throw new Error(`Invalid midpoint while deriving SDK bin edges at ${i}`);
    }
    edges.push(mid);
  }
  edges.push(hi);
  // Clamp monotonicity in case the requested bounds cut into edge bins.
  for (let i = 1; i < edges.length; i++) {
    if (!(edges[i] > edges[i - 1])) {
      edges[i] = Math.nextUp ? Math.nextUp(edges[i - 1]) : edges[i - 1] * (1 + 1e-12);
    }
  }
  return edges;
}

function overlapLength(aLo, aHi, bLo, bHi) {
  const lo = Math.max(aLo, bLo);
  const hi = Math.min(aHi, bHi);
  return Math.max(0, hi - lo);
}

function normalizeRawWeightsToBpsWithConstraints({
  rawWeights,
  eligibleMask,
  sideEnabled,
  activeIndex,
  activeBinFloorBps,
  maxBinBpsPerSide
}) {
  const n = Array.isArray(rawWeights) ? rawWeights.length : 0;
  if (!sideEnabled) {
    return {
      bps: Array.from({ length: n }, () => 0),
      meta: {
        side_enabled: false,
        eligible_bins: 0,
        cap_applied: false,
        cap_reason: "side_disabled",
        active_floor_requested_bps: 0,
        active_floor_applied_bps: 0,
        active_floor_eligible: false
      }
    };
  }
  if (!Array.isArray(eligibleMask) || eligibleMask.length !== n) {
    throw new Error("eligibleMask must match rawWeights length");
  }

  const sanitized = rawWeights.map((v) => (Number.isFinite(Number(v)) && Number(v) > 0 ? Number(v) : 0));
  const eligible = [];
  for (let i = 0; i < n; i++) {
    if (eligibleMask[i]) eligible.push(i);
  }
  if (eligible.length === 0) {
    throw new Error("No eligible bins for enabled side");
  }
  const rawEligibleSum = eligible.reduce((acc, idx) => acc + sanitized[idx], 0);

  let cap = clampInt(Number(maxBinBpsPerSide ?? 0), 0, 10000);
  let capApplied = false;
  let capReason = "disabled";
  if (cap > 0 && cap < 10000) {
    if (cap * eligible.length < 10000) {
      capReason = "infeasible_for_eligible_bin_count";
      cap = 10000;
    } else {
      capApplied = true;
      capReason = "applied";
    }
  } else if (cap === 10000) {
    capReason = "not_binding";
  }

  const activeEligible = Number.isInteger(activeIndex) && activeIndex >= 0 && activeIndex < n && Boolean(eligibleMask[activeIndex]);
  const activeFloorRequested = clampInt(Number(activeBinFloorBps ?? 0), 0, 10000);
  const capacities = Array.from({ length: n }, (_, i) => (eligibleMask[i] ? cap : 0));
  const mins = Array.from({ length: n }, () => 0);
  let activeFloorApplied = 0;
  if (activeEligible && activeFloorRequested > 0) {
    activeFloorApplied = Math.min(activeFloorRequested, capacities[activeIndex], 10000);
    mins[activeIndex] = activeFloorApplied;
  }
  const reserve = mins.reduce((a, b) => a + b, 0);
  if (reserve > 10000) {
    throw new Error(`Active-bin floor reserve exceeds 10000 bps: reserve=${reserve}`);
  }

  const capRemaining = capacities.map((c, i) => Math.max(0, c - mins[i]));
  const alloc = allocateConstrainedBps({
    rawWeights: sanitized,
    eligibleMask,
    capacityRemaining: capRemaining,
    totalBps: 10000 - reserve
  });
  const bps = mins.map((m, i) => m + alloc[i]);
  const bpsTotal = bps.reduce((a, b) => a + b, 0);
  if (bpsTotal !== 10000) {
    throw new Error(`BPS allocation sum mismatch: got ${bpsTotal}, expected 10000`);
  }
  if (rawEligibleSum <= 0 && activeFloorApplied <= 0) {
    throw new Error("No eligible bins have positive Synth weights and no active-bin floor could be applied");
  }

  return {
    bps,
    meta: {
      side_enabled: true,
      eligible_bins: eligible.length,
      raw_eligible_sum: rawEligibleSum,
      cap_applied: capApplied,
      cap_reason: capReason,
      max_bin_bps_per_side_effective: capApplied ? cap : null,
      active_floor_requested_bps: activeFloorRequested,
      active_floor_applied_bps: activeFloorApplied,
      active_floor_eligible: activeEligible
    }
  };
}

function allocateConstrainedBps({ rawWeights, eligibleMask, capacityRemaining, totalBps }) {
  const n = Array.isArray(rawWeights) ? rawWeights.length : 0;
  const out = Array.from({ length: n }, () => 0);
  if (totalBps <= 0) return out;
  if (!Array.isArray(eligibleMask) || !Array.isArray(capacityRemaining) || eligibleMask.length !== n || capacityRemaining.length !== n) {
    throw new Error("allocateConstrainedBps input lengths mismatch");
  }
  const eligibleIdx = [];
  for (let i = 0; i < n; i++) {
    if (eligibleMask[i] && capacityRemaining[i] > 0) eligibleIdx.push(i);
  }
  const totalCapacity = eligibleIdx.reduce((acc, idx) => acc + capacityRemaining[idx], 0);
  if (totalCapacity < totalBps) {
    throw new Error(`Insufficient BPS capacity: capacity=${totalCapacity}, target=${totalBps}`);
  }
  if (eligibleIdx.length === 0) {
    if (totalBps === 0) return out;
    throw new Error("No capacity available for constrained allocation");
  }

  const frac = Array.from({ length: n }, () => 0);
  const remCap = capacityRemaining.map((v) => Math.max(0, clampInt(Number(v), 0, 10000)));
  let active = eligibleIdx.slice();
  let remaining = Number(totalBps);
  let guard = 0;
  while (remaining > 1e-9 && active.length > 0 && guard < n * 5 + 50) {
    guard += 1;
    let rawSum = 0;
    for (const idx of active) {
      const w = Number(rawWeights[idx]);
      if (Number.isFinite(w) && w > 0) rawSum += w;
    }
    const useUniform = !(rawSum > 0);
    const tentative = active.map((idx) => ({
      idx,
      amount: remaining * (useUniform ? (1 / active.length) : (Math.max(0, Number(rawWeights[idx]) || 0) / rawSum))
    }));

    let saturatedAny = false;
    for (const t of tentative) {
      if (t.amount > remCap[t.idx] + 1e-9) {
        frac[t.idx] += remCap[t.idx];
        remaining -= remCap[t.idx];
        remCap[t.idx] = 0;
        saturatedAny = true;
      }
    }
    if (saturatedAny) {
      active = active.filter((idx) => remCap[idx] > 1e-9);
      continue;
    }
    for (const t of tentative) {
      frac[t.idx] += t.amount;
    }
    remaining = 0;
    break;
  }
  if (remaining > 1e-6) {
    throw new Error(`Constrained allocation did not converge; remaining=${remaining}`);
  }

  let used = 0;
  const remainderRows = [];
  for (let i = 0; i < n; i++) {
    out[i] = Math.min(capacityRemaining[i], Math.floor(frac[i]));
    used += out[i];
    remainderRows.push({
      idx: i,
      frac: frac[i] - out[i],
      capLeft: capacityRemaining[i] - out[i]
    });
  }
  let toDistribute = totalBps - used;
  remainderRows.sort((a, b) => (b.frac - a.frac) || (a.idx - b.idx));
  while (toDistribute > 0) {
    const availableRows = remainderRows.filter((r) => r.capLeft > 0);
    if (availableRows.length === 0) {
      throw new Error(`Unable to distribute remaining ${toDistribute} bps within caps`);
    }
    for (const row of availableRows) {
      if (toDistribute <= 0) break;
      if (row.capLeft <= 0) continue;
      out[row.idx] += 1;
      row.capLeft -= 1;
      toDistribute -= 1;
    }
  }
  return out;
}

function buildStrategyParams({ lowerBinId, upperBinId, totalXAmount, totalYAmount }) {
  const spotEnum =
    StrategyType?.Spot ??
    StrategyType?.SPOT ??
    StrategyType?.SpotBalanced; // backward-compat fallback for older SDKs

  if (spotEnum == null) {
    throw new Error(`Unsupported DLMM SDK StrategyType enum shape: ${JSON.stringify(StrategyType)}`);
  }

  const hasX = !new BN(totalXAmount.toString()).isZero();
  const hasY = !new BN(totalYAmount.toString()).isZero();
  const params = {
    minBinId: Number(lowerBinId),
    maxBinId: Number(upperBinId),
    strategyType: spotEnum
  };

  // Current SDK expects singleSidedX only for one-sided strategies.
  if (hasX && !hasY) {
    params.singleSidedX = true;
  } else if (!hasX && hasY) {
    params.singleSidedX = false;
  }
  return params;
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

function makeRpcRetryingFetch({ maxAttempts, baseDelayMs, maxDelayMs }) {
  return async (input, init) => {
    let lastError = null;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const response = await fetch(input, init);
        if (!shouldRetryHttpStatus(response.status) || attempt === maxAttempts) {
          return response;
        }
        const retryAfterMs = parseRetryAfterMs(response.headers?.get?.("retry-after"));
        const waitMs = retryAfterMs ?? backoffWithJitterMs(attempt, baseDelayMs, maxDelayMs);
        console.error(
          `[rpc-fetch-retry] HTTP ${response.status}; attempt=${attempt}/${maxAttempts}; waiting=${waitMs}ms`
        );
        await sleep(waitMs);
      } catch (error) {
        lastError = error;
        if (!isRetryableRpcError(error) || attempt === maxAttempts) {
          throw error;
        }
        const waitMs = backoffWithJitterMs(attempt, baseDelayMs, maxDelayMs);
        console.error(
          `[rpc-fetch-retry] network error; attempt=${attempt}/${maxAttempts}; waiting=${waitMs}ms; err=${String(error?.message || error)}`
        );
        await sleep(waitMs);
      }
    }
    throw lastError || new Error("RPC fetch retries exhausted");
  };
}

async function withRpcOperationRetries(fn, { label = "rpc-operation" } = {}) {
  let lastError = null;
  for (let attempt = 1; attempt <= RPC_OPERATION_MAX_ATTEMPTS; attempt++) {
    try {
      const value = await fn();
      return { value, attempts: attempt };
    } catch (error) {
      lastError = error;
      if (!isRetryableRpcError(error) || attempt >= RPC_OPERATION_MAX_ATTEMPTS) {
        throw error;
      }
      const waitMs = backoffWithJitterMs(attempt, RPC_OPERATION_BASE_DELAY_MS, RPC_OPERATION_MAX_DELAY_MS);
      console.error(
        `[rpc-op-retry] ${label}; attempt=${attempt}/${RPC_OPERATION_MAX_ATTEMPTS}; waiting=${waitMs}ms; err=${String(error?.message || error)}`
      );
      await sleep(waitMs);
    }
  }
  throw lastError || new Error(`${label} failed after retries`);
}

function shouldRetryHttpStatus(statusCode) {
  return statusCode === 408 || statusCode === 425 || statusCode === 429 || statusCode === 500 || statusCode === 502 || statusCode === 503 || statusCode === 504;
}

function parseRetryAfterMs(raw) {
  if (!raw) return null;
  const seconds = Number(raw);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return Math.round(seconds * 1000);
  }
  const parsedDate = Date.parse(raw);
  if (!Number.isFinite(parsedDate)) return null;
  const delta = parsedDate - Date.now();
  return delta > 0 ? delta : 0;
}

function isRetryableRpcError(error) {
  const message = String(error?.message || error || "").toLowerCase();
  return (
    message.includes("429") ||
    message.includes("too many requests") ||
    message.includes("rate limit") ||
    message.includes("timed out") ||
    message.includes("timeout") ||
    message.includes("econnreset") ||
    message.includes("econnrefused") ||
    message.includes("eai_again") ||
    message.includes("fetch failed") ||
    message.includes("service unavailable") ||
    message.includes("gateway timeout")
  );
}

function backoffWithJitterMs(attempt, baseDelayMs, maxDelayMs) {
  const exp = Math.min(maxDelayMs, Math.round(baseDelayMs * (2 ** Math.max(0, attempt - 1))));
  const jitter = Math.floor(Math.random() * Math.max(1, Math.round(exp * 0.35)));
  return Math.min(maxDelayMs, exp + jitter);
}

async function sleep(ms) {
  const wait = Math.max(0, Number(ms) || 0);
  if (wait <= 0) return;
  await new Promise((resolve) => setTimeout(resolve, wait));
}

function parseEnvInt(name, defaultValue, minValue, maxValue) {
  const raw = process.env[name];
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return defaultValue;
  return clampInt(parsed, minValue, maxValue);
}

function relativeError(a, b) {
  const denom = Math.max(Math.abs(b), 1e-12);
  return Math.abs(a - b) / denom;
}

function numberOrNull(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function clampInt(value, lo, hi) {
  if (!Number.isFinite(value)) return lo;
  return Math.max(lo, Math.min(hi, Math.trunc(value)));
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
