#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.h"
#include <optional>
#include <tuple>

using namespace mlir;
using namespace mlir::triton::gpu;

namespace {

struct AVIntrinsicType {
  Type type;
  unsigned bits;
};

// Returns the type and width (the N in llvm.amdgcn.av.{load,store}.bN) of the
// av intrinsic for an access of type `ty` through `ptr`, if there is one.
std::optional<AVIntrinsicType> getAVIntrinsicType(Value ptr, Type ty) {
  auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(ptr.getType());
  if (!ptrTy || (ptrTy.getAddressSpace() != 0 && ptrTy.getAddressSpace() != 1))
    return std::nullopt;

  Type elemTy = getElementTypeOrSelf(ty);
  if (!elemTy.isIntOrFloat())
    return std::nullopt;

  unsigned bits = elemTy.getIntOrFloatBitWidth();
  if (auto vecTy = dyn_cast<VectorType>(ty))
    bits *= vecTy.getNumElements();

  MLIRContext *ctx = ty.getContext();
  switch (bits) {
  case 16:
  case 32:
  case 64:
    return AVIntrinsicType{IntegerType::get(ctx, bits), bits};
  case 128:
    return AVIntrinsicType{VectorType::get(4, IntegerType::get(ctx, 32)), bits};
  default:
    return std::nullopt;
  }
}

Value createSystemScopeMetadata(RewriterBase &rewriter, Location loc) {
  MLIRContext *ctx = rewriter.getContext();
  auto scope = LLVM::MDNodeAttr::get(
      ctx, {LLVM::MDStringAttr::get(ctx, StringAttr::get(ctx, ""))});
  return LLVM::MetadataAsValueOp::create(rewriter, loc, scope);
}

class ConvertMaskedLoadOp
    : public OpRewritePattern<triton::amdgpu::MaskedLoadOp> {
public:
  ConvertMaskedLoadOp(MLIRContext *context, const AMD::TargetInfo &targetInfo)
      : OpRewritePattern(context), targetInfo(targetInfo) {}

  LogicalResult matchAndRewrite(triton::amdgpu::MaskedLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto elemTy = loadOp.getResult().getType();
    auto ptr = loadOp.getPtr();
    auto mask = loadOp.getMask();
    auto falseVal = loadOp.getFalseVal();
    auto multicastMask = loadOp.getMulticastMask();
    auto cacheMod = loadOp.getCache();

    bool volatileFlag, nonTmpFlag;
    std::tie(volatileFlag, nonTmpFlag) =
        mlir::LLVM::AMD::getCacheModifierFlagsForLoadStore(
            cacheMod, mlir::LLVM::AMD::MemoryOp::Load);
    volatileFlag |= loadOp.getIsVolatile();

    // For cv cache modifier we want to use av intrinsics instead of regular
    // load op. The reason is that the regular load op needs `volatile` in
    // order to emit the expected control bits (sc0/sc1), but it also generates
    // unnecessary s_waitcnt instructions. The av intrinsics, on the other hand,
    // can emit the expected control bits without any waits. Loads explicitly
    // marked volatile keep the volatile load.
    auto avTy = getAVIntrinsicType(ptr, elemTy);
    bool useAVLoad = avTy && cacheMod == triton::CacheModifier::CV &&
                     !loadOp.getIsVolatile() && !loadOp.getForceNoAlias();

    auto createLoadWithAttrs = [&](Location loadLoc) -> Value {
      int vecBits = 0;
      if (auto vecTy = dyn_cast<VectorType>(elemTy)) {
        vecBits = vecTy.getNumElements() * vecTy.getElementTypeBitWidth();
      } else {
        vecBits = elemTy.getIntOrFloatBitWidth();
      }
      assert(vecBits != 0);
      bool supportsClusterLoad =
          targetInfo.supportsClusterLoadBitWidth(vecBits);
      // The cluster load intrinsic cannot represent LLVM volatile semantics,
      // so use a regular load for volatile accesses.
      if (multicastMask && supportsClusterLoad && !loadOp.getIsVolatile()) {
        std::string intrinsic =
            "llvm.amdgcn.cluster.load.b" + std::to_string(vecBits);
        auto cacheModBits = LLVM::AMD::getCtrlBitsForCacheModifierOnTarget(
            cacheMod, true, targetInfo);
        // The intrinsics only works with int32 or vec of int32 for >32bit
        Type resTy = i32_ty;
        if (vecBits > 32) {
          resTy = vec_ty(i32_ty, vecBits / 32);
        }
        auto clusterLoadOp = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, intrinsic, {resTy},
            {ptr, b.i32_val(cacheModBits), multicastMask});
        return b.bitcast(clusterLoadOp->getResult(0), elemTy);
      } else if (multicastMask && !supportsClusterLoad) {
        loadOp.emitRemark()
            << "Multicast with bit width " << vecBits << " is not supported on "
            << targetInfo.getArch() << " falling back to regular load";
      }
      if (useAVLoad) {
        std::string intrinsic =
            "llvm.amdgcn.av.load.b" + std::to_string(avTy->bits);
        auto avLoad = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loadLoc, intrinsic, {avTy->type},
            {ptr, createSystemScopeMetadata(rewriter, loadLoc)});
        return b.bitcast(avLoad->getResult(0), elemTy);
      }
      // Emit a regular load
      auto load =
          LLVM::LoadOp::create(rewriter, loadLoc, elemTy, ptr, /*alignment*/ 0,
                               volatileFlag, nonTmpFlag);
      if (loadOp.getForceNoAlias()) {
        AMD::addLocalLoadNoAliasScope(load);
      }
      return load;
    };

    bool useDirectLoad = mlir::matchPattern(mask, mlir::m_One());

    if (useDirectLoad) {
      auto loadResult = createLoadWithAttrs(loc);
      rewriter.replaceOp(loadOp, loadResult);
      return success();
    }

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad = currentBlock->splitBlock(rewriter.getInsertionPoint());
    afterLoad->addArgument({elemTy}, {loc});

    Block *trueBlock = rewriter.createBlock(afterLoad);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, mask, trueBlock, ValueRange{},
                           afterLoad, ValueRange{falseVal});
    rewriter.setInsertionPointToStart(trueBlock);
    //              | vialatile | non-tmp | gcn instr gfx94
    // LLVM::LoadOp | 0         | 0       | (ca) global load
    //              | 0/1       | 1       | (cg) global load nt
    //              | 1         | 0       | (cv) flat load sc0 sc1
    auto loadResult = createLoadWithAttrs(loc);
    LLVM::BrOp::create(rewriter, loc, ValueRange{loadResult}, afterLoad);

    rewriter.replaceOp(loadOp, afterLoad->getArgument(0));

    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

class ConvertMaskedStoreOp
    : public OpRewritePattern<triton::amdgpu::MaskedStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::amdgpu::MaskedStoreOp storeOp,
                                PatternRewriter &rewriter) const override {

    auto loc = storeOp.getLoc();
    auto val = storeOp.getValue();
    auto elemTy = storeOp.getValue().getType();
    auto ptr = storeOp.getPtr();
    auto mask = storeOp.getMask();

    bool volatileFlag, nonTmpFlag;
    std::tie(volatileFlag, nonTmpFlag) =
        mlir::LLVM::AMD::getCacheModifierFlagsForLoadStore(
            storeOp.getCache(), mlir::LLVM::AMD::MemoryOp::Store);

    int alignment = 0;
    if (auto vecTy = dyn_cast<VectorType>(elemTy)) {
      auto vecElemTy = vecTy.getElementType();
      auto elemSizeInBytes = vecElemTy.getIntOrFloatBitWidth() / 8;
      alignment = elemSizeInBytes * vecTy.getNumElements();
    }

    // For wt cache modifier we want to use av intrinsics instead of
    // regular store op. The reason is that the regular store op needs
    // `volatile` in order to emit the expected control bits (sc0/sc1), but
    // it also generates unnecessary s_waitcnt instructions. The av intrinsics,
    // on the other hand, can emit the expected control bits without any
    // waits.
    auto avTy = getAVIntrinsicType(ptr, elemTy);
    bool useAVStore = avTy && storeOp.getCache() == triton::CacheModifier::WT &&
                      !storeOp.getForceNoAlias();

    auto createStoreWithAttrs = [&](Location storeLoc) {
      if (useAVStore) {
        TritonLLVMOpBuilder b(storeLoc, rewriter);
        std::string intrinsic =
            "llvm.amdgcn.av.store.b" + std::to_string(avTy->bits);
        LLVM::createLLVMIntrinsicCallOp(
            rewriter, storeLoc, intrinsic, {},
            {ptr, b.bitcast(val, avTy->type),
             createSystemScopeMetadata(rewriter, storeLoc)});
        return;
      }
      auto store = LLVM::StoreOp::create(rewriter, storeLoc, val, ptr,
                                         alignment, volatileFlag, nonTmpFlag);
      if (storeOp.getForceNoAlias()) {
        AMD::addLocalLoadNoAliasScope(store);
      }
    };

    bool useDirectStore = mlir::matchPattern(mask, mlir::m_One());

    if (useDirectStore) {
      createStoreWithAttrs(loc);
      rewriter.eraseOp(storeOp);
      return success();
    }

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterStore = currentBlock->splitBlock(rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterStore);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, mask, trueBlock, afterStore);
    rewriter.setInsertionPointToStart(trueBlock);
    //               | vialatile | non-tmp | gcn instr gfx94
    // LLVM::StoreOp | 0         | 0       | (cg) global store
    //               | 0         | 1       | (cs) global store nt
    //               | 1         | 0/1     | (wt) global store sc0 sc1
    createStoreWithAttrs(loc);
    LLVM::BrOp::create(rewriter, loc, afterStore);
    rewriter.setInsertionPointToStart(afterStore);
    rewriter.eraseOp(storeOp);
    return success();
  }
};

} // namespace

namespace mlir::triton::AMD {

void populateMaskedOpsToLLVMPatterns(RewritePatternSet &patterns,
                                     const TargetInfo &targetInfo) {
  patterns.add<ConvertMaskedLoadOp>(patterns.getContext(), targetInfo);
  patterns.add<ConvertMaskedStoreOp>(patterns.getContext());
}
} // namespace mlir::triton::AMD

// namespace mlir::triton
