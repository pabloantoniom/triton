// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 --convert-builtin-func-to-llvm | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=NO-AV

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_store_vec8
    tt.func @global_load_store_vec8(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    // Load 8 elements from A with two vectorized load instruction
    // CHECK-COUNT-2: llvm.load {{.*}} : !llvm.ptr<1> -> vector<4xf32>
    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    // Load 8 elements from B with two vectorized load instruction
    // CHECK-COUNT-2: llvm.load {{.*}} : !llvm.ptr<1> -> vector<4xf32>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: volatile_load_in_loop
  tt.func @volatile_load_in_loop(%flag: !tt.ptr<i32>) {
    %c0_i32 = arith.constant 0 : i32
    // CHECK: llvm.br ^[[LOOP:bb[0-9]+]]
    // CHECK-NEXT: ^[[LOOP]]:
    scf.while : () -> () {
      // CHECK-NEXT: llvm.load volatile
      %0 = tt.load %flag {isVolatile = true} : !tt.ptr<i32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      // CHECK: llvm.cond_br {{.*}}, ^[[LOOP]], ^[[EXIT:bb[0-9]+]]
      scf.condition(%1)
    } do {
      scf.yield
    }
    // CHECK-NEXT: ^[[EXIT]]:
    // CHECK-NEXT: llvm.return
    tt.return
  }
}

// -----

// On CDNA3/CDNA4, .cv loads and .wt stores use the av intrinsics at system
// scope instead of volatile accesses. Other targets keep the volatile accesses.
#blocked4 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: load_cv_store_wt_av
  // NO-AV-LABEL: load_cv_store_wt_av
  // NO-AV-NOT: llvm.amdgcn.av
  tt.func @load_cv_store_wt_av(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst16: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %dst8: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
    %range4 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked4>
    %src_splat = tt.splat %src : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked4>
    %src_ptrs = tt.addptr %src_splat, %range4 : tensor<256x!tt.ptr<f32>, #blocked4>, tensor<256xi32, #blocked4>
    // CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_node<#llvm.md_string<"">>
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.av.load.b128"(%{{.*}}, %[[MD]]) : (!llvm.ptr<1>, !llvm.metadata) -> vector<4xi32>
    // NO-AV: llvm.load volatile {{.*}} : !llvm.ptr<1> -> vector<4xf32>
    %x = tt.load %src_ptrs {cachePolicy = #tt.cache_policy<cache_modifier = cv, eviction_policy = evict_normal>} : tensor<256x!tt.ptr<f32>, #blocked4>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked4>
    %dst_ptrs = tt.addptr %dst_splat, %range4 : tensor<256x!tt.ptr<f32>, #blocked4>, tensor<256xi32, #blocked4>
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.av.store.b128"(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<1>, vector<4xi32>, !llvm.metadata) -> ()
    // NO-AV: llvm.store volatile {{.*}} : vector<4xf32>, !llvm.ptr<1>
    tt.store %dst_ptrs, %x {cachePolicy = #tt.cache_policy<cache_modifier = wt, eviction_policy = evict_normal>} : tensor<256x!tt.ptr<f32>, #blocked4>
    %h = arith.truncf %x : tensor<256xf32, #blocked4> to tensor<256xf16, #blocked4>
    %dst16_splat = tt.splat %dst16 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked4>
    %dst16_ptrs = tt.addptr %dst16_splat, %range4 : tensor<256x!tt.ptr<f16>, #blocked4>, tensor<256xi32, #blocked4>
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.av.store.b64"(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<1>, vector<2xi32>, !llvm.metadata) -> ()
    // NO-AV: llvm.store volatile {{.*}} : vector<4xf16>, !llvm.ptr<1>
    tt.store %dst16_ptrs, %h {cachePolicy = #tt.cache_policy<cache_modifier = wt, eviction_policy = evict_normal>} : tensor<256x!tt.ptr<f16>, #blocked4>
    %range1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %dst8_splat = tt.splat %dst8 : !tt.ptr<i8> -> tensor<64x!tt.ptr<i8>, #blocked1>
    %dst8_ptrs = tt.addptr %dst8_splat, %range1 : tensor<64x!tt.ptr<i8>, #blocked1>, tensor<64xi32, #blocked1>
    %bytes = arith.constant dense<1> : tensor<64xi8, #blocked1>
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.av.store.b8"(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<1>, i8, !llvm.metadata) -> ()
    // NO-AV: llvm.store volatile {{.*}} : vector<1xi8>, !llvm.ptr<1>
    // NO-AV-NOT: llvm.amdgcn.av
    tt.store %dst8_ptrs, %bytes {cachePolicy = #tt.cache_policy<cache_modifier = wt, eviction_policy = evict_normal>} : tensor<64x!tt.ptr<i8>, #blocked1>
    tt.return
  }
}

// -----

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = true}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: global_store_mfma_vec16
  tt.func public @global_store_mfma_vec16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x32xf32, #mma>
    %1 = math.exp2 %0 : tensor<32x32xf32, #mma>
    %2 = arith.truncf %1 : tensor<32x32xf32, #mma> to tensor<32x32xf16, #mma>
    %c32_i32 = arith.constant 32 : i32
    %100 = tt.get_program_id x : i32
    %101 = arith.muli %100, %c32_i32 : i32
    %102 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %300 = tt.expand_dims %102 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x32xi32, #mma>
    %200 = tt.broadcast %300 : tensor<1x32xi32, #mma> -> tensor<32x32xi32, #mma>
    %103 = tt.splat %101 : i32 -> tensor<32x32xi32, #mma>
    %104 = arith.addi %103, %200 : tensor<32x32xi32, #mma>
    %105 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #mma>
    %106 = tt.addptr %105, %104 : tensor<32x32x!tt.ptr<f16>, #mma>, tensor<32x32xi32, #mma>
    // Store 16 elements with four vectorized store instruction
    // CHECK-COUNT-4: llvm.store {{.*}} : vector<4xf16>, !llvm.ptr<1>
    tt.store %106, %2 : tensor<32x32x!tt.ptr<f16>, #mma>
    tt.return
  }
}
