// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 --convert-builtin-func-to-llvm | mlir-translate --mlir-to-llvmir | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 | FileCheck %s

// Consecutive .wt stores and .cv loads must not wait for each other.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: store_wt_no_wait_between_stores:
  // CHECK: global_store_dwordx4 {{.*}} sc0 sc1
  // CHECK-NOT: s_waitcnt vmcnt
  // CHECK: global_store_dwordx4 {{.*}} sc0 sc1
  // CHECK-NOT: s_waitcnt vmcnt
  // CHECK: global_store_dwordx4 {{.*}} sc0 sc1
  // CHECK-NOT: s_waitcnt vmcnt
  // CHECK: global_store_dwordx4 {{.*}} sc0 sc1
  tt.func public @store_wt_no_wait_between_stores(%dst0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %x = arith.constant dense<1.0> : tensor<256xf32, #blocked>
    %s0 = tt.splat %dst0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %p0 = tt.addptr %s0, %range : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %p0, %x {cachePolicy = #tt.cache_policy<cache_modifier = wt, eviction_policy = evict_normal>} : tensor<256x!tt.ptr<f32>, #blocked>
    %s1 = tt.splat %dst1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %p1 = tt.addptr %s1, %range : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %p1, %x {cachePolicy = #tt.cache_policy<cache_modifier = wt, eviction_policy = evict_normal>} : tensor<256x!tt.ptr<f32>, #blocked>
    %s2 = tt.splat %dst2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %p2 = tt.addptr %s2, %range : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %p2, %x {cachePolicy = #tt.cache_policy<cache_modifier = wt, eviction_policy = evict_normal>} : tensor<256x!tt.ptr<f32>, #blocked>
    %s3 = tt.splat %dst3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %p3 = tt.addptr %s3, %range : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %p3, %x {cachePolicy = #tt.cache_policy<cache_modifier = wt, eviction_policy = evict_normal>} : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: load_cv_no_wait_between_loads:
  // CHECK: global_load_dwordx4 {{.*}} sc0 sc1
  // CHECK-NOT: s_waitcnt vmcnt
  // CHECK: global_load_dwordx4 {{.*}} sc0 sc1
  // CHECK-NOT: s_waitcnt vmcnt
  // CHECK: global_load_dwordx4 {{.*}} sc0 sc1
  // CHECK-NOT: s_waitcnt vmcnt
  // CHECK: global_load_dwordx4 {{.*}} sc0 sc1
  tt.func public @load_cv_no_wait_between_loads(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %c256 = arith.constant dense<256> : tensor<256xi32, #blocked>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %p0 = tt.addptr %s, %range : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %x0 = tt.load %p0 {cachePolicy = #tt.cache_policy<cache_modifier = cv, eviction_policy = evict_normal>} : tensor<256x!tt.ptr<f32>, #blocked>
    %p1 = tt.addptr %p0, %c256 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %x1 = tt.load %p1 {cachePolicy = #tt.cache_policy<cache_modifier = cv, eviction_policy = evict_normal>} : tensor<256x!tt.ptr<f32>, #blocked>
    %p2 = tt.addptr %p1, %c256 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %x2 = tt.load %p2 {cachePolicy = #tt.cache_policy<cache_modifier = cv, eviction_policy = evict_normal>} : tensor<256x!tt.ptr<f32>, #blocked>
    %p3 = tt.addptr %p2, %c256 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %x3 = tt.load %p3 {cachePolicy = #tt.cache_policy<cache_modifier = cv, eviction_policy = evict_normal>} : tensor<256x!tt.ptr<f32>, #blocked>
    %a = arith.addf %x0, %x1 : tensor<256xf32, #blocked>
    %b = arith.addf %x2, %x3 : tensor<256xf32, #blocked>
    %sum = arith.addf %a, %b : tensor<256xf32, #blocked>
    %sd = tt.splat %dst : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %pd = tt.addptr %sd, %range : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %pd, %sum : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
