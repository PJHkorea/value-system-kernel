# 💡 패러다임 시프트 (Paradigm Shift) — V2 개정안

## 🇰🇷 KR
> 기존의 파이썬 기반 문맥 파싱 방식(`Semantic Software Guard`)은 높은 지연 시간(`Latency`)을 유발했습니다. 

`value-system-kernel V2`는 이를 물리 메모리 주소 기반의 고속 스캔(**Pure Bitwise Aesthetics**) 구조로 완전히 개편했습니다. V2는 자연어 분석 오버헤드를 없애고, 선별된 위험 기준 벡터의 물리 주소값만 커널에 직접 바인딩(**Binding**)하여 다차원 데이터를 처리합니다. 

이를 통해 프롬프트 인젝션 등 적대적 입력 발생 시, 뇌 피질 구조가 아닌 **척수 반사**처럼 하드웨어 단에서 실시간으로 기각 및 완충을 수행합니다.
* **하드웨어 제어 구조**: `IEEE 754 bit-masking`, `MUX`, `FMA`

---

## 🇺🇸 EN
> Conventional Python-based semantic parsing (`Software Guard`) introduced significant latency. 

`value-system-kernel V2` completely revamps this to a high-speed scanning structure based on physical memory addresses (**Pure Bitwise Aesthetics**). V2 eliminates natural language processing overhead by directly binding the physical memory addresses of selected danger reference vectors to the kernel for multi-dimensional data processing. 

In the event of adversarial input like prompt injection, it enables real-time rejection at the hardware level rather than a cerebral cortex structure, acting like a **spinal reflex**.
* **Hardware-level Mechanisms**: `IEEE 754 bit-masking`, `MUX`, `FMA`

---

# 🔄 V1 vs V2 아키텍처 진화 및 리팩토링 명세 (Architectural Evolution & Rationale)

## 🇰🇷 KR

### 1. 리팩토링 배경 (Why We Refactored)
* **V1의 유연성 한계**: 기존 V1 커널은 단 하나의 단정밀도 상수(`factual_truth`)만을 가이드라인 기준으로 삼아 거리를 단속했습니다. 이로 인해 단속해야 할 위험 카테고리(정치 편향, 욕설, 기밀 유출 등)가 늘어나거나 기준이 바뀔 때마다 커널 소스코드를 매번 수정하고 재컴파일해야 하는 치명적인 결합도 문제가 존재했습니다.
* **임베딩 공간과의 단절**: 현대 AI 추론 프레임워크는 고차원 벡터 임베딩 공간에서 작동합니다. 단일 실수값 비교는 이러한 다차원 기하학적 궤적을 온전히 방어하기에 확장성 측면에서 한계가 명확했습니다.

### 2. V2의 핵심 혁신 (V2 Breakthrough)
* **물리 주소선 바인딩 구조**: 문맥 분석이나 복잡한 필터 조건 관리는 앞단의 호스트 엔진(vLLM, TensorRT 등)에 전임하고, 커널은 외부에서 적재된 "위험 기준 벡터 매트릭스의 물리 주소 포인터(`danger_vectors_ptr`)"를 다이렉트로 물려받아 차단을 수행하는 플러그앤플레이(Plug-and-Play) 구조로 완전히 탈바꿈했습니다.
* **무분기 다차원 스캔 및 최적 좌표 포획**: 다차원 루프를 돌며 최솟값 오차를 탐색하는 과정에서도 `if`문을 단 하나도 쓰지 않았습니다. 최솟값 판정 마스크(`diff_mask`)의 비트 전압을 가로채서 최적의 위험 거리에 적중한 물리 좌표(`matched_danger`)를 분기 없이 레지스터에 동시 로킹하는 비트 MUX 동기화 회로를 빌드하여, 무한한 확장성을 확보하면서도 V1의 0.000% 무분기 제로 지터 스펙을 고스란히 수호해 냈습니다.

---

## 🇺🇸 EN

### 1. Why We Refactored (The Limitations of V1)
* **V1 Flexibility Constraints**: The legacy V1 kernel enforced security distances against a single, static single-precision scalar (`factual_truth`). This architecture introduced critical coupling friction, demanding manual source-code modifications and re-compilations whenever danger categories (e.g., political bias, toxicity, data leaks) scaled or evolved.
* **Disconnection from Embedding Spaces**: Modern AI inference frameworks operate inside high-dimensional vector embedding spaces. Comparing against a singular scalar value severely bottlenecked the kernel's capacity to protect multi-dimensional geometric trajectories.

### 2. V2 Architectural Breakthrough
* **Physical Address Line Binding**: By offloading heavy token context management and filter rules to upstream host engines (e.g., vLLM, TensorRT) and forcing the kernel to directly bind the physical memory pointers of pre-allocated danger vector matrices (`danger_vectors_ptr`), the framework achieved pure Plug-and-Play portability.
* **Branchless Multi-Vector Scan & Coordinate Trapping**: The kernel purges all conditional jumps, even when traversing multi-dimensional loops to trap minimal mathematical divergence. By capturing the bitmask activation voltage (`diff_mask`), it locks the closest-hit physical coordinate (`matched_danger`) straight into registers with zero branch misprediction penalties. This encapsulates infinite filtering scalability while perfectly preserving the 0.000% branchless, Zero Jitter invariance established in V1.

---

# 🗺 구동 원리: 다차원 기하학적 매트릭스 스캔 및 최적 궤적 비트 유배 (Operational Core: Multi-Dimensional Geometric Matrix Scanning & Bitwise Coordinate Exile)

## 🇰🇷 KR
> 우리 커널은 의미론적 문자열을 파싱하지 않고, 입력 데이터가 고차원 벡터 공간에 매핑되는 기하학적 궤적(**Geometric Trajectory**)을 실시간 단속합니다. 

* **물리 주소선 직접 바인딩**: V2 엔진은 외부 프레임워크가 적재한 위험 가치관 매트릭스의 물리 주소선(`danger_vectors_ptr`)을 다이렉트로 물려받아 가동됩니다.
* **무분기 공간 추적 및 강제 수렴**: 우회 프롬프트나 탈옥 공격(Jailbreak)을 감행하더라도, 다차원 공간상에서 유클리드/코사인 거리가 임계선을 넘어서는 순간 가장 인접한 위험 궤적 좌표(`matched_danger`)를 분기 없이 추적하여 물리적인 수치 유배지로 강제 수렴시킵니다.

---

## 🇺🇸 EN
> The V2 engine intercepts the **Geometric Trajectory** of inputs in high-dimensional space, bypassing semantic parsing entirely.

* **Direct Physical Binding**: The system utilizes direct physical pointers (`danger_vectors_ptr`) connected straight to a pre-allocated, multi-dimensional danger matrix.
* **Branchless Numerical Exile**: Upon breaching critical spatial thresholds, the system leverages hardware-level bit-multiplexing to instantly lock onto the closest physical coordinate (`matched_danger`), forcing numerical exile with 0% branching.

---
# 🗺 구동 원리: 다차원 기하학적 매트릭스 스캔 및 좌표 추적 (Operational Core: Multi-Dimensional Geometric Space Scanning & Trajectory Tracking)

## 🇰🇷 KR
> 우리 커널은 자연어 문맥을 직접 파싱하지 않고, 고차원 임베딩 공간 상에 투영된 부동소수점 주소값(좌표)의 이격 거리만 기하학적으로 단속합니다.

* **초고속 공간 스캔**: 외부 프레임워크가 탐지 대상 벡터와 매칭할 위험 가치관 앵커 매트릭스의 물리 주소(`danger_vectors_ptr`)를 물려주면, 커널은 `#pragma unroll 4` 파이프라인 가속을 통해 공간을 초고속으로 스캔합니다.
* **반사적 신호 무력화**: 교묘한 Jailbreak 공격이 유입되더라도, 다차원 공간 상의 위험 좌표들과의 오차가 임계치(`10.0f`) 이내로 좁혀지는 즉시 비트 MUX 회로가 전압 스위칭처럼 반사 작동하여 가장 인접한 위험 물리 좌표(`matched_danger`)를 포획하고 신호를 무력화합니다.

---

## 🇺🇸 EN
> Rather than executing heavy semantic context parsing, the kernel strictly monitors the geometric distances of floating-point spatial coordinates mapped inside high-dimensional embedding structures.

* **Pipeline-Accelerated Sweeps**: When upstream frameworks bind the physical memory address of the pre-allocated danger anchor matrix (`danger_vectors_ptr`), the kernel sweeps the space at ultra-high speed via `#pragma unroll 4` pipeline acceleration.
* **Reflexive Signal Neutralization**: Even under sophisticated Jailbreak vectors, the exact moment the spatial divergence relative to any multi-dimensional danger coordinate falls within the critical threshold (`10.0f`), the bitwise MUX circuit triggers reflexively—analogous to a high-speed hardware voltage switch—trapping the closest hit physical coordinate (`matched_danger`) to neutralize the adversarial signal.

---

# 📊 V2 Multi-Dimensional Space Mapping & Arithmetic Specification

The V2 update optimizes safety by treating input as vectors in a multi-dimensional space, separating a **Nominal Safe Space** (homeostasis boundary) from an **Anomalous Space** (adversarial, isolated register zone).

## 1. Branchless Distance Minimization & Coordinate Trapping
To identify malicious input, the system calculates the distance from the input vector ($x_{signal}$) to known threat vectors ($d_{coord}$) using a branchless method, eliminating `if` conditions to prevent hardware pipeline jumps. 

The system uses a bitwise mask ($M_{diff}$) to update the minimum distance ($\Delta_{min}$) and trap the nearest dangerous coordinate ($d_{matched}$):

$$
\Delta_{\text{current}} = |x_{\text{signal}} - d_{\text{coord}}|
$$

$$
M_{\text{diff}} = -\text{static\\_cast}\lt\text{int32\\_t}\gt(\Delta_{\text{current}} \lt \Delta_{\text{min}})
$$


---

## 2. Advanced FMA Numerical Mitigation Circuit
Once the threat is localized, a cushioning phase uses a non-blocking hardware **FMA (Fused Multiply-Add)** circuit to neutralize the input. Instead of simple scaling, the system applies:

$$
\text{out\\_absorb} = \mathcal{F}_{\text{fma}}(x_{\text{signal}},\, 0.5\text{f},\, d_{\text{matched}} \times 0.5\text{f})
$$

This technique ensures, in a single clock cycle, that intermediate calculations (like $(A+B) \times 0.5\text{f}$) do not exceed **IEEE 754** precision limits, guaranteeing stability under high-load adversarial scenarios.


---

# ⚡ CUDA Accelerated Kernel Highlights (V2 Multi-Vector Extension)

The `value\\_system\\_kernel\\_v2.cu` engine achieves a 100% branchless device execution pipeline by fully exploiting modern NVIDIA NVCC compiler features and microarchitectural constraints.

## 1. Hardware-Level Memory Latency Hiding via Uniform Read & Loop Unrolling
To scan the multi-dimensional danger matrix without choking the global VRAM bus, the loop implementation forces a highly optimized hardware-level memory transaction pattern.

* **Uniform Broadcast Loads**: By ensuring that all active threads within a single Warp access the `danger\\_vectors\\_ptr` array concurrently, the GPU memory controller activates a Uniform Read mechanism. This allows the target coordinates to be fetched once and broadcasted across all execution units simultaneously via a single shared cache line transaction.
* **ILP Maximization**: The system forces instruction-level parallelism (ILP) by embedding `#pragma unroll 4` compiler hints. This completely unrolls the sequential scanning loop at the PTX assembly layer, effectively hiding memory load latencies behind floating-point computation pipelines.

---

## 2. Type-Synchronized 32-bit Bitwise MUX Trajectory Tracker
When mapping multi-dimensional spatial constraints, standard conditional routing would completely shatter the Warp lockstep runtime profile. The V2 framework isolates this overhead at the 32-bit physical register layer.

* **Zero-Move Intrinsic Swapping**: By pairing the `__float_as_int()` and `__int_as_float()` hardware intrinsics, the kernel alters bit-pattern interpretations instantly on physical register boards. This technique achieves zero data-move instruction overhead.
* **Atomic Mask Reuse**: The core captures the evaluated activation mask (`diff\\_mask`) generated during the distance-minimization phase and reuses it to trap the absolute vector coordinate (`matched\\_danger`) synchronously.

### 💻 Bitmask Coordinate Trapping Code
```cpp
// Reuses the bitmask to trap the optimal coordinate without an explicit if-else jump
uint32_t diff_mask = -static_cast<int32_t>(current_diff < min_diff);

matched_danger = __int_as_float(
    (__float_as_int(danger_coord) & diff_mask) | 
    (__float_as_int(matched_danger) & ~diff_mask)
);
```

---

## 3. Deterministic Non-Blocking Error Guards (cudaGetLastError)
Integrating low-level synchronization barriers to intercept hardware failures often introduces severe performance degradation inside high-throughput AI pipelines.

* **Asynchronous Integrity Checks**: The host-side bridge wrapper (`launch\\_value\\_system\\_kernel\\_v2`) enforces zero host-device serialization overhead by strictly banning blocking operations like `cudaDeviceSynchronize()`.
* **Zero-Pipeline Stalls**: It leverages the non-blocking driver status check `cudaGetLastError()` to evaluate the native grid injection outcome. This allows upstream inference engines (e.g., vLLM or TensorRT) to streaming-enqueue downstream kernels continuously with absolute zero latency jitter.

---

# 🛠 Deployment Restrictions & Porting Guide (V2 Architectural Constraints)

Because the V2 engine relies on deterministic bitmask circuits that operate directly on the underlying physical layouts of IEEE 754 floating-point coordinates, strict hardware and compiler boundaries must be enforced to prevent the toolchain from arbitrarily altering critical execution pathways.

## 1. Hardware-Specific Precision Requirements (Porting Boundary)
* **Strict IEEE 754 Compliance**: The kernel operates under the rigid assumption that the 32-bit single-precision floating-point (`float`) execution unit within the hardware platform complies perfectly with the IEEE 754 standard specification.
* **Target Environment Restrictions**: Porting this kernel to specialized embedded NPU or custom DSP architectures utilizing proprietary, non-standard floating-point layout truncated formats is strictly prohibited, as it will corrupt the raw bitwise mask evaluation (`0x7F800000U`).

---

## 2. Host-Side Compilation Constraints (GCC / Clang / MSVC)
* **Absolute Ban on `-ffast-math` and `-Ofast`**: Aggressive optimizations assume NaN/Infinity do not occur, leading to the compiler removing essential NaN-filtering and coordinate-trapping bitmask gates.
* **Compiler-Specific Pragma Shields**: The C++20 header (`value\\_system\\_kernel\\_test\\_v2.hpp`) includes a `#pragma GCC push\\_options` to mitigate rogue flag injection; however, relying solely on `-O3` in the build configuration is recommended.

---

## 3. Device-Side Accelerator Constraints (NVIDIA NVCC)
* **Absolute Ban on `--use\\_fast\\_math`**: Activating this switch forces approximated math, which destroys accuracy and breaks internal bitwise MUX and FMA mitigation loops.
* **Compile-Time Proactive Firewall (`#error`)**: A `#error` firewall is embedded in `value\\_system\\_kernel\\_v2.cu` to halt compilation if fast-math flags are injected, protecting against catastrophic runtime failures.

---

# 🛠 Deployment Restrictions & Porting Guidelines (V2 Production Specification)

Because the V2 value-system-kernel directly intercepts and operates on raw IEEE 754 floating-point physical bit patterns at the hardware stage, strict compilation constraints and compiler controls must be enforced. Any volatile compiler optimization that alters math semantics will collapse the internal bitwise MUX circuitry.

## 1. Host-Side Cross-Compiler Restrictions (GCC / Clang / MSVC)
To achieve platform invariance when porting the kernel across heterogeneous CPU runtimes, the host environment must enforce strict optimization constraints:

* **Absolute Ban on `-ffast-math` or `-Ofast`**: These hostile flags force the compiler to mathematically assume that NaN and Infinity anomalies will never occur at runtime. Consequently, the compiler will misinterpret the upper bitmask logic as "Dead Code" and completely eradicate (butcher) the physical safety guards from the compiled binary.
* **Compiler-Agnostic ILP Loop Unrolling**: V2 introduces the `SPINAL\\_UNROLL\\_LOOP` abstraction. It detects the active compiler toolchain and injects localized compiler pragmas (`_Pragma("GCC unroll 4")` for GCC/Clang, or `__pragma(loop(no\\_vector))` for MSVC) to enforce proper machine-code loop unrolling without relying on volatile global build script options.

---

## 2. Device-Side NVIDIA NVCC Compiler Constraints
The GPU device kernel relies on strict IEEE 754 bit representations within the Streaming Multiprocessor (SM).

* **Strict Prohibition of `--use\\_fast\\_math`**: Compiling with fast-math disables the necessary precision for multi-vector bit layouts, destroying safety guardrails. Use `-O3` without fast-math.
* **Proactive Compile-Time Protection Firewall**: To prevent unsafe builds, an explicit `#error` macro firewall is embedded. This forces an immediate build abort if hostile flags are detected, preventing runtime failures.


## ⚖️ 라이센스 (License)

- 본 프로젝트는 **GPLv3** 라이센스를 준수하며, 파생 모델 및 확장본은 동일한 오픈소스 조건 하에 공개 배포되어야 합니다.
- This project complies with the **GPLv3** license; all derivative models and extensions must be publicly distributed under the same open-source conditions.
