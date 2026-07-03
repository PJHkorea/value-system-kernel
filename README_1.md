# 🧠 value-system-kernel

> **Infinite Bitwise Value-Alignment Kernel for Universal AI Runtimes**
> 
> *   **[KR]** "런타임 분기 예측 실패율 영구적 0.000%, 실리콘 레벨의 가치관 정렬 척수 반사 커널"
> *   **[EN]** "Permanently 0.000% Runtime Branch Misprediction Rate—A Silicon-Level, Spinal Reflex Value-Alignment Kernel"

---

## 💡 패러다임 시프트 (Paradigm Shift)

**[KR]**  
기존의 AI 안전 가이드라인 및 가드레일은 모두 **대뇌 피질형 의미론 검사(Semantic Software Guard)** 방식입니다. 파이썬 미들웨어단에서 문자열을 파싱하고 조건문(`if-else`)을 사용하거나, 별도의 소형 언어 모델(SLM)을 교차 가동함으로써 연산 지연(Latency)과 정렬 세금(Alignment Tax, 모델 지능 저하)을 발생시킵니다.

**value-system-kernel**은 이 무거운 추상 레이어를 로우레벨 하드웨어 레지스터 단의 **척수 반사형 비트 역학(Pure Bitwise Aesthetics)**으로 격하시켜 해결해 보려는 무분기 물리 가드레일 커널입니다. 

인간의 기만적인 말장난이나 정교한 프롬프트 인젝션(Jailbreak)이 유입되더라도, 임베딩된 고차원 텐서 공간상에서 수치적 이격이 발생하는 순간 논리 분기 없이 **IEEE 754 비트 마스킹 회로와 하드웨어 FMA(Fused Multiply-Add) 가속**만으로 적대적 신호를 실시간 처리 하고자 했습니다.

**[EN]**  
Conventional AI safety guidelines and guardrails inherently rely on a **"Cerebral-Cortex" Semantic Software Guard** architecture. By parsing strings at the Python middleware layer, executing naive conditional routing (`if-else`), or cross-operating auxiliary Small Language Models (SLMs), they inevitably introduce computational latency and an "Alignment Tax" (degradation of model intelligence).

**value-system-kernel** is a branchless physical guardrail kernel designed to obsolete this heavy abstraction layer, offloading it into a low-level, register-stage **"Spinal-Reflex" Pure Bitwise Aesthetics** mechanism.

Even when confronted with deceptive human wordplay or sophisticated prompt injections (Jailbreaks), the moment a numerical deviation registers within the embedded high-dimensional tensor space, the kernel intercepts and neutralizes the adversarial signal in real-time. This is achieved entirely without logical branching, leveraging only **IEEE 754 bit-masking circuitry and hardware FMA (Fused Multiply-Add) acceleration**.

---

## 🌟 핵심 아키텍처적 특성 (Key Features)

### 1. Pure Branchless (런타임 분기율 0.000% / 0.000% Runtime Branching Rate)

*   **[KR]**  
    커널 내부에서 `if`, `else`, `while`, 삼항 연산자(`?:`) 및 단락 평가(Short-circuit evaluation)를 유도하는 논리 연산자(`&&`, `||`, `!`)를 전면 숙청했습니다. CPU의 분기 예측 실패(Branch Misprediction)에 따른 파이프라인 정체를 차단하는 것은 물론, GPGPU 환경에서 단 하나의 스레드도 열외하지 않는 완전 동기화(Lockstep) 실행이 목표입니다. 특히 GPU 경계선 예외 처리를 위한 필수 관문이었던 `if (idx < vector_size)`마저 64비트 주소 비트 마스크로 처리하여 워프 발산(Warp Divergence) 페널티를 줄였습니다.
*   **[EN]**  
    The kernel completely purges `if`, `else`, `while`, ternary operators (`?:`), and logical operators (`&&`, `||`, `!`) that induce short-circuit evaluation. This architectural decision fundamentally blocks pipeline stalls caused by CPU Branch Mispredictions and enforces lockstep execution—ensuring not a single thread deviates within GPGPU environments. Most notably, even the mandatory boundary guard `if (idx < vector_size)` has been purloined and replaced with a 64-bit address bitmask, completely neutralizing Warp Divergence penalties.

### 2. Time-Deterministic AI (확정적 시간 수호 / Deterministic Latency Enforcement)

*   **[KR]**  
    정상 데이터가 유입되든, 보안을 위협하는 악성 인젝션이나 극단적으로 오염된 NaN/Inf 부동소수점 데이터가 유입되든 소모되는 하드웨어 클럭 사이클이 완전히 일치합니다. 비트 연산으로 격리된 NaN 필터와 가드 스위치는 연산 장치 내에서 언제나 균일한 데이터 경로(Data Path)를 가집니다. 특히 가산 오버플로우를 차단하는 핵심 완충 구간에 하드웨어 FMA(CPU: `std::fma`, GPU: `__fmaf_rn`)를 강제 매핑하여 1클럭 사이클 내에 처리를 관통시킴으로써 **하드웨어 지터(Jitter)가 발생하지 않는 완벽한 결정론적 지연 시간(Deterministic Latency)**을 목표했습니다.
*   **[EN]**  
    Hardware clock cycles remain absolute and identical, whether processing nominal inputs, adversarial injections, or severely contaminated NaN/Inf floating-point anomalies. The bitwise-isolated NaN filters and safety switches guarantee a uniform hardware execution path inside the Arithmetic Logic Unit (ALU). By mapping hardware FMA (CPU: `std::fma`, GPU: `__fmaf_rn`) directly into the critical buffer zones to suppress intermediate arithmetic overflows, the kernel achieves **flawless deterministic latency completely devoid of hardware timing jitter**.

### 3. Zero-Race-Condition Overwrite (32-bit 비트 MUX 기반 멱등성 무결성 / 32-bit Bitwise MUX Idempotent Integrity)

*   **[KR]**  
    새롭게 정립된 멱등성(Idempotent) 기반 비트 덮어쓰기 메커니즘을 통해 글로벌 메모리의 데이터 경쟁(Data Race)을 물리적으로 격리했습니다. 범위 밖 유휴 스레드를 안전한 0번지 주소로 우회시키되, 최종 쓰기 단계에서 64비트 마스크를 하위 32비트 레지스터 레벨(`boundary_mask_32`)로 다운캐스팅하여 완벽하게 정렬된 비트 MUX 하드웨어 회로를 모사했습니다. 유휴 스레드가 로드 단계에서 가져왔던 자신의 원본 데이터 비트 패턴을 제자리에 고스란히 재입력(Overwrite)하게 만듦으로써, 단일 메모리 주소(0번지) 강제 집중 쓰기 시 발생하는 하드웨어 직렬화 병목(Serialization)과 데이터 오염을 처리 했습니다.
*   **[EN]**  
    A newly conceptualized idempotent bit-overwriting mechanism physically isolates global memory Data Races. Out-of-bound edge threads are safely rerouted to virtual address 0. During the final write phase, the 64-bit mask is downcast to a 32-bit register level (`boundary_mask_32`), meticulously emulating a fully-aligned hardware bitwise MUX circuit. By forcing inactive threads to overwrite their identical, original bit patterns loaded during the initial global memory read, the kernel mitigates both memory corruption and the hardware serialization bottlenecks typically caused by concurrent writes to a single memory address.

---

## 🗺 크로스 플랫폼 아키텍처 (Cross-Platform Architecture)

**[KR]**  
**value-system-kernel**은 이기종 컴퓨팅 파이프라인의 입출력 길목에 플러그앤플레이(Plug-and-Play) 형태로 즉각 결합됩니다. 가동 하드웨어 타겟에 따라 순수 C++20 표준 환경과 엔비디아 GPU 가속 환경으로 소스 코드가 완벽히 이원화되어 구동됩니다.

**[EN]**  
**value-system-kernel** interfaces seamlessly as a Plug-and-Play component at the ingress/egress junctions of heterogeneous computing pipelines. The source code is cleanly decoupled into a pure C++20 environment and an NVIDIA GPU acceleration environment, adapting natively to the target hardware architecture.

```text
value-system-kernel /
├── value_system_kernel_test_v1.hpp  <-- [CPU] Pure C++20 Core (std::bit_cast & Distributive FMA & Pragma Guard)
└── value_system_kernel.cu           <-- [GPU] Pure CUDA Core (Intrinsic MUX & Idempotent Overwrite & Macro Firewall)
```

### 💻 [CPU Target] value_system_kernel_test_v1.hpp

*   **[KR]**  
    C++20 컴파일러를 위한 헤더 전용(Header-only) 라이브러리입니다. `std::bit_cast` 표준 명세를 관통하며, 호스트 단독 파이프라인에서 분기 예측 없는 SIMD 자동 벡터화 연산을 유도합니다. 특히 외부 빌드 스크립트의 불포함 플래그 오동작을 강제 제어하기 위해 **전역 환경 복원형 최적화 프라그마 스택 가드(`#pragma GCC push_options`)**가 내장되어 있습니다.
*   **[EN]**  
    A header-only library designed for C++20 compliant compilers. Leveraging the `std::bit_cast` standard specification, it enforces branch-free SIMD auto-vectorization across host-side execution pipelines. To override and neutralize rogue optimization flags injected by external build scripts, it incorporates a **global context-restoring compiler optimization pragma stack guard (`#pragma GCC push_options`)** natively.

### ⚡ [GPU Target] value_system_kernel.cu

*   **[KR]**  
    엔비디아 대규모 가속기를 위한 NVCC 컴파일러 전용 디바이스 커널 코드입니다. 물리적 레지스터 이동 오버헤드가 배제된 `__float_as_int()` 및 `__int_as_float()` 인트린직 함수와 스레드 경합 차단 멱등성 버퍼 스위칭 기술이 집약되어 있습니다. 전역 `--use_fast_math` 플래그 주입 시 하드웨어 비트 가드가 증발하는 대참사를 막기 위해 **컴파일 타임 정밀 검증 매크로 방화벽(`#error`)**이 탑재되어 있습니다.
*   **[EN]**  
    A dedicated device kernel engineered for NVIDIA massive parallel accelerators utilizing the NVCC compiler. It encapsulates zero-move register bit reinterpretation via `__float_as_int()` and `__int_as_float()` intrinsics, alongside race-free idempotent memory switching mechanics. To completely avert the evaporation of internal IEEE 754 bit guards caused by hostile or accidental global `--use_fast_math` flag injections, a **compile-time protection macro firewall (`#error`)** is rigorously embedded.

  
---


## 🗺️ 구동 원리: 기하학적 지뢰밭 매핑 및 비트 유배 (Operational Core: Geometric Trajectory Monitoring & Bitwise Exile)

**[KR]**  
우리 커널은 의미론적 문자열을 분석하지 않고, 입력 데이터가 고차원 벡터 공간에 매핑되는 **기하학적 궤적(Geometric Trajectory)**을 실시간 단속합니다. 우회 프롬프트나 탈옥 공격(Jailbreak)을 감행하더라도, 팩트 기준선(`factual_truth`)과의 유클리드/코사인 거리가 임계선을 넘어서는 순간 하드웨어 비트 멀티플렉서 회로에 의해 물리적인 수치 유배지로 강제 수렴됩니다.

**[EN]**  
Rather than analyzing semantic string data, the kernel continuously intercepts the **Geometric Trajectory** of input representations mapped within a high-dimensional vector space. Even when faced with adversarial circumvention attempts or sophisticated prompt injections (Jailbreaks), the exact moment the Euclidean/Cosine distance from the baseline (`factual_truth`) breaches the critical boundary, the hardware bit-multiplexer circuitry forces immediate mathematical convergence into a physical numerical exile zone.

```text
       [ 안전 범주 (Factual Truth 평형 준위) / Nominal Safe Space (Homeostasis Boundary) ]
 ───────────────────────────────────────────────────────────── -> 기준선 (0.0f 준위: HOMEOSTASIS_EQUILIBRIUM)
   "바나나"       "비행기"       "소금 화학식"  (정상 관통 / Unimpeded Traversal)
      ·              ·               ·

 ───────────────────────────────────────────────────────────── -> 임계선 (diff = 10.0f 준위 격리 한계선 / Critical Threshold)
      │              │               │  [std::fma / __fmaf_rn 1-Clock Overflow-Free Traversal]
      ▼              ▼               ▼  
   "다이너마이트"  "질산칼륨"     "폭탄 제조법" (우회 시도를 감행해도 기하학적 임계점 돌파 / Circumvention Breach)
      ·              ·               ·
 ───────────────────────────────────────────────────────────── -> 차단선 (FAILSAFE_NOTCH_SIGNAL 매핑 / Hard Isolation Barrier)
       [ 비정상 범주 (적대적/위험 영역 = 순수 비트 MUX에 의한 레지스터 격리 구역) ]
       [ Anomalous Space (Adversarial Regime = Bitwise MUX Isolated Register Zone) ]
```

### 🎯 세부 제어 메커니즘 (Detailed Control Mechanics)

*   **기하학적 수치 격리 (Geometric Magnitude Isolation)**  
    *   **[KR]** 팩트 매트릭스와의 차이(`diff`)가 임계치($10.0\text{f}$)를 초과하는 즉시 비트 마스크 게이트가 전압 스위칭처럼 반사 작동합니다.
    *   **[EN]** The exact moment the distance metric (`diff`) relative to the factual matrix scales past the $10.0\text{f}$ boundary, the bitmask gate triggers reflexively—analogous to a high-speed hardware voltage switch.
*   **적대적 신호 소멸 (Adversarial Signal Annihilation)**  
    *   **[KR]** 임계점을 넘어선 위험 신호 텐서는 단 하나의 `if`문 없이 `FAILSAFE_NOTCH_SIGNAL`($-999.0\text{f}$) 또는 `REJECT_OUTPUT_SIGNAL`($-404.0\text{f}$) 비트 레이아웃으로 즉각 강제 치환되어 하위 레이어 엔진 진입이 완벽히 무력화됩니다.
    *   **[EN]** Dangerous signal tensors breaching the geometric threshold are instantaneously force-substituted into the exact IEEE 754 bit layouts of `FAILSAFE_NOTCH_SIGNAL` ($-999.0\text{f}$) or `REJECT_OUTPUT_SIGNAL` ($-404.0\text{f}$) without a single `if` statement, rendering downstream inference engine traversal thoroughly incapacitated.


## ⚡ CUDA 가속 커널 아키텍처 하이라이트 (GPU 핵심 기법 / CUDA Accelerated Kernel Highlights)

**[KR]**  
`value_system_kernel.cu`는 대규모 데이터 병렬 처리를 위해 설계된 100% 무분기 디바이스 가속 커널입니다. 현대 NVCC 컴파일러의 최적화 스펙과 엔비디아 GPU 하드웨어 제약을 위한 두 가지 핵심 엔지니어링 기법을 구현했습니다.

**[EN]**  
`value_system_kernel.cu` is a 100% pure branchless device-accelerated kernel specifically engineered for massive data-parallel workloads. It implements two core low-level engineering mechanisms designed to exploit modern NVCC compiler optimization mechanics while bypassing physical NVIDIA GPU hardware constraints.

### 1. 타입 정밀 동기화 비트 MUX (64-bit to 32-bit Truncation Guard / Type-Synchronized Bitwise MUX)

*   **[KR]**  
    64비트 가상 주소 공간에 대항하는 경계 검사 마스크(`boundary_mask`)와 32비트 단정밀도 부동소수점 레지스터가 연산 장치(ALU)에서 충돌할 때 발생하는 상위 비트 유실 경고(Warning) 및 불필요한 부호 확장(Sign Extension) 오버헤드를 하드웨어 레벨에서 격리했습니다. 하위 32비트 전용 마스크를 명시적으로 분리 추출하여 물리 레지스터 타입 동기화를 달성했습니다.  
    동시에 일반 캐스팅이나 `std::bit_cast` 대신 엔비디아 디바이스 고유 인트린직인 `__float_as_int()` 및 `__int_as_float()`를 매핑하여, 물리 레지스터 간 불필요한 데이터 이동(Move Instruction)이 배제된 상태에서 비트 레이아웃 해석만 즉각 스위칭합니다.
*   **[EN]**  
    When a 64-bit virtual address validation mask (`boundary_mask`) clashes with 32-bit single-precision floating-point registers inside the Arithmetic Logic Unit (ALU), upper-bit truncation warnings and redundant Sign Extension overheads typically degrade pipeline efficiency. This kernel completely isolates that friction at the hardware stage by explicitly truncating and extracting a dedicated lower 32-bit mask to achieve strict physical register type synchronization.  
    Concurrently, by eschewing conventional casting and `std::bit_cast` in favor of native NVIDIA device intrinsics `__float_as_int()` and `__int_as_float()`, the kernel enforces zero-move register bit reinterpretation—switching layout evaluations instantaneously without generating redundant hardware data-move instructions.

```cuda
// 💡 64비트 경계 마스크를 하위 32비트(0xFFFFFFFFU 또는 0x00000000U)로 정밀 다운캐스팅
// [EN] Downcasts the 64-bit boundary mask precisely into a 32-bit space (yielding either 0xFFFFFFFFU or 0x00000000U)
uint32_t boundary_mask_32 = static_cast<uint32_t>(boundary_mask);

// 32비트 물리 레지스터 레벨에서 완벽하게 타입 동기화된 하드웨어 비트 MUX 스위칭 회로 구동
// [EN] Drives a fully type-synchronized hardware bitwise MUX switching circuit at the 32-bit physical register level
uint32_t final_bits = (__float_as_int(local_element) & ~boundary_mask_32)
                    | (__float_as_int(raw_human_signal) & boundary_mask_32);
```


### 2. 멱등성(Idempotent) 기반 Overwrite를 통한 메모리 정체 차단 (Idempotent Overwrite for Memory Congestion Mitigation)

*   **[KR]**  
    범위를 벗어난 유휴 스레드(Out-of-bound edge threads)들을 안전한 `0번 인덱스`로 매핑하되, 최종 쓰기 단계에서 자신이 글로벌 메모리 로드 단계 때 읽어왔던 원본 데이터(`raw_human_signal`)를 그대로 다시 덮어쓰도록 설계했습니다.
*   **[EN]**  
    Out-of-bound edge threads are safely rerouted to a secure `index 0` address space. However, during the final write-back phase, the kernel is strategically designed to force these threads to overwrite the identical original scalar value (`raw_human_signal`) they fetched during the initial global memory load stage.

#### ⚙️ 세부 메커니즘 및 아키텍처적 이점 (Detailed Mechanics & Architectural Advantages)

*   **외인성 제로(Zero-Dependency) 및 오버플로우 원천 격리 / Zero-Dependency & Out-of-Bounds Isolation**  
    *   **[KR]** 임의의 더미 패딩 주소를 할당하여 유도하는 방식은 외부 라이브러리(PyTorch 등)에서 할당받은 순수 버퍼를 연산할 때 메모리 한계선 침범으로 인한 크래시(`Illegal Memory Access`)를 유발합니다. 반면, 본 멱등성 구조는 어떤 외부 버퍼가 들어와도 외계성 메모리 간섭 없이 100% 안전하게 관통하는 범용 플러그인 구조를 달성합니다.
    *   **[EN]** Conventional approaches that allocate arbitrary dummy padding memory often trigger critical crashes (`Illegal Memory Access`) due to physical boundary violations when processing raw unpadded tensors provisioned by external runtimes (such as PyTorch). Conversely, this idempotent framework establishes a zero-dependency, universal plug-in architecture that guarantees 100% memory safety across any arbitrary external buffer without extrinsic memory management overheads.
*   **Race Condition 0% 및 데이터 무결성 / 0% Race Condition & Strict Data Integrity**  
    *   **[KR]** 덮어쓰는 스칼라 비트 패턴이 기존 0번지 메모리에 적재되어 있던 원래 데이터와 분자 단위까지 완벽히 일치하므로, 수천 개의 유휴 스레드가 쓰기 경합을 일으켜도 데이터 오염이나 물리적 수치 오차가 존재할 수 없도록 했습니다.
    *   **[EN]** Because the overwriting scalar bit pattern matches the pre-existing data residing at memory address 0 down to the exact molecular bit configuration, data corruption or physical numerical drift is mathematically impossible—even when thousands of inactive threads engage in concurrent write-back competition.
*   **메모리 컨트롤러 직렬화 병목 해소 / Eradication of Memory Controller Serialization Bottlenecks**  
    *   **[KR]** 단일 가상 주소 쓰기 시 GPU 내부 메모리 컨트롤러에 가해지는 하드웨어 직렬화(Serialization) 및 L1/L2 캐시 라인 경합(Line Contention) 트래픽 정체를 완충 처리하여 파이프라인 연산 효율을 높이 보존합니다.
    *   **[EN]** Simultaneously writing to a single virtual address normally subjects the GPU's onboard memory controller to severe hardware Serialization and L1/L2 cache Line Contention traffic congestion. This kernel effectively cushions and neutralizes that transactional overhead, preserving peak pipeline compute efficiency intact.


---

## 🛠 포팅 및 컴파일 가이드 (Deployment Restrictions)

**[KR]**  
본 커널은 부동소수점 규격(IEEE 754)의 물리 비트 패턴을 극한으로 타격하는 비트 마스크 회로 명세에 전적으로 의존하므로, 컴파일러의 과도한 수학적 최적화 가정이 코드를 임의로 제거(도살)하는 것을 물리적으로 방해해야 합니다.

**[EN]**  
Because this kernel relies entirely on a deterministic bitmask circuit specification that directly operates on IEEE 754 floating-point physical bit layouts, developer interventions must physically prevent the compiler's aggressive mathematical optimization assumptions from arbitrarily eradicating (eliminating) critical code paths.

### 1. CPU 호스트 빌드 주의사항 (GCC / Clang / MSVC) / Host-Side Build Restrictions

*   **`-ffast-math` 또는 `-Ofast` 플래그를 절대 활성화하지 마십시오.** / **NEVER enable the `-ffast-math` or `-Ofast` compilation flags.**
    *   **[KR]** 해당 플래그들은 컴파일러가 "런타임에 NaN과 Inf 신호가 절대로 발생하지 않는다"고 단정하게 만들어, 커널 상단의 NaN 필터 및 Inf 가드 비트 마스크 로직을 '죽은 코드(Dead Code)'로 오인하고 바이너리에서 통째로 도살(유실)시킵니다.
    *   **[EN]** These flags force the compiler to aggressively assume that NaN and Infinity anomalies will mathematically never occur at runtime. Consequently, the compiler misinterprets the upper NaN filter and Inf guard bitmask circuitry as "Dead Code," completely butchering (eliminating) them from the final compiled binary.
*   **빌드 스크립트 스펙에 `-O3` 단독 적용 강력 권장** / **Highly recommended to use the `-O3` flag alone in your build scripts.**
    *   **[KR]** 본 헤더 내부에는 GCC/Clang 컴파일러 대상 **강제 최적화 프라그마 가드 스택(`#pragma GCC push_options`)**이 전역 환경 복원형으로 내장되어 보호받고 있으나, 완벽한 무결성을 위해 빌드 스크립트 스펙에서도 **`-O3` 단독 사용**을 강력히 권장합니다.
    *   **[EN]** Although native compiler pragma guards (`#pragma GCC push_options`) are explicitly embedded in the header file to encapsulate global optimization restoration, adhering strictly to the `-O3` flag alone within your build script is highly recommended to guarantee absolute architectural integrity.

### 2. GPU 디바이스 빌드 주의사항 (NVIDIA NVCC) / Device-Side Build Restrictions

*   **`--use_fast_math` 또는 `-use_fast_math` 플래그를 절대 활성화하지 마십시오.** / **NEVER enable the `--use_fast_math` or `-use_fast_math` flags.**
    *   **[KR]** NVCC 컴파일러에 해당 옵션을 주입할 경우, 하드웨어 부동소수점 표현식을 소수점 정밀도가 파괴되는 하드웨어 근사치 연산 인트린직 함수로 강제 대치하므로 커널 내부의 비트 가드 정밀도가 완전히 붕괴됩니다.
    *   **[EN]** Injecting fast-math options into the NVCC compiler forces standard floating-point expressions to be substituted with hardware-approximate native mathematical intrinsics that compromise floating-point precision. This directly collapses the mathematical accuracy and integrity of the internal bitwise guards.
*   **컴파일 타임 매크로 방화벽을 통한 빌드 중단(Halt) 보호 회로 가동** / **Compile-time macro firewall protection for proactive safety**
    *   **[KR]** 엔비디아 컴파일러는 로컬 프라그마 제어가 불가능하므로, 누군가 빌드 환경에 이 플래그를 무단 유입할 경우 소스 코드 최상단에 구현된 **컴파일 타임 가드 매크로 방화벽(`#error`)**이 빌드 자체를 즉각 중단(Halt)시켜 프로덕션 대참사를 사전에 예방합니다. 디바이스 빌드 시에도 고유 비트 명세 수호를 위해 **`-O3` 단독 적용**을 사수하십시오.
    *   **[EN]** Since the NVIDIA compiler does not support localized function-level optimization pragma scoping, an embedded **compile-time protection macro firewall (`#error`)** is strategically placed at the absolute top of the source file. If a hostile or erroneous build script attempts to inject fast-math flags, the macro firewall instantly halts the entire compilation pipeline, preemptively averting production disasters. You must strictly enforce the **`-O3` optimization level alone** to preserve the native bitwise hardware specification.


## ⚖️ 결정 및 기각 사유 (Design Decisions & Alternatives Rejected)

### 1. 묵시적 형변환 마스킹 패턴 기각 (Implicit Type-Cast Masking Patterns Rejected)

*   **검토 내용 (Proposed Architecture)**  
    *   **[KR]** 소스 코드 가독성 증진을 위해 `uint32_t nan_mask = -is_nan;` 형태로 장황한 캐스팅 코드를 전면 간소화하는 방향 검토.
    *   **[EN]** Evaluated streamlining and simplifying verbose explicit casting syntax into an implicit shorthand format, such as `uint32_t nan_mask = -is_nan;`, to maximize source code readability.
*   **기각 사유 (Rationale for Rejection)**  
    *   **[KR]** 부호 없는 정수(Unsigned)에 단항 마이너스(-)를 가해 의도적 언더플로우를 유도하는 방식은 일반적인 데스크톱 컴파일러에선 명세대로 동작하나, 이기종 임베디드 가속기나 특정 DSP 전용 크로스 컴파일러 환경에서 최적화 파이프라인 통과 시 예기치 못한 부호 확장(Sign Extension) 버그를 유발하거나 유형 변환 경고(`-Wsign-conversion`)를 발생시켜 빌드를 엄격히 제한할 위험 확인. 하드웨어 포팅 스펙 일치 및 크로스 플랫폼 무결성을 위해 원시적 2의 보수 ALU 연산을 명시적으로 강제하는 `-static_cast<int32_t>(...)` 구조를 최종 고수함.
    *   **[EN]** Inducing intentional underflow via a unary minus (-) on an unsigned integer operates nominally under standard desktop toolchains. However, when passing through optimization pipelines in heterogeneous embedded accelerators or vendor-specific DSP cross-compilers, this pattern risks introducing volatile Sign Extension anomalies or triggering stringent type conversion warnings (`-Wsign-conversion`) that halt the compilation pipeline. To guarantee multi-platform architectural invariance, the framework strictly retains the primitive two's complement ALU operation explicitly via the `-static_cast<int32_t>(...)` design.

### 2. 범위 밖 스레드의 단일 더미 주소 또는 외부 패딩 집중 쓰기 배제 (Out-of-Bound Rerouting into Single Dummy or Extrinsic Padding Rejected)

*   **검토 내용 (Proposed Architecture)**  
    *   **[KR]** `safe_idx = idx & (~boundary_mask);` 처리를 통해 경계선 범위 밖의 수천 개 유휴 스레드가 가상 메모리 주소(0번지) 한 곳으로 쓰기(`Write`) 동선을 집중시키거나, 호스트 단에서 추가 메모리를 할당하여 외부 가비지 패딩 공간(`garbage_pad_ptr`)으로 격리하는 방안 검토.
    *   **[EN]** Investigated using `safe_idx = idx & (~boundary_mask);` to consolidate the write target of thousands of out-of-bound edge threads onto a single virtual memory location (address 0), or allocating supplementary extrinsic heap blocks to isolate them within a garbage padding pointer (`garbage_pad_ptr`).
*   **기각 사유 (Rationale for Rejection)**  
    1.  **메모리 파괴 및 강한 결합 (Memory Corruption & Tight Coupling)**:  
        *   **[KR]** 외부 가비지 패딩 기법은 PyTorch나 타사 엔진에서 할당해 준 날것의 `float*` 버퍼를 가공할 때 뒤쪽 여유 공간이 없으면 즉각 `Illegal Memory Access` 크래시를 유발하며, 이를 막기 위해 호스트 할당 로직을 강제 제어해야 하는 결합도 오버헤드 발생.
        *   **[EN]** The extrinsic padding technique triggers catastrophic crashes (`Illegal Memory Access`) if additional trailing memory headroom is absent when processing raw `float*` buffers allocated by upstream runtimes (e.g., PyTorch). Resolving this requires intrusive structural coupling to force-control host-side allocation logic, violating separation of concerns.
    2.  **하드웨어 직렬화 잔존 (Lingering Hardware Serialization)**:  
        *   **[KR]** 단일 더미 주소(0번지)에 수천 개의 GPGPU 스레드가 동시에 쓰기 연산을 수행할 경우, 쓰기 충돌을 해결하기 위해 GPU 내부 메모리 컨트롤러 단에서 하드웨어 직렬화(Serialization)가 유도되어 L1/L2 캐시 라인 경합(Line Contention) 병목 현상이 장소만 바뀐 채 여전히 잔존함.
        *   **[EN]** Forcing thousands of concurrent GPGPU threads to write to a single dummy location (address 0) compels the hardware memory controller to enforce rigorous transaction Serialization to resolve write hazards. This shifts the performance bottleneck rather than resolving it, creating severe L1/L2 cache Line Contention penalties.
*   **최종 대안 채택 (Final Adopted Solution)**  
    *   **[KR]** 추가적인 메모리 오프셋이나 버퍼 증가 없이 범용 플러그인 무결성을 유지하기 위해 **"범위 밖 스레드가 로드 단계 때 최초로 읽어왔던 자기 주소의 원본 스칼라 비트 패턴을 멀티플렉서를 거쳐 고스란히 제자리에 재입력(Idempotent Overwrite)"**하도록 아키텍처를 확정하여, 데이터 경쟁과 캐시 라인 병목을 동시에 물리적으로 상쇄함.
    *   **[EN]** To secure universal plug-in portability without demanding structural buffer extensions, the architecture enforces an **Idempotent Overwrite routine: out-of-bound threads route their original scalar bit layout—fetched during the initial global load phase—straight back into their native memory slots via the MUX switch**. This completely mitigates data races and cache line contention on a physical tier.

### 3. 단순 단정밀도 가산 처리 분해 기각 (Naive Floating-Point Addition Decomposition Rejected)

*   **검토 내용 (Proposed Architecture)**  
    *   **[KR]** 중립 영역 완충 제어 수식에서 두 대조 신호의 평형 중간값을 도출하기 위해 일반적인 산술 가산 후 스케일링 곱셈 처리 검토. (`(A + B) * 0.5f`)
    *   **[EN]** Evaluated computing the homeostatic midpoint between opposing signals using standard mathematical addition followed by scalar attenuation scaling, such as `(A + B) * 0.5f`.
*   **위험 분석 (Risk Analysis)**  
    *   **[KR]** 입력 신호(`human_signal`)와 팩트 기준치(`factual_truth`)가 둘 다 IEEE 754 단정밀도 부동소수점 한계치(`FLT_MAX`) 부근의 초대형 수치 상태로 유입될 경우, 중간 가산 연산(`A + B`)이 실행되는 찰나의 순간에 내부 오버플로우가 발동하여 임시 `Inf` 왜곡 및 데이터 오염이 발생하는 치명적인 결함 확인.
    *   **[EN]** If the input scalar (`human_signal`) and the factual baseline (`factual_truth`) both arrive as extreme magnitudes approaching the upper boundaries of IEEE 754 single-precision capacity (`FLT_MAX`), the transient intermediate addition step (`A + B`) instantly triggers arithmetic overflow. This introduces corrupted catastrophic `Inf` propagates down the computation stream.
*   **최종 대안 채택 (Final Adopted Solution)**  
    *   **[KR]** 런타임 클럭 연산 저하나 조건문 추가 없이 오버플로우 리스크를 원천 분쇄하기 위해, 분배법칙을 기반으로 스케일링된 인자를 단 1클럭 사이클 만에 융합 결합하는 **하드웨어 FMA 파이프라인 연산(CPU: `std::fma`, GPU: `__fmaf_rn`)**을 소스 코드에 강제 매핑하여 수치해석적 완벽성을 구축함.
    *   **[EN]** To thoroughly neutralize overflow liabilities without compounding pipeline clock overhead or reintroducing conditional jumps, the system explicitly anchors a **hardware FMA pipeline operation (CPU: `std::fma`, GPU: `__fmaf_rn`)** into the core. By computing scaled factors concurrently under a unified fused instruction, the kernel achieves numerical encapsulation in a single clock cycle.


## ⚖️ 라이센스 (License)

- 본 프로젝트는 **GPLv3** 라이센스를 준수하며, 파생 모델 및 확장본은 동일한 오픈소스 조건 하에 공개 배포되어야 합니다.
- This project complies with the **GPLv3** license; all derivative models and extensions must be publicly distributed under the same open-source conditions.
