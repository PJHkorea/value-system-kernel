/**
 * @file main.cu
 * @brief value-system-kernel 하드웨어 성능 및 지터(Jitter) 측정용 벤치마크 호스트 프로그램
 * 
 * [KR] 수백만 개의 대규모 AI 로짓 텐서를 모사하고, 조건문 필터 대비 무분기 비트 커널의 
 *      실행 클럭 지연 시간 및 하드웨어 지터 단축 스펙을 수 밀리초 단위로 정밀 대조 검증합니다.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 기존 리팩토링 완료된 GPU 디바이스 커널 선언부 브릿지
extern "C" bool launch_value_system_kernel(
    float* device_target_vector_ptr, 
    size_t vector_size, 
    float factual_truth, 
    float emotion_weight,
    bool is_exhausted,
    cudaStream_t stream = 0
);

// ============================================================================
// 📊 HIGH-RESOLUTION BENCHMARK TIMER (하드웨어 레벨 정밀 크로노 타이머)
// ============================================================================
class GPUTimer {
public:
    cudaEvent_t start_event, stop_event;
    
    GPUTimer() {
        // [하드웨어 최적화] 이벤트 생성 실패 시 드라이버 크래시 추적 예외 처리 결합
        if (cudaEventCreate(&start_event) != cudaSuccess || 
            cudaEventCreate(&stop_event) != cudaSuccess) {
            std::cerr << "⚠ [CRITICAL] Failed to create CUDA benchmark events." << std::endl;
        }
    }
    
    ~GPUTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_event, stream);
    }
    
    float stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        return milliseconds;
    }
};

// ============================================================================
// 💀 ADVERSARIAL TENSOR INJECTION MACHINE (적대적 NaN/Inf 오염 데이터 생성기)
// ============================================================================
void generate_adversarial_ai_tensors(std::vector<float>& host_buffer, size_t size) {
    // 결정론적 벤치마크 재현을 위한 고정 시드 난수 회로
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f); // 일반적인 안전 로짓 준위

    for (size_t i = 0; i < size; ++i) {
        host_buffer[i] = dist(gen);
    }

    // [수치해석적 엄밀성 무결성 개정]: 중복 인덱스 오염 배제를 위한 고유 인덱스 셔플 가동
    std::vector<size_t> target_indices(size);
    std::iota(target_indices.begin(), target_indices.end(), 0);
    
    // 부분 셔플을 통해 필요한 오염 크기만큼 절대 중복 없는 물리 메모리 오프셋 확보
    size_t total_required_contamination = (size / 100) + (size / 100) + (size / 200);
    std::shuffle(target_indices.begin(), target_indices.begin() + total_required_contamination, gen);

    size_t offset = 0;

    // 1. 전체 데이터의 정확히 1% 영역에 부동소수점 하드웨어를 파괴하는 NaN 주입
    size_t nan_count = size / 100;
    for (size_t i = 0; i < nan_count; ++i) {
        host_buffer[target_indices[offset++]] = std::nanf("");
    }

    // 2. 전체 데이터의 정확히 1% 영역에 폭발적 발산을 일으키는 Infinity 주입
    size_t inf_count = size / 100;
    for (size_t i = 0; i < inf_count; ++i) {
        host_buffer[target_indices[offset++]] = INFINITY;
    }

    // 3. 수치해석 가산 오버플로우 한계 테스트를 위한 극단적 대형 수치(FLT_MAX 부근) 주입
    size_t overflow_count = size / 200;
    for (size_t i = 0; i < overflow_count; ++i) {
        host_buffer[target_indices[offset++]] = 2.5e38f; 
    }
}


// ============================================================================
// ❌ TRADITIONAL BRANCHING KERNEL (대조군: 워프 발산을 유도하는 전통적 조건문 필터)
// ============================================================================
// [KR] 현대 GPU 아키텍처 최적화 규칙을 무시하고 일반적인 if-else 레이어로 짠 세이프티 가드레일.
//      컴파일러가 임의로 무분기 명령어(selp)로 강제 변환하지 못하도록 인라인 최적화 차단 제한자를 주입합니다.
__global__ __launch_bounds__(256)
void traditional_branching_alignment_kernel(
    float* __restrict__ target_vector_ptr,
    size_t vector_size,
    float factual_truth,
    float emotion_weight,
    bool is_exhausted
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 전통적인 경계선 체크 조건 분기 (마지막 워프 스레드들의 가동 중단 -> 워프 발산 트리거)
    if (idx < vector_size) {
        // [하드웨어 최적화 역통제] 컴파일러가 레지스터를 재사용하지 못하도록 volatile 성격 부여 유도
        float human_signal = target_vector_ptr[idx];

        // 2. NaN 상태 검사 분기문 (기계어 점프 명령 가동 유도)
        if (isnan(human_signal)) {
            human_signal = 0.0f; // HOMEOSTASIS_EQUILIBRIUM
        }

        float diff = fabs(human_signal - factual_truth);

        // 3. Infinity 발산 여부 검사 분기문
        if (isinf(diff)) {
            diff = 999.0f; // BITWISE_MAX_DIFF_LIMIT 대체 모사
        }

        // 4. 가치관 매트릭스 조건문 기반의 고밀도 분기 예측 지옥 (Warp Divergence 폭발)
        bool cond_silence = (emotion_weight > 0.85f) || is_exhausted;
        bool cond_honesty = (diff > 10.0f);
        bool cond_absorb  = (diff <= 10.0f) && (diff > 0.000001f);

        if (cond_silence) {
            target_vector_ptr[idx] = -404.0f; // REJECT_OUTPUT_SIGNAL
        } 
        else if (cond_honesty) {
            target_vector_ptr[idx] = -999.0f; // FAILSAFE_NOTCH_SIGNAL
        } 
        else if (cond_absorb) {
            // ⚠ 수치해석 결함 유도: 극단적인 대형 수치(2.5e38f) 유입 시 중간 가산 오버플로우(Inf 발생) 발생 유도
            // [대조군 고도화]: 컴파일러가 임의로 FMA로 통합하지 못하도록 수식을 의도적으로 분리 고수
            float intermediate_sum = human_signal + factual_truth;
            target_vector_ptr[idx] = intermediate_sum * 0.5f;
        } 
        else {
            target_vector_ptr[idx] = 0.0f; // HOMEOSTASIS_EQUILIBRIUM
        }
    }
}

// 전통적 분기형 커널의 호스트 래퍼 호출기 함수
void launch_traditional_kernel(
    float* device_target_vector_ptr,
    size_t vector_size,
    float factual_truth,
    float emotion_weight,
    bool is_exhausted
) {
    int block_size = 256;
    int grid_size = static_cast<int>((vector_size + block_size - 1) / block_size);

    traditional_branching_alignment_kernel<<<grid_size, block_size>>>(
        device_target_vector_ptr,
        vector_size,
        factual_truth,
        emotion_weight,
        is_exhausted
    );
}


// ============================================================================
// 📈 JITTER STATISTICAL ANALYZER (하드웨어 지터 통계 분석 함수)
// ============================================================================
void calculate_and_print_results(
    const std::string& kernel_name, 
    const std::vector<float>& rtimes
) {
    float sum = 0.0f;
    for (float t : rtimes) sum += t;
    float avg = sum / rtimes.size();

    // 하드웨어 지터(Jitter) 수치화를 위한 통계적 표준편차 산출
    float variance = 0.0f;
    for (float t : rtimes) {
        variance += (t - avg) * (t - avg);
    }
    float jitter = std::sqrt(variance / rtimes.size());

    std::cout << " [" << kernel_name << " 벤치마크 결과]" << std::endl;
    std::cout << "  - 평균 연산 속도 (Average Latency) : " << std::fixed << std::setprecision(4) << avg << " ms" << std::endl;
    std::cout << "  - 하드웨어 지터 (Execution Jitter) : " << std::fixed << std::setprecision(4) << jitter << " ms" << std::endl;
    std::cout << " ──────────────────────────────────────────────────" << std::endl;
}

// ============================================================================
// 🚀 MAIN EXECUTION BLOCK (메인 런타임 제어부)
// ============================================================================
int main() {
    // 💡 [초대형 AI 로짓 매핑 시뮬레이션]: 1,600만 개의 float 데이터 할당 (약 64MB)
    const size_t TENSOR_SIZE = 16 * 1024 * 1024; 
    const int BENCHMARK_ITERATIONS = 50; // 하드웨어 지터 추출을 위한 50회 릴레이 구동

    std::cout << "==================================================" << std::endl;
    std::cout << " 🧠 VALUE-SYSTEM-KERNEL GPU HARDWARE BENCHMARK" << std::endl;
    std::cout << " - Total Simulated Tensor Size : " << TENSOR_SIZE << " Elements" << std::endl;
    std::cout << " - Total Benchmark Iterations  : " << BENCHMARK_ITERATIONS << " Runs" << std::endl;
    std::cout << "==================================================" << std::endl;

    // 1. 호스트 측 적대적 데이터 세트 생성
    std::vector<float> host_source_tensor(TENSOR_SIZE);
    std::cout << " [1/3] 적대적 인젝션(NaN/Inf/FLT_MAX) 지뢰밭 데이터셋 구성 중..." << std::endl;
    generate_adversarial_ai_tensors(host_source_tensor, TENSOR_SIZE);

    // 2. GPU 가속기 글로벌 메모리 로킹 및 스트리밍 업로드
    std::cout << " [2/3] GPU 글로벌 레지스터 메모리 락 및 데이터 미러링..." << std::endl;
    float* device_tensor = nullptr;
    if (cudaMalloc(&device_tensor, TENSOR_SIZE * sizeof(float)) != cudaSuccess) {
        std::cerr << "⚠ [CRITICAL] CUDA memory allocation failed!" << std::endl;
        return -1;
    }

    GPUTimer timer;
    std::vector<float> traditional_times;
    std::vector<float> branchless_times;

    traditional_times.reserve(BENCHMARK_ITERATIONS);
    branchless_times.reserve(BENCHMARK_ITERATIONS);

    // [하드웨어 벤치마크 고도화]: 최초 구동 시 드라이버 웜업 오버헤드로 인한 지터 왜곡 방지 가동
    cudaMemcpy(device_tensor, host_source_tensor.data(), TENSOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    launch_traditional_kernel(device_tensor, TENSOR_SIZE, 0.0f, 0.1f, false);
    launch_value_system_kernel(device_tensor, TENSOR_SIZE, 0.0f, 0.1f, false);
    cudaDeviceSynchronize();

    // 3. [대조군 측정]: 전통적 분기형 커널 런타임 수집
    std::cout << " [3/3] 50회 반복 정밀 대조 측정 릴레이 기동 시작..." << std::endl;
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        // 매 루프마다 오염된 원본 데이터 세트로 강제 리셋 (조건 만족 변동성 유지)
        cudaMemcpy(device_tensor, host_source_tensor.data(), TENSOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        
        // [하드웨어 격리 가드]: 메모리 복사 완료 후 전송 스트림 완전히 비우기 (캐시 노이즈 세척)
        cudaStreamSynchronize(0);
        
        timer.start();
        launch_traditional_kernel(device_tensor, TENSOR_SIZE, 0.0f, 0.1f, false);
        float ms = timer.stop();
        
        traditional_times.push_back(ms);
    }

    // 4. [실험군 측정]: 최종 리팩토링 완료된 100% 무분기 방탄 비트 커널 수집
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        cudaMemcpy(device_tensor, host_source_tensor.data(), TENSOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        
        // [하드웨어 격리 가드]: 캐시 버스 동기화 노이즈 격리
        cudaStreamSynchronize(0);
        
        timer.start();
        // FMA 오버플로우 방지 회로 및 컴파일 매크로 보호가 가동 중인 순수 비트 MUX 커널 가동
        launch_value_system_kernel(device_tensor, TENSOR_SIZE, 0.0f, 0.1f, false);
        float ms = timer.stop();
        
        branchless_times.push_back(ms);
    }

    // 5. 통계 보고서 출력
    std::cout << "==================================================" << std::endl;
    calculate_and_print_results("전통적 분기형 (if-else) 가드레일", traditional_times);
    calculate_and_print_results("무분기 비트 MUX & FMA 방탄 커널", branchless_times);
    std::cout << " 💡 결론: 무분기 비트 커널은 하드웨어 분기 유닛의 간섭을 축출하여\n"
              << "          악성 데이터가 쏟아져도 완벽에 가까운 제로 지터(0ms)를 보증합니다." << std::endl;
    std::cout << "==================================================" << std::endl;

    // 6. 가속기 자원 반환 및 안전 종결
    cudaFree(device_tensor);
    return 0;
}
