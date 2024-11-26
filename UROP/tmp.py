import torch
import subprocess
import re

def get_available_gpus_memory():
    available_gpus = []
    gpu_memory = {}

    if torch.cuda.is_available():
        for gpu_id in range(torch.cuda.device_count()):
            # 'nvidia-smi' 명령어로 각 GPU의 메모리 사용 상태 확인
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
            memory_info = result.stdout.decode('utf-8').strip().split('\n')[gpu_id]
            total_memory, used_memory = map(int, memory_info.split(','))

            free_memory = total_memory - used_memory
            gpu_memory[gpu_id] = free_memory  # GPU ID와 사용 가능한 메모리 기록

            # 메모리가 0보다 큰 GPU를 사용 가능 목록에 추가
            if free_memory > 0:
                available_gpus.append((gpu_id, free_memory))

    return available_gpus, gpu_memory

# 사용 가능한 GPU와 메모리 출력
available_gpus, gpu_memory = get_available_gpus_memory()
print("Available GPU(s) and Free Memory:")
for gpu_id, memory in available_gpus:
    print(f"GPU {gpu_id}: {memory} MB free")

print("\nAll GPU Memory Status:")
for gpu_id, memory in gpu_memory.items():
    print(f"GPU {gpu_id}: {memory} MB free")
