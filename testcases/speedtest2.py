import torch


def test_pytorch_gpu():
    print("\n===== PyTorch GPU Test Program =====\n")

    # 1. PyTorch 基本信息
    print(f"PyTorch version      : {torch.__version__}")
    print(f"Compiled CUDA version: {torch.version.cuda}")

    # 2. CUDA 可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available       : {cuda_available}")

    if not cuda_available:
        print("\n❌ CUDA 不可用：当前 PyTorch 为 CPU 版本或 CUDA 环境异常")
        return

    # 3. GPU 信息
    gpu_count = torch.cuda.device_count()
    print(f"GPU count            : {gpu_count}")

    for i in range(gpu_count):
        print(f"GPU {i} name          : {torch.cuda.get_device_name(i)}")

    # 4. Tensor GPU 运算测试（关键）
    device = torch.device("cuda")

    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)

    z = x @ y  # 矩阵乘法（GPU 计算）

    print("\n[Tensor Test]")
    print(f"x device             : {x.device}")
    print(f"z device             : {z.device}")
    print(f"z mean value         : {z.mean().item():.6f}")

    # 5. 额外确认：current device
    print(f"\nCurrent CUDA device  : {torch.cuda.current_device()}")

    print("\n✅ PyTorch GPU 测试通过")


if __name__ == "__main__":
    test_pytorch_gpu()
