import os
import torch
import yaml
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision import transforms

from models.siamese import SiameseNetwork
from data.dataset import SignetDataset
from train.eval import Evaluator


# =====================================================
# 获取项目根目录（🔥核心）
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =====================================================
# 创建不覆盖的目录
# =====================================================
def create_run_dir(base_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# =====================================================
# 主函数
# =====================================================
def main():

    # ===== 读取配置（绝对路径）=====
    config_path = os.path.join(BASE_DIR, "configs", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 修正所有路径（🔥关键一步）=====
    cfg["data"]["root_dir"] = os.path.join(BASE_DIR, cfg["data"]["root_dir"])
    cfg["eval"]["test"] = os.path.join(BASE_DIR, cfg["eval"]["test"])
    cfg["eval"]["train"] = os.path.join(BASE_DIR, cfg["eval"]["train"])

    print("📂 Dataset:", cfg["data"]["root_dir"])
    print("📂 Train Root:", cfg["eval"]["train"])

    # ===== 创建 test 输出目录 =====
    test_dir = create_run_dir(cfg["eval"]["test"])
    print("📁 Test Dir:", test_dir)

    # ===== 数据 =====
    transform = transforms.Compose([
        transforms.Resize((150, 220)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_dataset = SignetDataset(
        cfg["data"]["root_dir"],
        transform,
        "test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=4
    )

    # ===== 自动找到最新 best_model（更安全版）=====
    # ===== 自动找到最新 best_model =====
    train_root = cfg["eval"]["train"]

    if not os.path.exists(train_root):
        raise RuntimeError("❌ train目录不存在，请先运行 train.py")

    runs = [
        d for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
    ]

    if len(runs) == 0:
        raise RuntimeError("❌ 没有训练结果，请先训练模型")

    latest_run = sorted(runs)[-1]

    best_model_path = os.path.join(train_root, latest_run, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise RuntimeError(f"❌ 未找到 best_model: {best_model_path}")

    latest_run = sorted(runs)[-1]
    best_model_path = os.path.join(
        cfg["eval"]["train"],
        latest_run,
        "best_model.pth"
    )

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"❌ 模型不存在: {best_model_path}")

    print("📦 Load Model:", best_model_path)

    # ===== 加载模型 =====
    model = SiameseNetwork().to(device)

    state_dict = torch.load(
        best_model_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)

    model.eval()

    # ===== 评估 =====
    evaluator = Evaluator(model, device, test_dir)

    print("🚀 Running Test Evaluation...")
    metrics = evaluator.run(test_loader, epoch="final")

    print("✅ Test Finished!")


if __name__ == "__main__":
    main()