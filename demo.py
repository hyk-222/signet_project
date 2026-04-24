import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 引入你的孪生网络架构
from models.siamese import SiameseNetwork


# ==========================================
# 终端颜色代码（让输出更酷炫）
# ==========================================
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def load_image(img_path, transform, device):
    """加载图片并进行与验证集一致的预处理"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ 找不到图片: {img_path}")

    # 转为灰度图，与 dataset.py 中的逻辑一致
    img = Image.open(img_path).convert('L')
    # 增加 batch 维度: (1, 1, 150, 220)
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def main():
    parser = argparse.ArgumentParser(description="🚀 Signet 签名验证离线推理 Demo")
    parser.add_argument("--img1", type=str, required=True, help="第一张签名图片路径（通常是底库中的基准签名）")
    parser.add_argument("--img2", type=str, required=True, help="第二张签名图片路径（需要验证的现场签名）")
    parser.add_argument("--ckpt", type=str, required=True, help="训练好的 best_model.pth 路径")
    parser.add_argument("--threshold", type=float, default=0.3636, help="判定阈值 (默认基于 ArcFace 最佳结果 0.3636)")
    args = parser.parse_args()

    # 1. 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{Colors.CYAN}✨ 初始化推理引擎... 当前使用设备: {device}{Colors.RESET}")

    # 2. 图像预处理流水线（严格对齐 eval.py）
    transform = transforms.Compose([
        transforms.ResizeAndPad((150, 220)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 3. 加载模型
    print(f"{Colors.CYAN}📦 正在加载模型权重: {args.ckpt}{Colors.RESET}")
    model = SiameseNetwork().to(device)

    try:
        state_dict = torch.load(args.ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"{Colors.RED}🚨 模型加载失败，请检查路径是否正确。错误信息: {e}{Colors.RESET}")
        return

    model.eval()

    # 4. 读取图片
    print(f"{Colors.CYAN}🖼️ 正在处理并对齐输入签名图片...{Colors.RESET}")
    img1 = load_image(args.img1, transform, device)
    img2 = load_image(args.img2, transform, device)

    # 5. 提取特征并计算相似度
    print(f"{Colors.CYAN}🔍 神经特征提取中...{Colors.RESET}")
    with torch.no_grad():
        emb1, emb2 = model(img1, img2)

        # 计算余弦相似度 (Cosine Similarity)
        # 结果范围在 [-1, 1] 之间，越接近 1 越相似
        sim = F.cosine_similarity(emb1, emb2).item()

    # 6. 华丽的输出面板
    print("\n" + Colors.BOLD + "=" * 45 + Colors.RESET)
    print(f" 📂 {Colors.YELLOW}签名 A:{Colors.RESET} {args.img1}")
    print(f" 📂 {Colors.YELLOW}签名 B:{Colors.RESET} {args.img2}")
    print("-" * 45)
    print(f" 🎯 {Colors.CYAN}判定阈值 (Threshold) : {args.threshold:.4f}{Colors.RESET}")
    print(f" 📊 {Colors.CYAN}系统计算相似度 (Sim): {sim:.4f}{Colors.RESET}")
    print("=" * 45)

    if sim > args.threshold:
        print(f"\n{Colors.GREEN}{Colors.BOLD}   ✅ 验证通过：系统判定为同一人的签名！{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}   🚨 报警拦截：笔迹特征不匹配，疑似伪造！{Colors.RESET}")
    print(Colors.BOLD + "=" * 45 + Colors.RESET + "\n")


if __name__ == "__main__":
    main()