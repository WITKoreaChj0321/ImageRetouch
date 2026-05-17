"""
핀토 스튜디오 사진 보정 학습 — Pix2Pix
==============================================
학습 목표:
  원본사진 → 보정사진 (색감 + 형태 모두 학습)
  눈크기, 목살, 날씬하게, 점빼기, 피부보정, 색보정 등

아키텍처:
  Generator  : U-Net (skip connection으로 픽셀 단위 디테일 유지)
  Discriminator: PatchGAN (70×70 패치 단위 진위 판별)
  Loss       : L1(선명도) + 지각손실(VGG) + 적대적손실(GAN)

데이터 구조:
  data/
    original/    ← 원본사진 (DSC001.jpg …)
    retouched/   ← 보정사진 (DSC001.jpg …, 파일명 동일)

권장 쌍 수:
  색감 보정만   : 4~6쌍 충분
  형태 변환 포함 : 30쌍 이상 권장 (새 인물 일반화를 위해)

실행:
  python train.py train
  python train.py train --epochs 200 --size 512
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models

WEIGHT_DIR  = "weights"
GEN_PATH    = os.path.join(WEIGHT_DIR, "generator.pth")
META_PATH   = os.path.join(WEIGHT_DIR, "meta.json")
EXTS        = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


# ══════════════════════════════════════════════════════════════════
# 1. 모델 정의
# ══════════════════════════════════════════════════════════════════

class UNetDown(nn.Module):
    """인코더 블록: Conv → InstanceNorm → LeakyReLU"""
    def __init__(self, in_ch, out_ch, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    """디코더 블록: TransConv → InstanceNorm → ReLU  (skip connection용)"""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        # upsample 먼저 → skip과 해상도 맞춤 → concat
        return torch.cat([self.block(x), skip], dim=1)


class UNetGenerator(nn.Module):
    """
    U-Net Generator (Pix2Pix 표준 구조)
    입력: 원본 이미지 (3ch)
    출력: 보정 이미지 (3ch, tanh → [-1,1])
    skip connection이 픽셀 단위 형태 정보를 디코더로 직접 전달해
    눈크기·슬리밍 같은 공간 변환을 픽셀 수준에서 학습
    """
    def __init__(self, in_ch=3, out_ch=3, features=64):
        super().__init__()
        # Encoder (각 블록이 해상도를 절반으로 줄임)
        self.e1 = UNetDown(in_ch,       features,   normalize=False)  # /2
        self.e2 = UNetDown(features,    features*2)                   # /4
        self.e3 = UNetDown(features*2,  features*4)                   # /8
        self.e4 = UNetDown(features*4,  features*8)                   # /16
        self.e5 = UNetDown(features*8,  features*8)                   # /32
        self.e6 = UNetDown(features*8,  features*8)                   # /64
        self.e7 = UNetDown(features*8,  features*8)                   # /128
        self.e8 = UNetDown(features*8,  features*8, normalize=False)  # /256 (bottleneck)

        # Decoder (skip 연결로 in_ch 두 배)
        self.d1 = UNetUp(features*8,   features*8,  dropout=0.5)
        self.d2 = UNetUp(features*16,  features*8,  dropout=0.5)
        self.d3 = UNetUp(features*16,  features*8,  dropout=0.5)
        self.d4 = UNetUp(features*16,  features*8)
        self.d5 = UNetUp(features*16,  features*4)
        self.d6 = UNetUp(features*8,   features*2)
        self.d7 = UNetUp(features*4,   features)
        self.d8 = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_ch, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        d1 = self.d1(e8, e7)
        d2 = self.d2(d1, e6)
        d3 = self.d3(d2, e5)
        d4 = self.d4(d3, e4)
        d5 = self.d5(d4, e3)
        d6 = self.d6(d5, e2)
        d7 = self.d7(d6, e1)
        return self.d8(d7)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator
    전체 이미지 대신 70×70 패치마다 진짜/가짜 판단
    → 로컬 텍스처(피부, 눈 디테일)까지 평가
    입력: 원본 + 보정(또는 생성) 이미지를 concat (6ch)
    """
    def __init__(self, in_ch=6, features=64):
        super().__init__()
        def block(ic, oc, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, 2, 1, bias=False)]
            if norm:
                layers.append(nn.InstanceNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(in_ch,       features,   norm=False),
            block(features,    features*2),
            block(features*2,  features*4),
            block(features*4,  features*8),
            nn.Conv2d(features*8, 1, 4, 1, 1),  # 패치 단위 출력
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


class VGGPerceptualLoss(nn.Module):
    """
    VGG16 중간 레이어 특징 비교로 지각 손실 계산
    픽셀 단위 L1보다 인간이 느끼는 '보정 느낌'을 더 잘 반영
    """
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
        ]).to(device)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        loss = 0.0
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss += F.l1_loss(x, y)
        return loss


# ══════════════════════════════════════════════════════════════════
# 2. 데이터셋
# ══════════════════════════════════════════════════════════════════

class PairedDataset(Dataset):
    """
    원본-보정 쌍 데이터셋
    적은 쌍(4~6장)에서 최대한 많은 학습 샘플 추출:
      - random crop: 한 이미지에서 여러 위치의 패치 추출
      - flip, color jitter, rotate 등 증강
    """
    def __init__(self, orig_dir, ret_dir, size=256, augment=True, crops_per_pair=16):
        self.pairs = []
        self.size  = size
        self.augment     = augment
        self.crops_per_pair = crops_per_pair

        orig_stems = {Path(f).stem: Path(orig_dir)/f
                      for f in os.listdir(orig_dir) if Path(f).suffix.lower() in EXTS}
        ret_stems  = {Path(f).stem: Path(ret_dir)/f
                      for f in os.listdir(ret_dir)  if Path(f).suffix.lower() in EXTS}

        for stem in sorted(orig_stems):
            if stem in ret_stems:
                self.pairs.append((str(orig_stems[stem]), str(ret_stems[stem])))

        if not self.pairs:
            raise ValueError(
                "원본-보정 쌍이 없습니다.\n"
                "original/ 과 retouched/ 폴더의 파일명(확장자 제외)을 동일하게 맞추세요."
            )

        self.color_jitter = T.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03
        )

    def __len__(self):
        return len(self.pairs) * self.crops_per_pair

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx):
        orig_path, ret_path = self.pairs[idx % len(self.pairs)]
        orig = Image.open(orig_path).convert("RGB")
        ret  = Image.open(ret_path).convert("RGB")

        # 같은 크기로 맞추기
        if orig.size != ret.size:
            ret = ret.resize(orig.size, Image.LANCZOS)

        # 충분한 크기로 리사이즈 (crop 여유)
        min_side = min(orig.size)
        crop_size = self.size
        if min_side < crop_size:
            scale = crop_size / min_side * 1.1
            new_w = int(orig.width  * scale)
            new_h = int(orig.height * scale)
            orig = orig.resize((new_w, new_h), Image.LANCZOS)
            ret  = ret.resize( (new_w, new_h), Image.LANCZOS)

        # ── 동기화된 랜덤 augmentation ──
        # Random crop (동일 위치)
        i, j, h, w = T.RandomCrop.get_params(orig, (crop_size, crop_size))
        orig_c = TF.crop(orig, i, j, h, w)
        ret_c  = TF.crop(ret,  i, j, h, w)

        if self.augment:
            # 수평 플립
            if torch.rand(1) > 0.5:
                orig_c = TF.hflip(orig_c)
                ret_c  = TF.hflip(ret_c)

            # 미세 회전
            if torch.rand(1) > 0.7:
                angle = float(torch.empty(1).uniform_(-5, 5))
                orig_c = TF.rotate(orig_c, angle)
                ret_c  = TF.rotate(ret_c,  angle)

            # 원본에만 color jitter (모델이 다양한 촬영 조건에 강건해짐)
            orig_c = self.color_jitter(orig_c)

        return self._to_tensor(orig_c), self._to_tensor(ret_c)


# ══════════════════════════════════════════════════════════════════
# 3. 학습 루프
# ══════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── 데이터 ──
    dataset = PairedDataset(
        args.orig_dir, args.ret_dir,
        size=args.size,
        augment=True,
        crops_per_pair=args.crops_per_pair,
    )
    pair_count = len(dataset.pairs)
    print(f"[Data] 학습 쌍: {pair_count}쌍  |  샘플 수: {len(dataset)}  |  크기: {args.size}×{args.size}")
    if pair_count < 10:
        print(f"[Warn] 쌍이 {pair_count}개뿐입니다. 색감 보정은 잘 학습되지만 "
              "슬리밍·눈크기 등 형태 변환은 30쌍 이상 권장합니다.")

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── 모델 ──
    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    perc = VGGPerceptualLoss(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr,    betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr*0.5, betas=(0.5, 0.999))

    # LR 감쇠: 중반부터 선형 감쇠
    def lr_lambda(epoch):
        decay_start = args.epochs // 2
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / (args.epochs - decay_start))

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    bce = nn.BCEWithLogitsLoss()
    l1  = nn.L1Loss()

    os.makedirs(WEIGHT_DIR, exist_ok=True)

    print("[Train] 학습 시작...")
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        g_total = d_total = 0.0

        for orig, ret in loader:
            orig, ret = orig.to(device), ret.to(device)
            fake = G(orig)

            # ── Discriminator ──
            real_pred = D(orig, ret)
            fake_pred = D(orig, fake.detach())
            d_loss = 0.5 * (
                bce(real_pred, torch.ones_like(real_pred)) +
                bce(fake_pred, torch.zeros_like(fake_pred))
            )
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # ── Generator ──
            fake_pred2 = D(orig, fake)
            g_adv   = bce(fake_pred2, torch.ones_like(fake_pred2))
            g_l1    = l1(fake, ret) * args.lambda_l1
            g_perc  = perc(
                (fake  + 1) / 2,   # [-1,1] → [0,1]
                (ret   + 1) / 2
            ) * args.lambda_perc
            g_loss  = g_adv + g_l1 + g_perc
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            g_total += g_loss.item()
            d_total += d_loss.item()

        sched_G.step(); sched_D.step()

        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch:4d}/{args.epochs}] "
                  f"G: {g_total/len(loader):.3f}  D: {d_total/len(loader):.3f}")

        # 체크포인트 (매 50 epoch)
        if epoch % 50 == 0 or epoch == args.epochs:
            torch.save(G.state_dict(), GEN_PATH)

    # 메타 정보 저장
    meta = {"img_size": args.size, "pair_count": pair_count, "epochs": args.epochs}
    with open(META_PATH, "w") as f:
        json.dump(meta, f)

    print(f"[Done] Generator 저장: {GEN_PATH}")


# ══════════════════════════════════════════════════════════════════
# 4. 추론
# ══════════════════════════════════════════════════════════════════

def load_generator(device="cpu") -> tuple[UNetGenerator, dict]:
    if not os.path.exists(GEN_PATH):
        raise FileNotFoundError(f"학습된 모델 없음: {GEN_PATH}")
    meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {"img_size": 256}
    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load(GEN_PATH, map_location=device))
    G.eval()
    return G, meta


def apply_generator(G: UNetGenerator, image: np.ndarray,
                    tile_size: int = 256, device="cpu") -> np.ndarray:
    """
    고해상도 이미지에 타일 기반으로 Generator 적용
    메모리 절약 + 임의 해상도 지원
    """
    H, W = image.shape[:2]

    img_t = torch.from_numpy(
        image.astype(np.float32) / 127.5 - 1.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        if max(H, W) <= tile_size * 2:
            # 이미지가 작으면 한 번에 처리
            resized = F.interpolate(img_t, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
            out     = G(resized)
            out     = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        else:
            # 큰 이미지: 타일 단위로 처리 후 합성
            out = torch.zeros_like(img_t)
            cnt = torch.zeros(1, 1, H, W, device=device)
            step = tile_size * 3 // 4  # 25% 오버랩으로 경계 블렌딩

            for y in range(0, H, step):
                for x in range(0, W, step):
                    y2 = min(y + tile_size, H)
                    x2 = min(x + tile_size, W)
                    y1 = max(0, y2 - tile_size)
                    x1 = max(0, x2 - tile_size)
                    tile    = img_t[:, :, y1:y2, x1:x2]
                    tile_out = G(tile)
                    out[:, :, y1:y2, x1:x2] += tile_out
                    cnt[:, :, y1:y2, x1:x2] += 1
            out = out / cnt.clamp(min=1)

    result = ((out.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return result


# ══════════════════════════════════════════════════════════════════
# 5. CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="핀토 보정 학습/적용 (Pix2Pix)")
    sub = parser.add_subparsers(dest="cmd")

    # --- train ---
    p_tr = sub.add_parser("train", help="학습")
    p_tr.add_argument("--orig_dir",       default="data/original")
    p_tr.add_argument("--ret_dir",        default="data/retouched")
    p_tr.add_argument("--epochs",         type=int,   default=300)
    p_tr.add_argument("--batch_size",     type=int,   default=4)
    p_tr.add_argument("--size",           type=int,   default=256,  help="패치 크기 (256 or 512)")
    p_tr.add_argument("--lr",             type=float, default=2e-4)
    p_tr.add_argument("--lambda_l1",      type=float, default=100.0)
    p_tr.add_argument("--lambda_perc",    type=float, default=10.0)
    p_tr.add_argument("--crops_per_pair", type=int,   default=32,   help="쌍당 추출 패치 수 (적은 쌍 보완)")

    # --- apply ---
    p_ap = sub.add_parser("apply", help="단일 이미지 보정")
    p_ap.add_argument("input")
    p_ap.add_argument("output")

    # --- batch ---
    p_ba = sub.add_parser("batch", help="배치 보정")
    p_ba.add_argument("input_dir")
    p_ba.add_argument("output_dir")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.cmd == "train":
        train(args)

    elif args.cmd == "apply":
        G, meta = load_generator(device)
        img = np.array(Image.open(args.input).convert("RGB"))
        result = apply_generator(G, img, tile_size=meta.get("img_size", 256), device=device)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        Image.fromarray(result).save(args.output, quality=95)
        print(f"[Done] {args.output}")

    elif args.cmd == "batch":
        G, meta = load_generator(device)
        tile_size = meta.get("img_size", 256)
        os.makedirs(args.output_dir, exist_ok=True)
        files = [f for f in Path(args.input_dir).iterdir() if f.suffix.lower() in EXTS]
        print(f"[Batch] {len(files)}장 처리 중...")
        for i, f in enumerate(files, 1):
            img    = np.array(Image.open(str(f)).convert("RGB"))
            result = apply_generator(G, img, tile_size=tile_size, device=device)
            out    = os.path.join(args.output_dir, f.stem + ".jpg")
            Image.fromarray(result).save(out, quality=95)
            print(f"  [{i}/{len(files)}] {f.name}")
        print(f"[Done] {args.output_dir}/")

    else:
        parser.print_help()
