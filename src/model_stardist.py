import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CÁC BLOCKS CƠ BẢN (Giữ nguyên) ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x): return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x_dec, x_enc):
        x_dec = self.up(x_dec)
        diffY = x_enc.size(2) - x_dec.size(2)
        diffX = x_enc.size(3) - x_dec.size(3)
        x_dec = F.pad(x_dec, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_enc, x_dec], dim=1)
        return self.conv(x)

# --- KIẾN TRÚC STARDIST (Đã Fix) ---
class StarDist(nn.Module):
    def __init__(self, n_channels=3, n_rays=32, base_filters=32, shared_channels=128):
        super().__init__()
        self.n_channels = n_channels
        self.n_rays = n_rays
        
        # Backbone U-Net
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = EncoderBlock(base_filters, base_filters*2)
        self.down2 = EncoderBlock(base_filters*2, base_filters*4)
        self.down3 = EncoderBlock(base_filters*4, base_filters*8)
        self.down4 = EncoderBlock(base_filters*8, base_filters*16)
        
        self.up1 = DecoderBlock(base_filters*16, base_filters*8)
        self.up2 = DecoderBlock(base_filters*8, base_filters*4)
        self.up3 = DecoderBlock(base_filters*4, base_filters*2)
        self.up4 = DecoderBlock(base_filters*2, base_filters)

        # Shared Head
        self.shared = nn.Sequential(
            nn.Conv2d(base_filters, shared_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(shared_channels),
            nn.ReLU(inplace=True)
        )

        # Output Heads
        self.prob_head = nn.Conv2d(shared_channels, 1, kernel_size=1) 
        self.dist_head = nn.Conv2d(shared_channels, n_rays, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        feat = self.shared(x)

        # --- HEADS ---
        # 1. Probability: Trả về Logits (để tính BCELoss)
        prob_logits = self.prob_head(feat) 
        
        # 2. Distance: Dùng Softplus để đảm bảo > 0
        dist_raw = self.dist_head(feat)
        dist = F.softplus(dist_raw) 
        
        return prob_logits, dist