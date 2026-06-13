import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# นำเข้าโมดูลทั้ง 3 ที่ผู้ใช้เตรียมไว้
import rh_one
import grh_one
import bsd_one

# =============================================================================
# 1. Configuration Dataclass
# =============================================================================
@dataclass
class SGNO_NT_Config:
    node_in_dim: int = 2         # [position, local_spacing]
    global_dim: int = 3          # [degree (d), conductor (q), rank (r)]
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    
    # Loss Weights
    lambda_rh: float = 1.0
    lambda_grh: float = 1.0
    lambda_bsd: float = 1.5      # ให้น้ำหนัก BSD มากขึ้นสำหรับ Rank-aware loss

# =============================================================================
# 2. Structural FiLM-Modulated 1D Convolution
# =============================================================================
class FiLMConv1dBlock(nn.Module):
    """
    บล็อก Convolution 1 มิติที่ใช้ประมวลผลตำแหน่งอนุภาค (Particle spacings)
    ถูกปรับเทียบ (Modulated) โดยฟิลด์โครงสร้าง (sigma) และพารามิเตอร์ L-function
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * 2, dim, kernel_size=3, padding=1)
        )
        # FiLM Modulators: แปลง (sigma + L_params) -> gamma, beta
        self.film_gamma = nn.Linear(1 + 3, dim) 
        self.film_beta  = nn.Linear(1 + 3, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Dim, N_particles)
        gamma = self.film_gamma(context).unsqueeze(-1)  # (Batch, Dim, 1)
        beta  = self.film_beta(context).unsqueeze(-1)   # (Batch, Dim, 1)
        
        modulated_x = (gamma * x) + beta
        out = self.conv(modulated_x)
        
        # Residual + LayerNorm (ต้องสลับแกนเพื่อใช้ LayerNorm)
        out = out.permute(0, 2, 1)
        x_res = x.permute(0, 2, 1)
        return self.norm(x_res + out).permute(0, 2, 1)

# =============================================================================
# 3. Main AI Surrogate Model
# =============================================================================
class StructuralGNONumberTheory(nn.Module):
    """
    AI Surrogate สำหรับขับเคลื่อน SSC Dynamics ของสมมติฐานรีมันน์ทุกรูปแบบ
    """
    def __init__(self, cfg: SGNO_NT_Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        
        self.node_embed = nn.Sequential(
            nn.Linear(cfg.node_in_dim, d),
            nn.LayerNorm(d)
        )
        
        self.layers = nn.ModuleList([
            FiLMConv1dBlock(d, cfg.dropout) for _ in range(cfg.num_layers)
        ])
        
        # พยากรณ์ระยะกระจัด (Drift / Delta x) ของอนุภาค
        self.drift_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1)
        )

    def forward(self, x_pos: torch.Tensor, l_params: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        x_pos: (Batch, N) ตำแหน่งอนุภาค
        l_params: (Batch, 3) ตัวแปร [d, q, r]
        sigma: (Batch, 1) ค่า CSOC stress / ความเค้นโครงสร้าง
        """
        # คำนวณ Local Spacing
        sorted_x, _ = torch.sort(x_pos, dim=1)
        spacings = torch.cat([
            torch.zeros(x_pos.size(0), 1, device=x_pos.device),
            sorted_x[:, 1:] - sorted_x[:, :-1]
        ], dim=1)
        
        # Node features: (Batch, N, 2)
        nodes = torch.stack([sorted_x, spacings], dim=-1)
        h = self.node_embed(nodes).permute(0, 2, 1)  # (Batch, d, N)
        
        # Context สำหรับ FiLM: นำ sigma มารวมกับพารามิเตอร์ [d, q, r]
        context = torch.cat([sigma, l_params], dim=-1)  # (Batch, 4)
        
        for layer in self.layers:
            h = layer(h, context)
            
        h = h.permute(0, 2, 1)  # (Batch, N, d)
        drift = self.drift_head(h).squeeze(-1)  # (Batch, N)
        
        return sorted_x + drift

# =============================================================================
# 4. Multi-Module Unified Trainer
# =============================================================================
class UnifiedRHTrainer:
    def __init__(self, model: StructuralGNONumberTheory, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_step(self, x_init: torch.Tensor, sigma: torch.Tensor, mode: str, 
                   d: int = 1, q: float = 1.0, r: int = 0, target_spacings=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size = x_init.size(0)
        device = x_init.device
        
        # สร้าง Tensor พารามิเตอร์ L-function
        l_params = torch.tensor([[d, q, r]], dtype=torch.float32, device=device).repeat(batch_size, 1)
        
        # พยากรณ์ตำแหน่งอนุภาคใหม่
        x_pred = self.model(x_init, l_params, sigma)
        sorted_pos, _ = torch.sort(x_pred, dim=1)
        
        # =========================================================
        # จัดการ Unfolding และ Loss ตาม Module ที่ระบุ
        # =========================================================
        if mode == 'rh':
            # ใช้การคลี่พิกัดแบบ RH ONE ปกติ
            unfolded_pos = rh_one.unfold_positions_torch(sorted_pos)
            
        elif mode == 'grh':
            # ใช้การคลี่พิกัดตาม Generalized L-function
            l_func = grh_one.GeneralizedLFunction(degree=d, conductor=q)
            unfolded_pos = grh_one.unfold_l_positions_torch(sorted_pos, l_func)
            
        elif mode == 'bsd':
            # ใช้การคลี่พิกัดแบบ Elliptic Curve L-function
            ec_curve = bsd_one.EllipticCurveLFunction(label=f"{q}a1", conductor=int(q), rank=r)
            unfolded_pos = bsd_one.unfold_ec_positions_torch(sorted_pos, ec_curve)
        
        else:
            raise ValueError("Mode must be 'rh', 'grh', or 'bsd'")

        # คำนวณ GUE Statistical Losses พื้นฐาน
        spacings = unfolded_pos[:, 1:] - unfolded_pos[:, :-1]
        loss_s = rh_one.spacing_loss(spacings)
        loss_pc = rh_one.pair_correlation_loss(unfolded_pos)
        loss_sff = rh_one.spectral_form_factor_loss(unfolded_pos)
        loss_nv = rh_one.number_variance_loss(unfolded_pos)
        
        total_loss = loss_s + 0.5 * loss_pc + 0.1 * loss_sff + 0.2 * loss_nv

        # เพิ่ม Rank-aware Loss สำหรับ BSD Module
        if mode == 'bsd' and r > 0:
            K = min(50, spacings.shape[1])
            low_spacings = spacings[:, :K]
            target_frac = (r + 1) / K
            soft_frac = torch.sigmoid((1.0 - low_spacings) / 0.1).mean()
            rank_loss = (soft_frac - target_frac) ** 2
            total_loss = total_loss + 0.1 * rank_loss
            
            # Empirical loss หากมี target
            if target_spacings is not None:
                # การคำนวณ Empirical Target Loss ประยุกต์จาก bsd_one.py
                total_loss = total_loss + 0.5 * F.mse_loss(spacings, target_spacings)

        # Backpropagation
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
