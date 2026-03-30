import torch
import torch.nn as nn

class GeneTransformerHead(nn.Module):
    """
    Cross‐attention head: each of `num_genes` learnable query vectors
    attends over the H*W spatial tokens of the fused backbone output.
    """
    def __init__(
        self,
        feat_dim: int,
        num_genes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        # project each spatial token from feat_dim → d_model
        self.mem_proj = nn.Linear(feat_dim, d_model)
        # learnable gene‐queries: (num_genes, d_model)
        self.query_embed = nn.Parameter(torch.randn(num_genes, d_model))
        # stack of Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # final per‐gene scalar projection
        self.to_logits = nn.Linear(d_model, 1)

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """
        bottleneck: [B, feat_dim, H, W]  (H=W=7 here)
        returns:    [B, num_genes]
        """
        B, feat_dim, H, W = bottleneck.shape
        # 1) flatten spatial → (B, feat_dim, S) where S = H*W
        S = H * W
        mem = bottleneck.view(B, feat_dim, S).permute(2, 0, 1)  # [S, B, feat_dim]
        # 2) project to model-dim
        mem = self.mem_proj(mem)                                # [S, B, d_model]

        # 3) prepare queries: (T=num_genes, B, d_model) 
        T = self.query_embed.shape[0]
        q = self.query_embed.unsqueeze(1).expand(T, B, -1)

        # 4) Transformer decoder
        out = self.decoder(tgt=q, memory=mem)                   # [T, B, d_model]

        # 5) to (B, T, d_model)
        out = out.permute(1, 0, 2)

        # 6) project each gene‐query to a scalar
        return self.to_logits(out).squeeze(-1)                  # [B, num_genes]

class CITGenePredictor(nn.Module):
    """
    Wraps the existing CiT-Net-Tiny backbone for spot-level gene expression regression.
    Takes a 224×224 RGB patch and predicts a vector of length num_genes.
    """
    def __init__(self, cit_model: nn.Module, num_genes: int):
        super().__init__()
        self.cit = cit_model
        
        # freeze segmentation head if present
        # define a new regression head: fuse Cnn4 & Swin4 features
        # embed_dim=96 => stage4 channels = embed_dim * 8 = 768 each
        fused_ch = self.cit.embed_dim * 8 * 2  # Cnn4 + Swin4
        self.reg_head = nn.Sequential(
             nn.AdaptiveAvgPool2d((1,1)),    # [B, fused_ch, 1,1]
             nn.Flatten()    ,               # [B, fused_ch]
            nn.Linear(fused_ch, num_genes)   # [B, num_genes]
         )
        self.flatten = nn.Flatten()
        self.head = GeneTransformerHead(
            feat_dim=fused_ch,
            num_genes=num_genes,
            d_model=256,
            nhead=8,
            num_layers=2,
        )
       

    def forward(self, x):
        # forward through CiT-Net up to bottleneck
        # we need to extract Cnn4 and Swin4
        # repurpose the beginning of CIT.forward
        # assume CIT.forward returns (CiT_mask, CNN_out, Trans_out) by default
        # so we’ll instead call internal layers manually
        # Here we duplicate CIT.forward logic up to Cnn4/Swin4

        # initial patch & split
        x0 = self.cit.patch(x)
        Cnn = self.cit.Conv1e(x0)
        Swin = self.cit.Conv1s(x0)
        Cnn = self.cit.maxpool(Cnn)
        Cnn = self.cit.Conv2e(Cnn)
        Swin = self.cit.layer1(Swin)
        Cnn = self.cit.maxpool(Cnn)
        Cnn = self.cit.Conv3e(Cnn)
        Swin = self.cit.layer2(Swin)
        Cnn = self.cit.maxpool(Cnn)
        Cnn4 = self.cit.Conv4e(Cnn)
        Swin4 = self.cit.layer3(Swin)
        Swin4 = self.cit.layer4(Swin4)
        Swin4 = self.cit.norm(Swin4)

        # fuse bottle-neck features
        bottleneck = torch.cat((Cnn4, Swin4), dim=1)  # [B, fused_ch, 7,7]
        # gene expression regression
        expr = self.head(bottleneck)  # [B, num_genes]
        return expr
    #expr

# Example instantiation:
# base_cit = CIT(img_size=224, in_chans=3, embed_dim=96, ...)  # your trained backbone
# model = CITGenePredictor(base_cit, num_genes=1000)
# x = torch.randn(4,3,224,224)\# batch of 4 patches
# preds = model(x)  # [4,1000] gene expression predictions
