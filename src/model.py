import torch
import torch.nn as nn
import torchvision.models as models

class NeuralNavigator(nn.Module):
    def __init__(
        self,
        vocab_size=50,
        text_embed_dim=32,
        text_hidden_dim=64,
        max_seq_len=20
    ):
        super().__init__()

        # 🔹 Image Encoder (CNN)
        backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Identity()
        self.image_encoder = backbone
        self.image_feat_dim = 512

        # 🔹 Text Encoder
        self.embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.text_encoder = nn.LSTM(
            text_embed_dim,
            text_hidden_dim,
            batch_first=True
        )

        # 🔹 Fusion + Decoder
        fused_dim = self.image_feat_dim + text_hidden_dim
        self.decoder = nn.LSTM(
            fused_dim,
            128,
            batch_first=True
        )

        self.output_head = nn.Linear(128, 2)
        self.max_seq_len = max_seq_len

    def forward(self, images, text_tokens):
        """
        images: [B, 3, 128, 128]
        text_tokens: [B, T]
        """

        # 🔹 Image features
        img_feat = self.image_encoder(images)      # [B, 512]

        # 🔹 Text features
        embeds = self.embedding(text_tokens)       # [B, T, E]
        _, (h_n, _) = self.text_encoder(embeds)
        text_feat = h_n[-1]                         # [B, H]

        # 🔹 Fusion
        fused = torch.cat([img_feat, text_feat], dim=1)  # [B, 512+H]
        fused = fused.unsqueeze(1).repeat(1, self.max_seq_len, 1)

        # 🔹 Decode path
        decoded, _ = self.decoder(fused)
        path = self.output_head(decoded)            # [B, 20, 2]

        return path
