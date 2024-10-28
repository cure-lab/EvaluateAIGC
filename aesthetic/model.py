import torch.nn as nn
import torch, torchvision
import torchvision.models as models
import open_clip

clip_length = 768
resnet_feature_length = 512

class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(in_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        in: batch * in_dim
        out: batch * num_class
        """
        return self.softmax(self.linear(self.dropout(self.layernorm(x))))

class RegressionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.75)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        in: batch * in_dim
        out: batch * num_class
        """
        return self.linear(self.dropout(x))
    
class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, query=None, pos_emb=None):
        x2 = self.norm1(x)
        q = k = self.with_pos_embed(x2, pos_emb)
        if query is not None: # cross attention
            q = self.with_pos_embed(query, pos_emb)
        x2 = self.self_attn(q, k, value=x2)[0]
        x = x + self.dropout1(x2)

        # feed forward
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout2(x2)
        return x

class Transformer(nn.Module):
    def __init__(self, in_dim, n_layers=4):
        super().__init__()
        self.in_dim = in_dim
        self.cross_attentions = nn.Sequential(*[AttentionLayer(embed_dim=in_dim, nhead=2 ** min(3, i)) 
                                    for i in range(n_layers)])

    def forward(self, feature1, feature2=None) -> torch.Tensor:
        for layer in self.cross_attentions:
            feature = layer(feature1, feature2)
        return feature
    
class AttributeEncoder(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.num_attributes = opts["num_attributes"]
        feature_net = models.resnet34(weights=torchvision.models.resnet.ResNet34_Weights.DEFAULT).cuda()
        self.feature_net = nn.Sequential(*list(feature_net.children())[:-1])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        in: batch * c * h * w
        out: batch * out_dim * num_attributes
        """
        features = self.feature_net(image).squeeze(-1).squeeze(-1)
        return features

class AestheticModel(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.encoder = AttributeEncoder(opts)
        self.aesthetic_feature_layer = Transformer(resnet_feature_length + clip_length, opts["depth"])
        self.classifier = ClassificationHead(resnet_feature_length * 2, opts["num_distortions"])
        self.merge = nn.Linear(resnet_feature_length + clip_length, resnet_feature_length + clip_length)
        self.layernorm1 = nn.LayerNorm(512)
        self.layernorm2 = nn.LayerNorm(768)
        self.attr_linear = RegressionHead(resnet_feature_length + clip_length, opts["num_attributes"])
        self.linear = RegressionHead(resnet_feature_length + clip_length, 1)
    
    def forward(self, image, clip_feature):
        attribute_feature = self.encoder(image)
        attribute_feature = self.layernorm1(attribute_feature)
        clip_feature = self.layernorm2(clip_feature)
        attribute_feature = torch.cat([attribute_feature, clip_feature], dim=-1)
        attribute_feature = self.merge(attribute_feature)
        attribute_feature = self.aesthetic_feature_layer(attribute_feature)
        base_score = self.linear(attribute_feature)
        attribute_scores = self.attr_linear(attribute_feature)
        return base_score, attribute_scores