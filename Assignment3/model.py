from turtle import forward
import torch 
from torch import embedding
import torch.nn as nn
import math
import copy


class Embeddings(nn.Module):
    """
    对图像编码, 也就是把图片看作一个句子分割成块, 每一块表示一个单词
    """
    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        img_size = img_size
        patch_size = config.patches['size']  # 16

        # 将图像分成多少块 (224/16) * (224/16) = 196
        n_patches = (img_size//patch_size) * (img_size//patch_size)

        # 对图片进行卷积获取图片的块，并将每一块映射城config.hidden_size维(768)
        self.patch_embeddings = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=config.hidden_size,
                    kernel_size=patch_size,
                    stride=patch_size
                )
        
        # 设置可学习的位置编码信息 (1, 196+1, 786)
        self.position_embeddings=nn.Parameter( torch.zeros(1, n_patches+1, config.hidden_size))

        # 设置可学习的分类信息的维度
        self.classifer_token = nn.Parameter( torch.zeros(1, 1, config.hidden_size))
        self.dropout = nn.Dropout((config.transformer['dropout_rate']))

    def forward(self, x):
        bs = x.shape[0]
        cls_tokens = (self.classifer_token.expand(bs, -1, -1)).expand(bs, 1, 768)
        x = self.patch_embeddings(x)  # (bs, 768, 14, 14)
        x = x.flatten(2)  # (bs, 768, 196)
        x = x.transpose(-1, -2)  # (bs, 196, 768)
        x = torch.cat((cls_tokens, x), dim=1)  # 将分类信息和图片块拼接 (bs, 197, 768)
        embeddings = x + self.position_embeddings # 将图片块信息和对应位置信息相加 (bs, 197, 768)
        embeddings = self.dropout(embeddings)

        return embeddings


### 构建self-Attention 模块
class Attention(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer['num_heads']  # 12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768

        self.query = nn.Linear(config.hidden_size, self.all_head_size)  # 768->768，Wq矩阵为（768,768）
        self.key = nn.Linear(config.hidden_size, self.all_head_size)  # 768->768,Wk矩阵为（768,768）
        self.value = nn.Linear(config.hidden_size, self.all_head_size)  # 768->768,Wv矩阵为（768,768）
        self.out = nn.Linear(config.hidden_size, config.hidden_size)  # 768->768
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
                self.num_attention_heads, self.attention_head_size)  # (bs,197)+(12,64)=(bs,197,12,64)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (bs,12,197,64)

    def forward(self, hidden_states):
        # hidden_states为：(bs,197,768)
        mixed_query_layer = self.query(hidden_states)  # 768->768
        mixed_key_layer = self.key(hidden_states)  # 768->768
        mixed_value_layer = self.value(hidden_states)  # 768->768

        query_layer = self.transpose_for_scores(mixed_query_layer)  # wm，(bs,12,197,64)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # 将q向量和k向量进行相乘（bs,12,197,197)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # 将结果除以向量维数的开方
        attention_probs = self.softmax(attention_scores)  # 将得到的分数进行softmax,得到概率
        weights = attention_probs if self.vis else None  # 实际上就是权重
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # 将概率与内容向量相乘
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # (bs,197)+(768,)=(bs,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights  # (bs,197,768),(bs,197,197)


### 构建前向传播网络
# 两个全连接层中间加激活函数
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer['mlp_dim'])  #  786 -> 3072
        self.fc2 = nn.Linear(config.transformer['mlp_dim'], config.hidden_size)  #  3072->786
        self.act_fn = torch.nn.functional.gelu  # 激活函数
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)  # 786->3072
        x = self.act_fn(x)  # 激活函数
        x = self.dropout(x)  # 丢弃
        x = self.fc2(x)  # 3072->786
        x = self.dropout(x)
        return x


#4.构建编码器的可重复利用的Block()模块：每一个block包含了self-attention模块和MLP模块
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size  # 768
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)  # 层归一化
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        self.ffn = MLP(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h  # 残差结构

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h  # 残差结构
        return x, weights


#5.构建Encoder模块，该模块实际上就是堆叠N个Block模块
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


#6构建transformers完整结构，首先图片被embedding模块编码成序列数据，然后送入Encoder中进行编码
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)  # 对一幅图片进行切块编码，得到的是（bs,n_patch+1（196）,每一块的维度（768））
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)  # 输出的是（bs,196,768)
        encoded, attn_weights = self.encoder(embedding_output)  # 输入的是（bs,196,768)
        return encoded, attn_weights  # 输出的是（bs,197,768）


#7构建VisionTransformer，用于图像分类
class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=100, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = nn.Linear(config.hidden_size, num_classes)  # 768-->100

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        #如果传入真实标签，就直接计算损失值
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

