---
title: Pandas
date: 2024-8-31
external_link: https://github.com/pandas-dev/pandas
tags:
  - Hugo
  - Wowchemy
  - Markdown
---
### **Diffusion模型（DDPM）简介**

1. **模型的基本思想**：

   - **正向扩散过程**：DDPM 将数据逐步添加噪声，最终将数据转换为纯噪声。这个过程是预定义的、不可学习的。具体来说，论文中是使用的一系列从小到大的beta， beta从小到大(在DDPM中是线性增长的，但是在openai的论文中也使用了cosine的增长是)，表示在前向过程中加噪声的权重，具体的公式是

     ?        
     $$
     X_t = \sqrt{1-\beta_t}*x_{t-1}+\sqrt{\beta_t}*?_t
     $$
     ?    beta是被事先固定的，又可以定义alpha_t = 1- beta_t，?_t是正态分布中获得的噪声
     $$
     X_t = \sqrt{1-\beta_t}*x_{t-1}+\sqrt{\beta_t}*?_t\\
     =\sqrt{1-\beta_t}*\sqrt{1-\beta_{t-1}}*x_{t-2}+\sqrt{1-\beta_t}\sqrt{\beta_{t-1}}*?_{t-1}+\sqrt{\beta_t}*?_t
     $$
     

   - **逆向去噪过程**：通过学习一个神经网络，该模型能够逐步去除噪声，从纯噪声中恢复出原始数据。这个过程是可学习的，并通过训练来优化。

2. **工作原理**：

   - **正向过程**：从数据分布（如图像）出发，在每一步都添加少量的高斯噪声，使数据逐渐接近标准正态分布。
   - **逆向过程**：训练一个神经网络，逐步预测和移除噪声，以恢复数据。这是通过最小化与真实数据的去噪差距来实现的。

3. **损失函数**：

   - **重构损失**：通过最小化每一步去噪结果与原始无噪声数据之间的差距来优化模型。
   - **KL 散度**：用于评估模型生成的分布与真实分布之间的差异。

4. **DDPM 的优势**：

   - **高生成质量**：生成的数据质量较高，尤其是在图像生成任务中表现突出。
   - **灵活性**：可以用于各种数据类型（如图像、音频），并且容易与其他模型结合。

5. **训练与生成**：

   - **训练阶段**：训练模型去学习如何从噪声中恢复原始数据。
   - **生成阶段**：通过反向的去噪步骤，从随机噪声开始，逐步生成符合数据分布的样本。

# Norm 函数

?	在不同的任务中常常看到许多的Norm函数，他们本来是为了做不同任务的，但是在最近的神经网络中，可以看到许多他们分别作用的身影。他们的本质都是减去均值除以方差，转化为均值为0，方差为1的分布，然后做一个线性映射，只不过是作用的范围不同。在此总结一下他们不同的特点，以及使用范围

![img](https://i-blog.csdnimg.cn/blog_migrate/b31e2f2e936ce1b1baf08e5bb5a3ff0c.png)

## Batch_Norm

?	BN的适用范围是单个channel的一个Batch做归一化。如上图，一个chanel的每一个batch都参与归一化。假设有N本书，每本书有C页，每页可容纳HxW个字符，Batch Norm就是页为单位：假设每本书都为C页，首先计算N本书中第1页的字符`【N, H, W】`均值方差，得到统计量`u1，δ1`，然后对N本书的第一页利用该统计量对第一页的元素进行归一化操作，剩下的C-1页同理。

?	特点：加快训练速度，使得模型训练更稳定，避免梯度爆炸或者梯度消失，并且有一定的正则化作用。

?	适用范围：大量数据，batch_size比较大的时候效果比较好，适用范围比较广。

## Layer_Norm

?	上面说过，当Batchsize比较小，甚至为1的时候，效果就不太好，甚至没有作用（当size==1）.Layer_norm则是单个batch的所有channel做归一化。

?	还是同样的例子，有N本书，每本书有C页，每页可容纳HxW个字符，Layer Norm就是以本为单位：首先计算第一本书中的所有字符`【H, W, C】`均值方差，得到统计量`u1，δ1`，然后利用该统计量对第一本数进行归一化操作，剩下的N-1本书同理。

?	特点：对于当channel没有特别意义的时候，或者channel为1的时候，batch需要独立的时候效果比较好

?	适用范围： 主要对RNN(处理序列)作用明显，目前大火的Transformer也是使用的这种归一化操作；

## Instacnce_Norm

?	这种归一化方法最初用于图像的风格迁移。其作者发现，在生成模型中， feature map 的各个 channel 的均值和方差会影响到最终生成图像的风格，因此可以先把图像在 channel 层面归一化，然后再用目标风格图片对应 channel 的均值和标准差“去归一化”，以期获得目标图片的风格。本质是对单个channel的单个样本内做归一化。(如果用所有样本就会影响单个图片的风格)

?	还是同样的例子，有N本书，每本书有C页，每页可容纳HxW个字符，Instance Norm就是以每本书的每一页为单位：首先计算第1本书中第1页的所有字符`【H, W】`均值方差，得到统计量`u1，δ1`，然后利用该统计量对第1本书第1页进行归一化操作，剩下的NC-1页同理。

?	特点：加速模型收敛，保持各个图像之间的独立。

?	使用范围：适用于channel，batch都有独特意义的任务，比如图像风格化迁移。

## Group_Norm

?	其在计算均值和标准差时，先把每一个样本feature map的 channel 分成 G 组，每组将有 C/G 个 channel，然后将这些 channel 中的元素求均值和标准差。各组 channel 用其对应的归一化参数独立地归一化。

?	还是同样的例子，有N本书，每本书有C页，每页可容纳HxW个字符，Group Norm就是以每本书的G页为单位：首先计算第1本书中第1组G页中的所有字符`【H, W, G】`均值方差，得到统计量`u1，δ1`，然后利用该统计量对第1本书第1组G页进行归一化操作，剩下的`NC/G-1`组同理。

?	特点：与layer_norm比较类似，但是需要算力显存更小

?	适用范围：需要显存比较大的任务，比如图像分割与检测。

***

## Residual Block

```python
class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, time_channels, n_groups, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1))
        
        self.norm2 = nn.GroupNorm(n_groups, in_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1))
        if in_channels != out_channels:
            #consider to map the feature size of the input and output
            self.shortcut = nn.Conv2d(in_channels , out_channels, kernel_size = (1,1))
        else:
            self.shortcut = nn.Identity()
            
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()
	    self.dropout = nn.Dropout(dropout)
    def forward(self, x, torch.Tensor, t:torch.Tensor):
        h = self.conv1(self.act1(self.norm1(x)))
        #define the last two dims to be none to correspond to the featuere map
        #this is a boardcast by torch, which transform those dim=1 to map other tensor
        h += self.time_emb(self.time_act(t))[:, :, None, None]#actualy transto[:,:,1,1]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x) #connect the input
```

?	这个代码定义了一个 `ResidualBlock` 类，它是一个典型的残差块（Residual Block），用于神经网络的深度学习模型中，特别是在时间依赖或扩散模型中使用。这段代码实现的残差块结合了标准的卷积操作、归一化、激活函数、时间嵌入和跳跃连接（shortcut connection）。

?	其中时间向量的嵌入使用了torch的广播功能，能匹配特征图的长宽大小，将时间信息加入到每一个特征图的像素上。

## Self-Attention

?	DDPM的diffuson模型受到transformer的影响，在U-net中加入了self-attention的模块，后来的U-net甚至出现了transformer为基础的模型。

?	Self-attention可以做到序列内的注意力权重计算，具体的思想和数据库的q,k,v比较类似，模块会通过线性层或者卷积层，输入x产生三个矩阵,W_q,W_k,W_v，这三个矩阵和x相乘分别获得q,k,v，attention_value的计算公式是: 
$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d}}V)
$$
?	其中d是特征图的长宽维度

?	一般常用的是多头注意力，以表现平均意义上的注意力权重,n_heads * d_k = n_channels,n_channels才是输入的特征维度

```python
class AttentionBlock(Module):
	def __init__(self, n_channels, n_heads, d_k, n_groups):
        super().__init__()
        if d_k is None: #if not defined, then get a single head attn
            d_k = n_channels
        self.d_k =d_k
        self.scale = d_k**-0.5
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.linear(n_channels, n_heads*d_k*3)
     
    def forward(self, x:torch.Tensor, t:Optional[torch.Tensor] = None):
        '''
        	x in shape[batch_size, in_channels, H, W]
        	t in shape[batch_size, time_channels]
        '''
        _ = t
        batch_size, n_channels, H, W = x.shape #get the shape
        #trans x to shape in [batch_size, seq, n_channels],seq means H*W
        x = x.view(batch_size, n_channels, -1).permute(0,2,1)
        # gey g,k,v, in shape [batch_size, seq, n_heads, d_k]
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3*self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        attn = torch.einsm('bihd,bjhd->bijh', q, k)*self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsm('bijh,bjhd->bikd', attn, v)
        res = res.view(batch_size, -1,self.n_heads*self.d_k)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res
        
```

## DownBlock

?	包含了残差块，和注意力块，被用于U-Net的每一个水平层

```python
class DownBlock(Module):
    '''
    	use the attn is a optional function
    '''
	def __init__(self, in_channels, out_channels, time_channels, has_attn:bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
    def forward(self, x, t):
        x = self.res(x,t)
        x = self.attn(x)
        return x
```

## UpBlock

?	基本和DownBlock一致，但是由于向下采样的时候忽略了边缘特征，所以在跳跃连接的时候，拼接了之前的输出，可以看虚线箭头

```python
class DownBlock(Module):
    '''
    	use the attn is a optional function
    	the residual input is input+output?
    '''
	def __init__(self, in_channels, out_channels, time_channels, has_attn:bool):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
    def forward(self, x, t):
        x = self.res(x,t)
        x = self.attn(x)
        return x
```

## MiddleBlock

?	基本结构是Res_block+attn_block+Res_block,作用在整个U-Net最底部的水平层

```python
class MiddleBlock(Module):
	def __init__(self, n_channels, time_channels):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)
    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x,t)
        return x
```

## Down_Sample

?	这一部分是向下采样的部分，将特征减少为原来的1/2

```python
class Downsample(nn.module):
    def __init__(self, n_channels):
        super __init__()
        # k = 3, padding =1 ,stride = 2
    	self.conv = nn.Conv2d(n_channels, n_channels, (3,3),(2,2),(1,1))
    def forward(self,x, t):
        _ = t
        return self.conv(x)

```

## Up_Sample

?	同理，这一部分是向上采样的部分，使用转置卷积来实现，回忆转置卷积的性质。

注意torch.Conv2d和torch.ConvTranspose2d的参数顺序是一样的，都是kernel,stride,padding
$$
h' = (h-1)*s+k-2*padding
$$

```python
class Upsample(Module):
	def __init__(self, n_channels):
		super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))\
    def forward(self, x, t):
        _ = t
        return self.conv(x)
```



# U-Net

?	U-Net 是diffusion模型中预测噪声的模块，预测出模型后可以通过DDPM论文中推导得出的公式来预测图像。U-Net本身是源自图像分割领域的，这与预测噪声实际上有着共通的地方，因为图像分割，可以被理解为分为前景，背景或者更多的图层，然后预测噪声的话，实际上就是面向噪声来图像分割。

## 模块形状

![img](https://nn.labml.ai/unet/unet.png)

?	模块可以被分为3个部分，分别是向下的箭头，被称为down-sample，向下采样，或者是**编码器encoder**,实际上是图像的长宽维度在减小，维度在增加，浓缩图像的特征，这一步一般使用卷积神经网络来进行；然后是向上的箭头，被称为up-sample， 向上采样，**编码器decoder**,实际上是将图像由小的浓缩特征图复原到原来的尺度，这一步一般是通过插值或者卷积转置的方法来进行。最后是横着的箭头，被称为**跳跃连接，（skip connection）**。这一步连接了编码器和解码器，我们注意到在卷积编码的过程中一些边缘的信息会被卷积核忽略，所以这一步需要将忽略的信息补回，注意这时候是很玄学的一个部分，看上图编码器输出的特征图通道数比解码器输出的要小一倍，特征维度要大一点，DDPM论文中是将编码器的维度裁剪（虚线），然后键入到编码器输出的一半的通道上，然后做卷积处理，缩小一半通道（让没有被加的通道也扯上关系）。以此类推直到图片还原。

## 代码实现

准备工作，定义激活函数,原文使用了Swish函数（x*sigmoid(x)）

```python
class Swish(module):
    def forward(self,x):
        return x*torch.sigmoid(x) 
```

考虑到噪声是一个t相关的函数，所以要对时间做一个embedding,使用的是transformer的positional_embedding

```python
class nn.TimeEmbedding(nn.Module):
	def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels//4, self.n_channels)
        self.act = Swish() #activate function
        self.lin2 = nn.Linear(self.n_channel, self.n_channels)
    def forward(self, t:torch.Tensor):
        half_dim = self.n_channels //8
        emb = math.log(10_000) / (half_dim-1)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb,cos()), dim=1)
        emb = self.act(self.lin1(emb))                                                         emb = self.lin2(emb)	
		return emb #time emb
		
```


<!--more-->
