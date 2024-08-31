---
title: Pandas
date: 2024-8-31
external_link: https://github.com/pandas-dev/pandas
tags:
  - Hugo
  - Wowchemy
  - Markdown
---
### **Diffusionģ�ͣ�DDPM�����**

1. **ģ�͵Ļ���˼��**��

   - **������ɢ����**��DDPM ��������������������ս�����ת��Ϊ�����������������Ԥ����ġ�����ѧϰ�ġ�������˵����������ʹ�õ�һϵ�д�С�����beta�� beta��С����(��DDPM�������������ģ�������openai��������Ҳʹ����cosine��������)����ʾ��ǰ������м�������Ȩ�أ�����Ĺ�ʽ��

     ?        
     $$
     X_t = \sqrt{1-\beta_t}*x_{t-1}+\sqrt{\beta_t}*?_t
     $$
     ?    beta�Ǳ����ȹ̶��ģ��ֿ��Զ���alpha_t = 1- beta_t��?_t����̬�ֲ��л�õ�����
     $$
     X_t = \sqrt{1-\beta_t}*x_{t-1}+\sqrt{\beta_t}*?_t\\
     =\sqrt{1-\beta_t}*\sqrt{1-\beta_{t-1}}*x_{t-2}+\sqrt{1-\beta_t}\sqrt{\beta_{t-1}}*?_{t-1}+\sqrt{\beta_t}*?_t
     $$
     

   - **����ȥ�����**��ͨ��ѧϰһ�������磬��ģ���ܹ���ȥ���������Ӵ������лָ���ԭʼ���ݡ���������ǿ�ѧϰ�ģ���ͨ��ѵ�����Ż���

2. **����ԭ��**��

   - **�������**�������ݷֲ�����ͼ�񣩳�������ÿһ������������ĸ�˹������ʹ�����𽥽ӽ���׼��̬�ֲ���
   - **�������**��ѵ��һ�������磬��Ԥ����Ƴ��������Իָ����ݡ�����ͨ����С������ʵ���ݵ�ȥ������ʵ�ֵġ�

3. **��ʧ����**��

   - **�ع���ʧ**��ͨ����С��ÿһ��ȥ������ԭʼ����������֮��Ĳ�����Ż�ģ�͡�
   - **KL ɢ��**����������ģ�����ɵķֲ�����ʵ�ֲ�֮��Ĳ��졣

4. **DDPM ������**��

   - **����������**�����ɵ����������ϸߣ���������ͼ�����������б���ͻ����
   - **�����**���������ڸ����������ͣ���ͼ����Ƶ������������������ģ�ͽ�ϡ�

5. **ѵ��������**��

   - **ѵ���׶�**��ѵ��ģ��ȥѧϰ��δ������лָ�ԭʼ���ݡ�
   - **���ɽ׶�**��ͨ�������ȥ�벽�裬�����������ʼ�������ɷ������ݷֲ���������

# Norm ����

?	�ڲ�ͬ�������г�����������Norm���������Ǳ�����Ϊ������ͬ����ģ�������������������У����Կ���������Ƿֱ����õ���Ӱ�����ǵı��ʶ��Ǽ�ȥ��ֵ���Է��ת��Ϊ��ֵΪ0������Ϊ1�ķֲ���Ȼ����һ������ӳ�䣬ֻ���������õķ�Χ��ͬ���ڴ��ܽ�һ�����ǲ�ͬ���ص㣬�Լ�ʹ�÷�Χ

![img](https://i-blog.csdnimg.cn/blog_migrate/b31e2f2e936ce1b1baf08e5bb5a3ff0c.png)

## Batch_Norm

?	BN�����÷�Χ�ǵ���channel��һ��Batch����һ��������ͼ��һ��chanel��ÿһ��batch�������һ����������N���飬ÿ������Cҳ��ÿҳ������HxW���ַ���Batch Norm����ҳΪ��λ������ÿ���鶼ΪCҳ�����ȼ���N�����е�1ҳ���ַ�`��N, H, W��`��ֵ����õ�ͳ����`u1����1`��Ȼ���N����ĵ�һҳ���ø�ͳ�����Ե�һҳ��Ԫ�ؽ��й�һ��������ʣ�µ�C-1ҳͬ��

?	�ص㣺�ӿ�ѵ���ٶȣ�ʹ��ģ��ѵ�����ȶ��������ݶȱ�ը�����ݶ���ʧ��������һ�����������á�

?	���÷�Χ���������ݣ�batch_size�Ƚϴ��ʱ��Ч���ȽϺã����÷�Χ�ȽϹ㡣

## Layer_Norm

?	����˵������Batchsize�Ƚ�С������Ϊ1��ʱ��Ч���Ͳ�̫�ã�����û�����ã���size==1��.Layer_norm���ǵ���batch������channel����һ����

?	����ͬ�������ӣ���N���飬ÿ������Cҳ��ÿҳ������HxW���ַ���Layer Norm�����Ա�Ϊ��λ�����ȼ����һ�����е������ַ�`��H, W, C��`��ֵ����õ�ͳ����`u1����1`��Ȼ�����ø�ͳ�����Ե�һ�������й�һ��������ʣ�µ�N-1����ͬ��

?	�ص㣺���ڵ�channelû���ر������ʱ�򣬻���channelΪ1��ʱ��batch��Ҫ������ʱ��Ч���ȽϺ�

?	���÷�Χ�� ��Ҫ��RNN(��������)�������ԣ�Ŀǰ����TransformerҲ��ʹ�õ����ֹ�һ��������

## Instacnce_Norm

?	���ֹ�һ�������������ͼ��ķ��Ǩ�ơ������߷��֣�������ģ���У� feature map �ĸ��� channel �ľ�ֵ�ͷ����Ӱ�쵽��������ͼ��ķ����˿����Ȱ�ͼ���� channel �����һ����Ȼ������Ŀ����ͼƬ��Ӧ channel �ľ�ֵ�ͱ�׼�ȥ��һ���������ڻ��Ŀ��ͼƬ�ķ�񡣱����ǶԵ���channel�ĵ�������������һ����(��������������ͻ�Ӱ�쵥��ͼƬ�ķ��)

?	����ͬ�������ӣ���N���飬ÿ������Cҳ��ÿҳ������HxW���ַ���Instance Norm������ÿ�����ÿһҳΪ��λ�����ȼ����1�����е�1ҳ�������ַ�`��H, W��`��ֵ����õ�ͳ����`u1����1`��Ȼ�����ø�ͳ�����Ե�1�����1ҳ���й�һ��������ʣ�µ�NC-1ҳͬ��

?	�ص㣺����ģ�����������ָ���ͼ��֮��Ķ�����

?	ʹ�÷�Χ��������channel��batch���ж�����������񣬱���ͼ����Ǩ�ơ�

## Group_Norm

?	���ڼ����ֵ�ͱ�׼��ʱ���Ȱ�ÿһ������feature map�� channel �ֳ� G �飬ÿ�齫�� C/G �� channel��Ȼ����Щ channel �е�Ԫ�����ֵ�ͱ�׼����� channel �����Ӧ�Ĺ�һ�����������ع�һ����

?	����ͬ�������ӣ���N���飬ÿ������Cҳ��ÿҳ������HxW���ַ���Group Norm������ÿ�����GҳΪ��λ�����ȼ����1�����е�1��Gҳ�е������ַ�`��H, W, G��`��ֵ����õ�ͳ����`u1����1`��Ȼ�����ø�ͳ�����Ե�1�����1��Gҳ���й�һ��������ʣ�µ�`NC/G-1`��ͬ��

?	�ص㣺��layer_norm�Ƚ����ƣ�������Ҫ�����Դ��С

?	���÷�Χ����Ҫ�Դ�Ƚϴ�����񣬱���ͼ��ָ����⡣

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

?	������붨����һ�� `ResidualBlock` �࣬����һ�����͵Ĳв�飨Residual Block������������������ѧϰģ���У��ر�����ʱ����������ɢģ����ʹ�á���δ���ʵ�ֵĲв�����˱�׼�ľ����������һ�����������ʱ��Ƕ�����Ծ���ӣ�shortcut connection����

?	����ʱ��������Ƕ��ʹ����torch�Ĺ㲥���ܣ���ƥ������ͼ�ĳ����С����ʱ����Ϣ���뵽ÿһ������ͼ�������ϡ�

## Self-Attention

?	DDPM��diffusonģ���ܵ�transformer��Ӱ�죬��U-net�м�����self-attention��ģ�飬������U-net����������transformerΪ������ģ�͡�

?	Self-attention�������������ڵ�ע����Ȩ�ؼ��㣬�����˼������ݿ��q,k,v�Ƚ����ƣ�ģ���ͨ�����Բ���߾���㣬����x������������,W_q,W_k,W_v�������������x��˷ֱ���q,k,v��attention_value�ļ��㹫ʽ��: 
$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d}}V)
$$
?	����d������ͼ�ĳ���ά��

?	һ�㳣�õ��Ƕ�ͷע�������Ա���ƽ�������ϵ�ע����Ȩ��,n_heads * d_k = n_channels,n_channels�������������ά��

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

?	�����˲в�飬��ע�����飬������U-Net��ÿһ��ˮƽ��

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

?	������DownBlockһ�£������������²�����ʱ������˱�Ե��������������Ծ���ӵ�ʱ��ƴ����֮ǰ����������Կ����߼�ͷ

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

?	�����ṹ��Res_block+attn_block+Res_block,����������U-Net��ײ���ˮƽ��

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

?	��һ���������²����Ĳ��֣�����������Ϊԭ����1/2

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

?	ͬ����һ���������ϲ����Ĳ��֣�ʹ��ת�þ����ʵ�֣�����ת�þ�������ʡ�

ע��torch.Conv2d��torch.ConvTranspose2d�Ĳ���˳����һ���ģ�����kernel,stride,padding
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

?	U-Net ��diffusionģ����Ԥ��������ģ�飬Ԥ���ģ�ͺ����ͨ��DDPM�������Ƶ��ó��Ĺ�ʽ��Ԥ��ͼ��U-Net������Դ��ͼ��ָ�����ģ�����Ԥ������ʵ�������Ź�ͨ�ĵط�����Ϊͼ��ָ���Ա����Ϊ��Ϊǰ�����������߸����ͼ�㣬Ȼ��Ԥ�������Ļ���ʵ���Ͼ�������������ͼ��ָ

## ģ����״

![img](https://nn.labml.ai/unet/unet.png)

?	ģ����Ա���Ϊ3�����֣��ֱ������µļ�ͷ������Ϊdown-sample�����²�����������**������encoder**,ʵ������ͼ��ĳ���ά���ڼ�С��ά�������ӣ�Ũ��ͼ�����������һ��һ��ʹ�þ�������������У�Ȼ�������ϵļ�ͷ������Ϊup-sample�� ���ϲ�����**������decoder**,ʵ�����ǽ�ͼ����С��Ũ������ͼ��ԭ��ԭ���ĳ߶ȣ���һ��һ����ͨ����ֵ���߾��ת�õķ��������С�����Ǻ��ŵļ�ͷ������Ϊ**��Ծ���ӣ���skip connection��**����һ�������˱������ͽ�����������ע�⵽�ھ������Ĺ�����һЩ��Ե����Ϣ�ᱻ����˺��ԣ�������һ����Ҫ�����Ե���Ϣ���أ�ע����ʱ���Ǻ���ѧ��һ�����֣�����ͼ���������������ͼͨ�����Ƚ����������ҪСһ��������ά��Ҫ��һ�㣬DDPM�������ǽ���������ά�Ȳü������ߣ���Ȼ����뵽�����������һ���ͨ���ϣ�Ȼ�������������Сһ��ͨ������û�б��ӵ�ͨ��Ҳ���Ϲ�ϵ�����Դ�����ֱ��ͼƬ��ԭ��

## ����ʵ��

׼�����������弤���,ԭ��ʹ����Swish������x*sigmoid(x)��

```python
class Swish(module):
    def forward(self,x):
        return x*torch.sigmoid(x) 
```

���ǵ�������һ��t��صĺ���������Ҫ��ʱ����һ��embedding,ʹ�õ���transformer��positional_embedding

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
