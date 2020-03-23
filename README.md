# \[Paper Reading\] RecSys 2019 Compositional Network Embedding for Link Prediction

> 导语：类深度语义匹配模型 DSSM \[1，2\] 的深度学习双塔模型，一方面可以利用近似近邻检索 ANN 算法实现对海量数据的在线实时检索，另一方面，可以在几乎不影响在线系统复杂度的情况下灵活开展大量离线策略调研优化工作，目前在工业界得到了广泛应用，尤其适用于搜索引擎、推荐系统和计算广告业务中的检索召回和排序模块，如 DSSM \[1，2\]，DNN4YouTube \[3\]。
>
> 2019 年，我们团队探索了基于深度学习双塔模型的预训练模型（User/Item Embedding），主要目的是特征压缩（如将 User 大量丰富的原始行为数据或画像标签压缩成低维稠密向量表示），以精简广告排序模型，达到模型效果、预测效率和机器资源之间的最佳平衡，这在工业界实际业务系统中非常非常关键，收效显著。
>
> 论文基于组合性原则 Compositionality 将深度学习双塔模型应用于网络表示学习（Network Embedding），创新提出了 Compositional Network Embedding \[4\]，简称 CNE，在多个 Link Prediction 任务中相比业界经典图神经网络 GNN 方法，如 DeepWalk、GraphSage 等，均取得了最新 STOA 的效果，论文已发表于 [RecSys 2019](https://recsys.acm.org/recsys19/accepted-contributions/#content-tab-1-1-tab)。
>
> CNE 和我们的预训练 User Embedding 工作，无论是设计初衷还是模型结构都比较类似，但 CNE 是一个更加通用的归纳式（Inductive）网络表示学习框架，具有更加广泛的应用前景。所以，在这里，特别介绍推荐给大家。
>
> 注：本文包括但不限于 CNE paper 所提及内容。

![](.gitbook/assets/image%20%282%29.png)

## **1. 网络 Network**

网络（Network）可以自然地表达对象与对象之间的关系，其中，对象又称节点（Node）或顶点（Vertex），关系用边（Edge）来描述，在现实世界中无处不在，是一种描述和建模复杂系统的通用“语言”，例如，Facebook、微信等社交媒体构成了人与人之间的社交网络（Social Network），互联网海量网页之间通过超链接构成了万维网（ Webpage Network），各种移动设备之间通信构成了通信网络 （Communication Network），国家城市间的公路、铁路、航空交通构成了物流运输网络 （Transportation/Traffic Network），文章或句子中的词语之间构成了词共现网络（Word Co-occurrence Network），海量用户的搜索查询、资讯浏览、视频播放、电商购物、App 下载、公众号订阅等构成了庞大的用户行为网络（User Behavior Network）， 等等。如下图所示 \[5\]。

![](.gitbook/assets/image%20%2814%29.png)

_注：AHG = Attributed Heterogeneous Graph_

数学形式上，我们习惯于将网络采用图数据结构（Graph）表示，记作 $$G=(V, E, X, Y)$$ \[6\]，其中，$$V$$是节点集合（同构/异构），$$|V|$$是节点总数，$$E \subseteq (V*V)$$是边的集合（有向/无向，带权/无权），边$$e_{ij}=(v_i, v_j) \in E$$表示节点$$v_i$$和$$v_j$$存在连边，边权重为$$w_{ij} \in R^+$$，$$X \in R^{|V|*m}$$是节点属性矩阵，$$m$$是属性维度，$$X_{ij}$$表示节点$$v_i$$的第$$j$$个属性特征，$$Y \in R^{|V|*|L|}$$是节点标签矩阵，$$L$$是标签集合，$$Y_{ik} \in [0, 1]$$表示节点$$v_i$$属于第$$k$$个标签的概率。

邻接矩阵$$A \in R^{|V|*|V|}$$是网络数据$$G$$中节点的一种最简单直接的语义表达形式，矩阵中的每行表示一个节点和其他所有节点的连接关系，可以看作是对应节点的一种表示，其中，$$A_{ij}=w_{ij}$$表示$$v_i$$和$$v_j$$之间的相似度。通常情况下，$$A$$是一个典型的高维稀疏矩阵，存储空间大，计算效率低，且无法表示潜在语义信息。因此，我们需要使用更加简单且有效的表达形式，网络表示学习无疑是目前最热门、最成功的研究方向。

## **2. 网络表示学习 Network Embedding**

网络表示学习（Network Embedding，或称 Graph Embedding，Network Representation Learning）旨在学习网络中所有节点的低维稠密向量表示，形式化地，就是对每个节点 $$v \in V$$ 学习一个实数向量$$R_v \in R^k$$，其中向量的维度 $$k$$ 远小于节点总数 $$|V|$$ ，网络表示学习可以看作衔接网络原始数据和网络下游应用任务之间的一座桥梁，已被证明在诸如节点分类、链接预测、社区发现、推荐系统和网络可视化等各种网络分析任务中非常有用，如下图所示 \[6\]。

![](.gitbook/assets/image%20%2810%29.png)

不妨先一起简要回顾一下已有的网络表示学习方法，主要围绕基于深度学习的方法。网络表示学习通过编码器 encoder 将每个节点 ID 映射到一个低维向量，根据 encoder 设计方式不同，可以将现有的网络表示方法划分为两大类：Non-compositional 和 Compositional 方法，前者将不可分割的数据（如节点 ID）作为输入，而后者需要信息的聚合作为输入。

### 2.1 第一类：Non-compositional 方法

早期的降维方法通常仅保留网络的低阶邻近性（Lower-Order Proximity），而低阶邻近性往往是非常稀疏的，与之不同的是，最近的基于深度学习的网络表示学习方法试图对高阶邻近性（Higher-Order Proximity）进行建模，它们都或多或少受到神经语言模型（Neural Language Model，NLM）和 Skip-Gram 等词向量技术的启发。

DeepWalk \[7\] 把图中节点看做单词，图上生成的随机游走序列看做句子，基于领域相似假设，将 Skip-Gram 模型应用于随机游走序列从而学习得到 Node Embedding，如下图所示 \[6\]。

![](.gitbook/assets/image%20%287%29.png)

DeepWalk 方法简单易懂，具有高可扩展性，方便并行化和在线学习，最重要的是，这是首次将深度学习技术引入到网络表示学习，将其与词向量 word2vec 联系在了一起，二者对比如下表所示 \[8\]。

| **Model** | **Target** | **Input** | **Output** |
| :--- | :--- | :--- | :--- |
| Word2vec | Words | Sentences | Word embeddings |
| DeepWalk | Nodes | Node sequences | Node embeddings |

受 DeepWalk 工作的启发，许多使用随机游走序列来学习 Node Embedding 的算法相继被提出，直观上，这些方法都是使用不同的随机游走策略来度量网络节点之间的邻近性，如 LINE \[9\]，Node2vec \[10\]，如下表所示 \[11\]。

| **Model** | **Published** | **Neighbor Expansion** | **Proximity** | **Optimization** |
| :--- | :--- | :--- | :--- | :--- |
| DeepWalk | KDD 2014 | DFS，Truncated Random Walks | $$1^{st} - k^{th} $$ order | Skip-gram with Hierarchical Softmax |
| LINE | WWW 2015 | BFS，1-hop and 2-hop Neighbors | $$1^{st} $$ or $$ 2^{ed}$$  | Skip-gram with Negative Sampling |
| Node2Vec | KDD 2016 | BFS + DFS，Biased Truncated Random Walks | $$1^{st} - k^{th} $$ order | Skip-gram with Negative Sampling |

![](.gitbook/assets/image%20%285%29.png)

Skip-Gram 模型将节点 ID 作为输入，最终为网络中节点 ID 学习得到一个 Embedding Lookup，这也是为什么我们将这些方法视为 non-compositional 方法的原因。这类方法设计简洁，但是对于训练过程中未见过的节点（unseen node），除非在这些节点上执行额外的优化工作 \[12\]，否则不能自然地为其生成 Embedding。此外，由于没有利用节点的属性特征，这也严重限制了其表示能力。

### 2.2 第二类：Compositional 方法

尽管在以往的工作中，组合性原则 Compositionally 的概念并未被明确提及，但是最近几年的研究以组合性原则解决网络表示学习问题已成为一种趋势，即 Aggregator。典型工作如 GraphSAGE \[13\]，Graph Convolutional Network（GCN） \[14\] 和 DeepGL \[15\] ，三者都是通过聚合一到多跳的邻居节点 Embedding 特征生成目标节点 Embedding，如下图所示 \[16\]。

![](.gitbook/assets/image%20%2815%29.png)

Graph Attention Network（GAT）\[17\] 在聚合计算中引入了注意力机制，目标节点也更加关注其直接邻居特征，网络中每个节点的更新都是基于其直接邻居节点 Embedding 的 Multi-head Attention 加权求和得到，如下图所示。

![](.gitbook/assets/image%20%2811%29.png)

在现实世界中，节点除了拥有邻居之外，还有丰富的节点属性特征和边信息，如 Facebook 社交网络中用户画像、好友之间互动频率等，它们也可以以相同的方式聚合使用。一些研究者试图同时对网络结构和边信息进行建模，这些联合学习方法主要是基于 Skip-gram 和矩阵分解等框架将网络拓扑信息和边信息进行等价集成，如 CANE \[18\]，TriDNR \[19\]。

### 2.3 现有方法的不足

由上可知，几乎所有现有的网络表示学习方法都是通过学习一个 encoder 将节点 ID 映射到对应的 Node Embedding，主要存在不足如下：

1. **节点 ID 并不具有泛化性，导致现有方法有着严重的冷启动问题，即无法对训练阶段未见过的节点做 Embedding 推理。**尽管 ****Dynamic Network Embedding 可以通过增量学习或归纳的方法生成未见过节点的 Embedding，但是，其前提是要求新加入节点和网络中已有节点有连边，但是在很多现实情况下这是不可能的，比如推荐系统中新上架商品、新发表文章，这种设计原则势必阻碍了在实际场景中的应用推广。
2. **异构网络中节点类型无法通过节点 ID 识别，所以通常需要额外的工作来编码节点类型信息。**传统的异构网络表示学习方法将不同类型的节点投影到不同的潜在空间中，在此基础上，还需要花很大的功夫对齐不同空间的 Embedding。
3. **节点 ID 携带的信息是太少，面对噪声数据时的鲁棒性较差。**网络表示学习方法在保存网络结构信息的同时，需要对拓扑结构的微小改动有一定的鲁棒性。但是，每条边都由一对节点 ID 表示。基于随机游走采样对边的存在与否貌似具有较强的鲁棒性，但是，恰恰相反，Node Embedding 依赖邻居节点聚合得到，反而可能更容易受到错误边的严重影响。

## **3. CNE 模型**

为了解决以上问题，受 NLP 领域众所周知的组合性原则（或称语义合成性）启发，该原则指出一个复杂表达式的含义由组成它的子表达式的含义和组合规则所决定，比如，语素组合成单词，单词组合成短语，短语组合成句子，句子组合成段落。本文以组合的方式建模网络，通过节点的属性特征组合得到节点 Embedding，从而提出了 CNE 模型，如下图所示。

![](.gitbook/assets/image%20%283%29.png)

CNE 是一个典型的深度学习双塔模型，一端是目标节点$$v$$，另一端是其邻居节点$$u \in N(v)$$ ，模型通过拟合网络/图中节点的临近性来学习节点属性的 Embedding $$A_v$$和组合函数$$\varnothing _i$$，组合函数$$\varnothing _i$$以相应节点的属性特征$$A_v$$作为输入，在同类型节点之间共享，节点 Embedding 只是作为中间结果计算得到，网络中节点的临近性由随机游走序列和滑动窗口使用方式灵活捕捉得到，即当网络中两个节点距离接近时，模型保证计算得到的节点 Embedding 也会比较相似。

### 3.1 Embedding 组合策略

节点$$v_i$$ Embedding 生成方式非常简单直接，如下式所示：

$$
v_i = \varnothing (A_i) = \varnothing (a_i^{(1)}, ... ,a_i^{(n_i)})
$$

其中，$$a_i^{(j)} \in R^d$$ 是节点 $$v_i$$ 第 $$j$$ 个属性特征对应的 Embedding，$$d$$是 $$a_i^{(j)}$$ 的维度，$$n_i$$是节点 $$v_i$$ 的属性总数，组合函数 $$\varnothing$$ 实际上是一个神经网络，或称为 Encoder，后续我们将二者等价对待。特别强调下， 属性 Embedding$$a$$在整个网络中的所有节点之间是共享的。这样的话，一旦模型训练完成，所有参数就固定下来了。对于哪怕训练阶段未见过的节点，也可以根据共享属性 Embedding 作为输入，通过组合函数 $$\varnothing$$ 计算得到对应的 Embedding。

### 3.2 组合函数（Encoder）

CNE 是一个非常通用的归纳式网络表示学习框架，主要体现在节点属性特征和组合函数的设计上非常灵活。例如，我们在用户购物行为网络中可以使用文本（如商品标题）作为特征，在 Flickr 图片社区网络中可以使用图片作为特征。而且，根据节点属性特征的不同，还可以设计不同的 Encoder，从简单的 concat，mean，sum 操作，到复杂的深度学习模型，如 DNN，LSTM/GRU，CNN，Self-Attention，Transformer。

为了简单起见，论文在不失一般性的基础上，给出了以文本属性特征输入的网络示例，编码器使用的是 GRU，如下式所示（省略了下标）。

$$
r^{(t)}=\sigma (W_ra^{(t)}+U_rh^{(t-1)}) \\
z^({t)}=\sigma (W_za^{(t)}+U_zh^{(t-1)}) \\
\hat h^{(t)}=tanh(Wa^{(t)}+U(r^{(t)} \odot h^{(t-1)})) \\
h^{(t)}=(1-z^{(t)}) \odot h^{(t-1)}+z^{(t)} \odot \hat h^{(t)}
$$

其中，$$\sigma$$是 sigmoid 函数，$$\odot $$是 element-wise 乘法，$$r^{(t)}$$ 是重置门（reset gate），$$z^{(t)}$$ 是更新门（update gate），所有的非线性操作都是 element-wise 计算，权重矩阵 $$W_r, W_z, W, U$$ 都是需要可学习的参数矩阵。最后的隐状态输出$$h^{(n_i)}_{v_i}$$将作为节点$$v_i$$的 Embedding。

### 3.3 优化目标

论文采用了 max-margin（或称 hinge）损失函数，最常见的应用是在支持向量机 SVM 模型中，不同于 SVM 的分类任务，这里是一种 pairwise 相似度量关系的损失函数，也是无监督的 Graph-based 损失函数，如下式所示。

$$
L(v, u)= \sum_{k=1}^K max(0, m- \delta (v, u) + \delta (v, \hat u_k))
$$

其中， $$u \in N(v)$$是的 $$v$$ 的正样本（邻居）， $$\hat u_k$$ 是从全部节点集合 $$V$$ 中随机采样得到关于节点 $$v$$ 的负样本， $$K$$ 是负样本总数； $$m$$ 是 margin 超参数，通常设置为 1。 $$\delta$$ 是节点之间相似度函数，如下式所示。

$$
\delta (v, u)=cos(v, u)=cos(\varnothing_1(A_v), \varnothing_2(A_u))
$$

其中， $$\varnothing_1$$ 和 $$\varnothing_2$$ 分别是节点 $$v$$ 和 $$u$$ 对应的组合函数（Encoder）。

直观理解，该损失函数希望正样本分数越高越好，负样本分数越低越好，但二者得分之差最多到 $$m$$ 就足够了，差距增大并不会有任何奖励。当然，这个也可以采用其他损失函数，比如 DeepWalk 模型使用的极大似然函数（或称交叉熵损失函数），只是通过大量实验证明，max-margin 损失函数效果最好。 

特别说明下，CNE 可以很方便的通过修改无监督损失函数支持到有监督任务的优化目标，如在电商推荐场景中用户购买下单行为可以认为是有监督标注数据，基于多源异构网络的表示学习对应下游主要应用场景是希望提升商品购买下单率，损失函数修改方法有多种，如只保留购买下单行为边带来的损失，或在已有损失函数基础上，对购买下单行为边做加权处理。又或者是采用类似 NLP/CV 常见的预训练+finetune两阶段做法：先按照无监督损失函数预训练，然后再针对有监督损失函数做 finetune。

### 3.4 邻居 $$N(v)$$ 选取策略

如上所述，训练阶段邻居 $$N(v)$$ 的定义至关重要。论文中采用了和 DeepWalk 一致的简单有效的随机游走策略。具体的，首先遍历每个节点作为初始采用点生成随机游走序列，下一个节点生成方法如下式所示（ $$Z$$ 是归一化因子），控制序列最大长度 $$l$$ ，然后定义序列中固定窗口大小 $$w$$ 内的节点之间互为邻居，这种样本生成策略使得节点 Embedding 相似度和测地距离（geodesic distance）会比较相关。

$$
P(c_i=u | c_{i-1}=v)=\left\{
\begin{aligned}
\frac {w_{vu}}Z, \qquad & if (v, u) \in E \\
0, \qquad & otherwise 
\end{aligned}
\right.
$$

注：测地距离是指在曲面上从 $$A$$ 点走到 $$B$$ 点（不允许离开曲面）的最短距离。

最开始提到我们采用深度学习双塔模型作为预训练模型（User/Item Embedding），无监督的训练数据构建方法有两种：

1. 固定窗口内的用户行为序列任意两个 Item 之间构成一个样本：两个塔都是 Item，即并不直接对 User 建模，User Embedding 根据最近的行为序列累加 Item Embedding 得到。
2. 存在特定行为的 User 和 Item Pair 构成一个样本：相比 CNE，只考虑一跳边，或称直接邻居，一个塔是 User，一个塔是 Item。

### 3.5 支持不同类型网络

#### 3.5.1 有向图

这种情况下，边的方向信息需要被保留下来。为此，我们可以为节点 $$v$$ 和 $$u$$学习不同的组合函数，不保证对称性，即 $$δ(v,u) \neq δ(u,v)$$ 。

#### 3.5.2 异构网络（含不同类型节点）

这种情况下，不同类型节点对应的属性特征集很可能不同，如在推荐系统中，一端是用户，一端是 item，前者主要是用户人口学属性、兴趣偏好、购物意图、搜索行为等属性特征，后者主要是商品标题、商品描述、价格、评分、销售量等属性特征。为此，我们同时可以为不同类型节点使用不同的组合函数，最终，把不同类型节点都投影到同一个低维向量空间中。

#### 3.5.3 异构网络（含不同类型边）

这种情况下，CNE 可以为不同类型边的数据训练不同的组合函数，但是，输入层的节点属性特征 Embedding 需要共享使用。比如 Twitter 社交网络中，好友之间可以有关注和转发推文的关系，针对这两种行为可以设计 4 种组合函数，这时和多任务学习非常类似，如下图所示。

![](.gitbook/assets/image%20%286%29.png)

在损失函数中，还要考虑不同类型边的损失如何融合，论文中并未提及，最简单的方法是直接累加。但是，实际情况中，不同边的稀疏度、重要度是不同的。所以，更好的做法是为不同边产生的部分损失函数指定一个超参数作为权重系数$$w_i$$，$$w_i$$通过人工经验或 Grid Search 等方法获得，修正后的损失函数如下式所示。

$$
L(v, u)= w_1 \cdot L_1(v, u) + w_2 \cdot L_2(v, u) + ... ...
$$

最终为每个节点学习得到一组 Embedding，可以单独使用，也可以考虑融合成一个节点 Embedding，怎么融合呢？可以由多个 Embedding Concat 在一起，主要要在 Concat 之前为 Embedding 乘上对应的权重系数 $$w_i$$ ，即 $$v=[w_1 \cdot v_1, w_2 \cdot v_2, ... ...]$$。 

除此之外，论文中并未提及如何处理**带权节点**和**带权边**的情况，这里谈下个人看法，主要有两种处理方式（推荐前者）：

* 样本加权采样（Weighed Sampling）：

> 对于无权边，每次从当前节点下所有直接邻居中随机采样得到下一个节点，对于带权边，可以参考 LINE \[9\] 的做法，根据边权重加权采样下一个节点。
>
> 对于无权节点，挨个遍历节点采用随机游走策略生成训练样本序列，也就是每个节点作为初始采样点的概率均等，对于带权节点，可以有三种做法：

> 1\) 每次从全部节点中按节点权重加权概率采样决定初始采样点，这种方法可能导致某些低权节点很难被采样命中，导致未参与训练；
>
> 2\) 挨个遍历节点作为初始采样点，每生成一个随机游走序列后，根据节点权重加权概率决定是否继续沿用当前节点做初始采样点；
>
> 3\) 挨个遍历节点作为初始采样点，每生成一个节点后，根据节点权重加权概率决定是否继续沿用当前节点做初始采样点。
>
> 这里加权采样算法推荐使用时间复杂度 $$O(1)$$ 的 Alias Sampling 算法，在 LINE \[9\] 和 Node2vec \[10\] 模型中都有应用。

* 引入样本 weight/important：

> 对于带权边，可以将边权重 $$w_{v,u}$$ 作为系数加入到损失函数中。对于带权节点，可以将相邻两个节点的权重相乘之后（也用 $$w_{v,u}$$ 表示），采用与带权边类似的做法。如下式所示。

$$
L(v, u)= w_{v,u} \cdot \sum_{k=1}^K max(0, m- \delta (v, u) + \delta (v, \hat u_k))
$$

## 4. 实验 

CNE 模型主要在链路预测任务中做了大量实验评估对比，包括缺失边的预测，不可见节点边的预测，多类型边的预测和多类型节点边的预测。实验结果表明，CNE 具有较好的表达能力、较强的泛化能力，以及异构网络下更高的灵活性。

## 5. 总结

简要总结一下，相比于传统的网络表示方法，CNE 模型至少具有如下优点：

1. **能够推理未见节点的 Embedding。**一旦 CNE 模型训练好之后，就可以使用节点属性作为输入来推理得到新节点的 Embedding。
2. **易于应用于异构网络。**不同类型的节点对应不同的节点属性和组合函数，CNE 就可以很自然地捕获到不同类型之间的差异。
3. **对拥有较少边信息的情况也有着较好的鲁棒性。**CNE 在共享节点属性和组合函数的基础上建模网络拓扑结构，起到了强大的正则化作用。CNE 对于拥有较少共现属性的节点之间的边也并不敏感，如一个皇家马德里球迷点击了一个巴塞罗那俱乐部的球衣。

### 和现有方法对比

在现有方法中，所有或部分输入都来自于邻居节点，如 GraphSAGE，GCN；Non-compositional 方法主要针对网络拓扑结构进行建模，如 CANE，TriDNR。CNE 模型和已有的 Compositional 方法最大的不同在于模型输入，CNE 只接受目标节点的属性作为输入，完全放弃节点 ID，这种差异使得 CNE 在推理训练阶段未见过的节点，甚至和现有网络没有连边的情况，明显优于现有的方法。只有 CNE 的编码器在训练和推理阶段利用了相同数量的信息，相反，现有的方法在推理时只能利用部分信息，甚至没有可用的信息。

## 6. 思考

Q：GraphSAGE，GAT 等方法如何利用节点属性信息？（节点权重和边权重不再展开）

Q：

## 7. 参考文献

1. P.-S. Huang, X. He, J. Gao, L. Deng, A. Acero, L. Heck, Learning deep structured semantic models for web search using clickthrough data, in: Proceedings of the 22Nd ACM International Conference on Information & Knowledge Management, CIKM ’13, ACM, New York, NY, USA, 2013, pp. 2333–2338. 
2. Ali Mamdouh Elkahky, Yang Song, and Xiaodong He. A multi-view deep learning approach for cross domain user modeling in recommendation systems. In WWW 2015. 278–288. [https://doi.org/10.1145/2736277.2741667](https://doi.org/10.1145/2736277.2741667)
3. Paul Covington, Jay Adams, and Emre Sargin. Deep Neural Networks for YouTube Recommendations. In Proceedings of the RecSys 2016. 191–198. [http://doi.acm.org/10.1145/2959100.2959190](http://doi.acm.org/10.1145/2959100.2959190)
4. [Tianshu Lyu](https://arxiv.org/search/cs?searchtype=author&query=Lyu%2C+T), [Fei Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+F), [Peng Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+P), [Wenwu Ou](https://arxiv.org/search/cs?searchtype=author&query=Ou%2C+W), [Yan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y). Compositional Network Embedding for Link Prediction. In Proceedings of the RecSys 2019.
5. R. Zhu, K. Zhao, H. Yang, W. Lin, C. Zhou, B. Ai, Y. Li, and J. Zhou, AliGraph: A Comprehensive Graph Neural Network Platform, In Proceedings of the 45th International Conference on Very Large Data Bases, 2019.
6. Daokun Zhang, Jie Yin, Xingquan Zhu, Chengqi Zhang. Network Representation Learning: A Survey. IEEE transactions on Big Data, 2018.
7. Perozzi B, Al-Rfou R, Skiena S. DeepWalk: Online Learning of Social Representations. In: Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, New York, 2014. 701–710.
8. 涂存超, 杨成, 刘知远, 孙茂松. 网络表示学习综述. 中国科学: 信息科学, 2017, 47: 980–996. [http://engine.scichina.com/doi/10.1360/N112017-00145](http://engine.scichina.com/doi/10.1360/N112017-00145)
9. J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, Q. Mei, Line: Large-scale information network embedding, in: Proceedings 24th International Conference on World Wide Web, 2015, pp. 1067–1077.
10. A. Grover, J. Leskovec, node2vec: Scalable feature learning for net- works, in: Proceedings of the 22nd International Conference on Knowledge Discovery and Data Mining, ACM, 2016, pp. 855–864.
11. Haochen Chen, Bryan Perozzi, Rami Al-Rfou, Steven Skiena. A Tutorial on Network Embeddings. 2018. [https://arxiv.org/pdf/1808.02590.pdf](https://arxiv.org/pdf/1808.02590.pdf)
12. Lun Du, Yun Wang, Guojie Song, Zhicong Lu, and Junshan Wang. 2018. Dynamic Network Embedding : An Extended Approach for Skip-gram based Network Embedding. In Proceedings ofthe Twenty-Seventh International Joint Conference on Artificial Intelligence. International Joint Conferences on Artificial Intelligence Organization, 2086–2092.
13. William L. Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive Representation Learning on Large Graphs. In Proceedings of Neural Information Processing Systems. MIT Press, Cambridge, MA, USA, 1025–1035.
14. Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. In Proceedings of International Conference for Learning Representations. 1–14.
15. Ryan A. Rossi, Rong Zhou, and Nesreen K. Ahmed. 2018. Deep Inductive Network Representation Learning. In Companion Proceedings of the The Web Conference. International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, Switzerland, 953–960. 
16. Jure Leskovec, William L. Hamilton, Rex Ying, Rok Sosic. WWW-18 Tutorial: Representation Learning on Networks. WWW 2018. [http://snap.stanford.edu/proj/embeddings-www/](http://snap.stanford.edu/proj/embeddings-www/)
17. Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li\`o, and Yoshua Bengio. Graph Attention Networks. In Proceedings of International Conference for Learning Representations 2018.
18. Cunchao Tu, Han Liu, Zhiyuan Liu, and Maosong Sun. Cane: Context-aware network embedding for relation modeling. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics \(Volume 1: Long Papers\), volume 1, pages 1722–1731, 2017.
19. Shirui Pan, Jia Wu, Xingquan Zhu, Chengqi Zhang, and Yang Wang. Tri-Party Deep Network Representation. In Proceedings ofthe Twenty-Fiſth International Joint Conference on Artificial Intelligence. International Joint Conferences on Artificial Intelligence, 2016. 1895–1901.
20. 




1. Palash Goyal and Emilio Ferrara. 2018. Graph embedding techniques, applications, and performance: A survey. Knowledge-Based Systems 151 \(2018\), 78–94.
2.  1. Jian Tang, Cheng Li, Qiaozhu Mei. KDD17 Tutorial: Learning Representations of Large-scale Networks. [https://sites.google.com/site/pkujiantang/home/kdd17-tutorial](https://sites.google.com/site/pkujiantang/home/kdd17-tutorial)

> 尾声：

Q：CNE 有哪些缺点/不足？

Q：inductive任务是指：训练阶段与测试阶段需要处理的graph不同。通常是训练阶段只是在子图（subgraph）上进行，测试阶段需要处理未知的顶点。（unseen node）

transductive任务是指：训练阶段与测试阶段都基于同样的图结构



带权异构网络：改loss；带权采样策略

