# State of the Art GNN models from scratch ! 

## Semi-supervised classification with Graph Convolution Networks (Kipf et al. ICLR 2017)

Paper here --> https://arxiv.org/abs/1609.02907

GCN is a very important paper in modern GNN. The objective of this paper is to generalize convolution to graphs. Several existing methods were based on the spectral decomposition of the Laplacian matrix of a graph to perform convolutions. However, these methods suffer from a high computational time (the eigenvector decomposition is expensive) and the slightest change in the graph makes the decomposition change and disturbs the learning and the predictions. 
GCN aims at simplifying these spectral methods and proposes a simpler, more efficient and inductive architecture for convolution on graphs. 


See this blog to understand the motivation of graph convolution https://distill.pub/2021/understanding-gnns/

The GCN architecture is based on a very comprehensive matrix multiplication, an exemple here with a 2 layer GCN : \
$ Z=f(X, A)=\operatorname{softmax}\left(\hat{A} \operatorname{ReLU}\left(\hat{A} X W^{(0)}\right) W^{(1)}\right) $

where $ \hat{A}=\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} $ and $\tilde{A} = A + I$ 

$A$ is the adjacency matrix of size $n\times n$ of a graph $G$  
$D$ the degree matrix of these graph. \
$X$ is a feature matrix of size $n \times m $ \
$W^{(0)}$ a weight matrix of size $m \times z$ \
$W^{(1)}$ a weight matrix of size $z \times n\_class$ in the case of a supervised training. 

In the GCN folder I reimplement from scratch in pytorch a simple two layer GCN on the dataset Zachary Karate Club https://en.wikipedia.org/wiki/Zachary%27s_karate_club