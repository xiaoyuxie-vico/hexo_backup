---
title: A Survey for Vision Transformers (By Mar. 2021)
date: 2021-04-25 13:02:11
categories: Course project 
tags: 
    - Survey
    - Vision transformer
    - Computer vision
    - Course project
---

**ABSTRACT:** Inspired by the impressive performance of transformer in natural language tasks, a growing number of researchers is exploring vision transformer and trying to apply to their own fields. In this paper, I will give a comprehensive literature review about vision transformer. I start with the background, motivation and paper structure int the first section. Then, I will go through the development and fundamental knowledge of convolution neural networks, self-attention, and Fourier-based networks. After that, I will discuss the transformer structure and its applications and make some classification and comparison of various methods. Finally, I will clarify some difficulties, potential research directions, and give course feedback.

<!-- more -->

**KEYWORDS:** *Self-attention, transformer, convolution neural networks, deep neural networks.*

# 1.  Introduction

Convolutional neural networks (CNN) has dominated computer vision field since the impressive success of a ImageNet Large Scale Visual Recognition Challenge in 2014 (Russakovsky et al., 2015). Various CNN architectures (He et al., 2015; Szegedy et al., 2016; G. Huang et al., 2016; Hu et al., 2017) springs up like mushrooms since that time. These carefully designed CNN architectures have created significant progress in various visual fields, like image classification (Howard et al., 2017; Szegedy et al., 2014; X. Wang et al., 2018), object detection (Redmon and Farhadi, 2018; Ren et al., 2016), semantic segmentation (Z. Huang et al., 2019; P. Wang et al., 2018; S. Zheng et al., 2020), image generation(X. Chen et al., 2016; Goodfellow, 2016), etc. Most of CNN architectures (He et al., 2015; Simonyan and Zisserman, 2014) heavily benefit from different convolution operations like 3×3 convolutions. These convolution operations can learn local information from visual inputs. Deep stacked convolutions then help to gather high level information (Goodfellow et al., 2016), which help to do different image understanding tasks. In about less than a decade, CNN has become the main design paradigm in many modern deep learning systems.

Despite the great success of CNN, there are some limitations. First, while convolution operations can process a local neighbor in space and time, convolution is hard to capture long-distance dependencies. These dependencies are vital to semantic concepts. For example, a bird break in different angles have to be treated with different convolution operations due to the changes of break position. Generally, we can use different convolution operations, increase kernel sizes, increase model depth, or dilated convolutions to solve this problem. But the computation cost can be high and require careful design. Second, convolution treats all information as equal, but in fact not all pixels should be regarded as equal. For example, when we classify dogs and cats, we pay more attention on their ears and noses and ignore some useless background information. That is to say, image classification should give priority to important foreground objects instead of background. But convolution operations process all local neighborhood information with equal weights and do not have a concept of importance. This also causes inefficiency in computation. 

At the same time, transformer shows its power in natural language processing (NLP) tasks and almost become the standard architecture in NLP tasks. One of important reason is that transformer is good at modeling long distance information. In 2017, Vaswani et at. (Vaswani et al., 2017) first replaced convolutions and recurrence with transformer which only based on attention modules. The architecture allows to train model in a more parallelizable way and significantly decrease the training time. In 2019, Devlin et al. (Devlin et al., 2019) proposed a new language representation model called BERT, which aims to train a deep learning model from unlabeled text by jointly conditioning on all context in all layers. One of the most impressive things is BERT model can be easily transferred to other tasks by just fine tuning one additional output layer. In 2020, Brown et al. (Brown et al., 2020) proposed an autoregressive language model GPT-3 which achieves impressive performance on various NLP tasks, like machine translation and question-answering system. 

Inspired by the significant improvement of transformer in natural language processing (NLP) tasks, a growing number of researchers are trying to take advantage of transformer in computer vision (Carion et al., 2020; Dosovitskiy et al., 2020; W. Wang et al., 2021; Srinivas et al., 2021). Chen et al. (M. Chen et al., 2020) firstly tried to explore the application of transformer in images. They trained a sequence transformer model to predict pixels and found that the model can learn strong image representations. Dosovitskiy et al. (Dosovitskiy et al., 2020) proposed a fully-transformer model called Vision Transformer (ViT) that can be directly used in image classification. Vit treats each image as a set of fixed-size patches (14×14 or 16×16) that will be feed into a transformer encoder to make classification. Though ViT achieves excellent results compared to stat-of-the-art CNN models, it heavily relies on a large-size dataset like JFT-300M (C. Sun et al., 2017) (300 million images) and have relative worse performance on a medium size dataset like ImageNet (Deng et al., 2009).  To tackle this problem and make models train effectively on a medium size dataset, Touvron et al. (Touvron et al., 2021) also proposed an efficient convolution-free transformer model called DeiT. DeiT can be trained on a single 8-GPU node in about three days by leveraging a new distillation token and a specific teacher-student strategy. In the early of 2021, various explorations in vision transformer has been proposed, including Pyramid Vision Transformer (PVT) (W. Wang et al., 2021), Transformer in Transformer (TiT) (Han et al., 2021), and Tokens-to-token ViT (Yuan et al., 2021). 

Even though the application of transformer has dominated a variety of NLP tasks since 2017, the application of transformer in CV is just rapidly developed especially since the late 2020. There is still no standard architecture in vision transformer and different models have their own advantages and disadvantages. Almost in every month, there are new proposed architectures about vision transformer. Thus, this paper tries to do a comprehensive survey about the recent vision transformer and compare different architectures in different applications. Specifically, this paper tries to answer four questions in vision transformer: 1) what is transformer? 2) why we need transformer in CV? 3) how transformer is applied in CV? 4) what are existing difficulties and potential development? 

The structure of the paper is as follows. In section 2, foundations and preliminary knowledge about transformer will be introduced. Some close related ideas like conventional CNN architectures, channel attention, self-attention, multi-head self-attention, and non-local network will be discussed. This section tries to discuss the evolution of the design of vision models from convolution to transformer. The first aforementioned question will be discussed in this section. In section 3, the core topic transformer will be discussed in detailed. The structure of transformer and different applications of transformer will be introduced, including image classification, object detection, segmentation, etc. This section will show the power of transformer in vision fields, which will also clarify the second and third question in depth. In section 4, vision transformer will be classified into three categories, including transformer with global attention, transformer plus CNN, and transformer with efficient attention. Section 3 and section 4 will help to answer the third aforementioned question. In section 5, existing problems and difficulties in vision transformer will be discussed. Some potential and valuable future work will also be discussed in this section. This section will answer the fourth aforementioned question. In the last section, I will conclude the development of vision transformer and summary the paper.

# 2.  Related work

Visio transformer closely relates to convolution operation and self-attention. Thus, in this section I will firstly go through the milestone vision architectures since 2012 to explain how researchers designed models and improved performance. Then, I will give an introduction about self-attention, which is a core module in transformer. I will also introduce non-local network and two Fourier-based networks to explain there are other ways to capture long-range dependencies instead of self-attention.

## 2.1 CNN

There are a lot of CNN model taking advantage from a deeper neural network. Going deeper for a neural network is always the first choice for complex tasks. In 1989, LeNet (LeCun et al., 1989) , one of the earliest CNN, was proposed by LeCun et al. It only has 7 layers, including 3 convolutional layers, 2 pooling layers, and 1 fully-connected layer. It defines the fundamental modules for CNN. Trained with backpropagation algorithms, it can be successfully identified handwritten zip code numbers. 

In 2012, AlexNet (Krizhevsky et al., 2012) increased the model depth to 8 layers. It achieved a great improvement (a top-5 error of 15.3%, more than 10.8 percentage points than second one) in ImageNet (Deng et al., 2009). AlexNet shows the power of increase model depth and makes it feasible when use GPU during training. After AlexNet, there are countless number of researchers trying to use Deep CNN to improve performance in their fields. This also greatly improve the development of deep learning in all fields. 

In 2015, VGG (Simonyan and Zisserman, 2014) achieved the state-of-the-art performance on ILSVRC classiﬁcation and localisation tasks. It heavily uses 3×3 convolutions and increases the depth by adding more convolutional layers. 

Considering the significance of the depth, Residual Network (ResNet) (He et al., 2015) tried to explore how to increase the depth and avoid vanishing/exploding gradients (Bengio et al., 1994; Glorot and Bengio, 2010). This is because He et al. found that directly stacked more layers cannot have a better performance. They proposed a residual block and let these layers fit a residual mapping. They increased the depth from 18 layers to 152 layers and increase performance, which is the first time to surpass 100-layer barrier. ResNet has also been a popular backbone in various vision tasks since then. 

In 2016, Densely Connected Convolutional Networks (DenseNet) not only creates short paths from early layers to later layers, it also connects all layers directly with each other.  So, it uses direct connections between any two layers with the same feature-map size to implement this idea. This method also helps to solve the gradient vanishing problem. The maximum depth of DenseNet reached 264. The architecture of DenseNet is shown in Figure 1.

![img](clip_image001.png)

Figure 1. DenseNet (G. Huang et al., 2016)

It is needed to note that convolution operations have dominated various visual recognition tasks with a deep neural network since 2012. There are several reasons behind its popular. First, convolution layers can capture favorable visual context using filters. These filters shared weights over the entire image space, so convolution operations have translation equivariance. Second, convolution operation can take advantage of the rapid development of computational resource like GPU because it is inherent parallelable. And GPU allows to train a very deep CNN with fewer time, which helps to train on large-scale image dataset, like ImageNet. Third, multiple kernel paths help to achieve a competitive performance. GoogleNet (Szegedy et al., 2014) carefully designs a Inception module with 1×1 convolution, 3×3 convolution, and 5×5 convolution to increasing the depth and width of the network while keeping the computational budget constant. Fourth, CNN can use hierarchical structure to learn high-level information. This is because spatial and channel-wise information can be fused in a local receptive filed in convolutional layers. The fused information can then be passed to non-linear activation functions or/and down-sampling layers to learn a better representation that is vital to capture global theoretical receptive ﬁelds. 

## 2.2 Attention in CNN

Considering the importance of channel relationship, SENet (Hu et al., 2017) uses a Squeeze-and-Excitation (SE) block to explicitly model the interdependencies between different channels. The architecture can learn different weights for different channels in SE block. It is an effective and lightweight gating mechanism to self-recalibrate the feature map. There are three important operations. First, in the squeeze operation, SE block will shrink features through spatial dimensions. Second, in the excitation operation, it will learn weights to explicitly model channel association. Third, it reweights the feature maps by the learned weights. The SE block is shown in Figure 2. 

![img](clip_image002.png)

Figure 2. A Squeeze-and-Excitation block. (Hu et al., 2017)

Inspired the SENet’s channel recalibration mechanism and multi-branch convolutional networks (He et al., 2015; Srivastava et al., 2015), SKNet uses a non-linear approach to aggregate multiple scale feature and constructs a dynamic selection mechanism in CNN instead of fixed receptive fields in standard CNN. It has three basic operators – Split, Fuse, and Select, which are shown in Figure 3. Split operator means that they use two filters with different filter sizes. Fuse operator aims to enable neurons to adaptively adjust receptive fields based on the stimulus content. Select operator is a soft attention across channels, which is similar to SENet. This architecture improves the efﬁciency and effectiveness of object recognition. 

![img](clip_image003.png)

Figure 3. Selective Kernel Convolution. (X. Li et al., 2019)

Some researchers tried to not only use channel attention, but also take into account spatial attention as well. For example, Bottleneck Attention Module (BAM) (Park et al., 2018) and Convolutional Block Attention Module (CBAM) (Woo et al., 2018) were proposed to infer an attention map along two separate dimensions, channel and spatial. But BAM’s attention computes in parallel, while CBAM computes attention sequentially. 

Other researchers only used spatial attention and did not consider channel attention. For example, Non-local Network (NLNet) (X. Wang et al., 2018) tries to capture long-range dependencies by avoiding messages delivered back and forth between distant positions. They proposed non-local operations to have spatial attention. This operation can use a weighted sum of the features to compute the response at a position. They applied NLNet in video classification and achieved competitive performance. The non-local block is shown in Figure 4. 

![img](clip_image004.png)

Figure 4. Non-local block. (X. Wang et al., 2018)

Although NLNet presents a way to capture long-range information, global contexts are almost the same for different query positions (Cao et al., 2019). So, construct a global context network (GCNet) is proposed to solve this problem by unify the basic idea of SNNet and NLNet, which means it takes into account of spatial and channel information at the same time. 

To summary the application of attention in CNN, the comparison of NLNet, SENet, SKNet, and GCNet, etc. are shown in Table 1.

Table 1. Model comparison for the application of attention in CNN.



| **Model name** | **Characteristics**                                  |
| -------------- | ---------------------------------------------------- |
| SENet          | Spatial aggregation, channel attention               |
| SKNet          | Adaptive receptive filed, channel attention          |
| BAM            | Channel attention and spatial attention in parallel  |
| CBAM           | Channel attention and spatial attention sequentially |
| NLNet          | Spatial weighted sum per pixel                       |
| GCNet          | Spatial weighted sum, channel attention              |

 

## 2.3 Self-attention

Self-attention (Vaswani et al., 2017) can transform an input to a better representation by allowing inputs to interact with each other. It will compute three hidden representations called query, key, value firstly by three learnable matrixes. After that, it will compute attention weights by the dot products of query and key. The weights will be divided by the sqrt root of the key’s dimension and obtain the weighted value as output. So, the output of self-attention operation is aggregates of the interactions between all inputs. Figure 5 shows how self-attention works. 

![img](clip_image005.png)

Figure 5. Self-attention operation.

The calculation of self-attention can be formulated as a single function: 



| ![img](clip_image007.png) | (1)  |
| ------------------------- | ---- |
|                           |      |

The first important thing to understand the mechanism of self-attention is remembering the dimension of the input and output of self-attention is the same. That is to say, our goal is to have a new and better representation for our inputs, and the output needs to be the same dimension. So, what is a good representation for the input? When we are doing machine translation, we understand that even for a same word it can have different or even opposite meaning. What make the word has different meaning? The answer is context. So, we want to consider the whole context and capture the different relationships between different words. Self-attention is designed to do a similar thing. The dot product of ![img](clip_image009.png) and ![img](clip_image011.png) aims to calculate the similarity/distance between two different vectors. The score of ![img](clip_image013.png) can be used to determine how much weights need to be put in a certain input. But before we calculate a weighted value as the output, we need to do some normalization, which is shown in Equation 1. 

It is also need to note that the difference of the encoder-decoder attention layer (Luong et al., 2015) between self-attention layer is where the key matrix ![img](clip_image011.png),value matrix ![img](clip_image016.png) and query matrix ![img](clip_image009.png) come from. For the encoder-decoder attention layer, ![img](clip_image019.png)and ![img](clip_image016.png) come from the encoder module and ![img](clip_image009.png) comes from the previous layer. Other operations are the same. 

The limitation for self-attention layer is that it cannot give different weights for different parts of the input. One way to solve that is to use multi-head attention instead of one single-head attention. The structure of multi-head attention is shown in Figure 6. It helps to boost the performance of the vanilla self-attention layer by giving attention layers different representation subspace. Different query, key, and value matrix are used for different heads So, they can learn different representation in training. For example, several heads are good at learning local information. Others are good at learning global information. Specifically, the operation of multi-head attention can be formulated as below:



| ![img](clip_image023.png) | (2)  |
| ------------------------- | ---- |
|                           |      |

where ![img](clip_image025.png), ![img](clip_image027.png), ![img](clip_image029.png), and ![img](clip_image031.png)are the concatenation of ![img](clip_image033.png). 

![img](clip_image034.png)

Figure 6. Multi-head attention. (Vaswani et al., 2017)

## 2.4 Fourier-based network

Convolution and attention are not the only choice to improve model performance. Fourier analysis tells us that any signal can be represented as a linear combination of sin and cos functions with different coefficients regardless of signal’s dimension. Some researcher recently tried to integrate Short Time Fourier Transformation (STFT) in to vision architectures. For example, Fast Fourier Convolution (FFC) (Lu Chi, 2020) adds a global branch that manipulates image-level spectrum and a semi-global branch to process stacked image patches. This design help to capture non-local receptive fields and cross-scale fusion. 

Another network is call Fourier Neural Operator (FNO) (Z. Li et al., 2020). Although FNO is designed to solve partial differential equations in engineering and science, it is a pure neural network and does not have close relationship with some numerical methods, like Finite Element Analysis or Finite Difference Method. The basic idea is that they build a Fourier Layer/ Block and stacked many Fourier Layers to get a better performance. Specifically, it has two branches. The first branch uses STFT to change hidden representations in spatial domain to frequency domain. Then high frequency modes are removed, which means only low frequency modes remains. The coefficients for each low frequency are learned by the Neural Network. And the frequency information is inversed to the original spatial domain by inverse STFT. The second is 1×1 convolution operation which is designed to let the model learn the residual and help to alleviate the gradient vanishing problem. Figure 7 show the structure of Fourier Layer.

![img](clip_image035.png)

Figure 7. Fourier Layer. (Z. Li et al., 2020)

There are several reasons for researchers to use Fourier Analysis in Neural Network. First, the computation is very efficient in a quasi-linear level due to the use of STFT. Second, it helps to capture coarse and global in a more efficient way by using learnable coefficients for different frequencies. Thus, in some applications, we may not need to use stacked convolutions with different size. We can use Fourier Layer to learn a good representation in frequency domain.

# 3.  Topic and its applications

In this section, I will firstly introduce the details of transformer and then provide a comprehensive review of its applications in different visual tasks, including image classification, object detection, segmentation, etc. 

## 3.1 Transformer

In the previous section, I have introduced the core module in transformer, single-head and multi-head attention layer. I will go through other modules and introduce the whole structure of transformer, which is shown in Figure 8. 

![img](clip_image036.png)

Figure 8. The architecture for transformer. (Vaswani et al., 2017)

The first thing is the positional encoding. Self-attention operation is irrelevant to the position of each input. So, it cannot capture the positional information of inputs. To solve this potential problem, they proposed a positional encoding that is added to the input embedding. 

The second thing is the Feed-forward neural network (FFNN). A FFNN is used after a self-attention layer in each encoder and decoder. This FFNN has two linear transformation layers and a ReLU activation function (Nair and Hinton, 2010). It can be expressed as:



| ![img](clip_image038.png), | (3)  |
| -------------------------- | ---- |
|                            |      |

where ![img](clip_image040.png)and ![img](clip_image042.png) are weights for the two linear transformation layers, ![img](clip_image044.png) is ReLU activation function. 

The third thing is Add & Norm module. To strength the information flow in a deep neural network, a residual connection is added in each module, including FFNN and self-attention. This is a similar design in ResNet (He et al., 2015). To boost the training, a layer-normalization (Cheng et al., 2016) is followed by the residual connection. 

The last thing is the output layer. After passing through ![img](clip_image046.png)decoders, the output will be passed to a linear layer and a softmax layer, which help to transform logits vector into word probabilities.

To summarize, Transformer, an encoder-decoder architecture, is designed to handle sequential data.

## 3.2 Applications

Since 2017, Transformer has achieved great success in various NLP tasks. For example, after pretraining, BERT (Devlin et al., 2019) can be easily used in a wide range of downstream tasks by fine-tuning one newly added output layer. Generative Pre-Trained Transformer models (GPT (Alec Radford et al., 2018), GPT-2 (A. Radford et al., 2019), and GPT-3 (Brown et al., 2020)) are another popular pretrained models in NLP. GPT-3 is designed to directly do multiple tasks without fine-tuning, but it has 175 billion parameters and requires 45 TB compressed plaintext data to train.

Inspired by the success of transformer in NLP, there are growing number of vision transformers being proposed (Beal et al., 2020; Dosovitskiy et al., 2020; Yang et al., 2020; Zhang et al., 2020; S. Zheng et al., 2020). Researchers also applied vision transformer in different fields. 

## 3.2.1  Image classification

ViT (Dosovitskiy et al., 2020) is a pure transformer applied directly to image patches. The architecture is shown in Figure 9. They reshape an image ![img](clip_image048.png) as a sequence of flattened 2D patches ![img](clip_image048.png), where H = 14 or 16. Then, a 1D positional embedding added with patch embedding are fed into transformer encoder.  The feature obtained from the first position is regarded as the global image representation and is used to do classification. ViT only uses the encoder of transformer and only has minimal changes compared to a standard transformer in NLP. 

![img](clip_image050.png)

Figure 9. Vision Transformer (ViT). (Dosovitskiy et al., 2020)

ViT achieved excellent results when pre-trained on a large dataset JFT-300M (300 million images), but it does not generalize well in mid-sized datasets like ImageNet (1.2 million images). One of the reasons is that transformer lack inductive biases inherent to CNN, such as translation equivariance and locality. So, vision transformer needs sufficient data to have a good generalization, which is similar in NLP transformer. Moreover, they compared ViT with a hybrid model that feeds intermediate feature maps into ViT by leveraging CNN model. They found that hybrid model only has better performance than ViT at small computational budgets and the difference vanishes given sufficient data. This shows that CNN may not be a necessary part for a large model. Furthermore, they analyzed the relationship between network depth and mean attention distance, which is shown in Figure 10. They found that some heads can attend at global information even at the lowest layer. 

![img](clip_image051.png)

Figure 10. Attention distance in ViT. (Dosovitskiy et al., 2020)

ViT needs to be trained on millions of images, which severely hinders its application in different fields. In the early of 2021, DeiT (Touvron et al., 2021) uses a teacher-student strategy and add a distillation token. It is a convolution-free architecture and similar to ViT, but it uses CNN as a teacher and transformer as a student. It is much more efficient and only needs to be trained about 3 days on a single 8-GPU node. To facilitate the training, it uses data augmentation and regularization strategies. These helps to greatly reduces the requirement of data and make it feasible to train on ImageNet instead of JFT-300M. They achieved better performance than ViT and EfficientNet. 

Another vision transformer called Tokens-to-Token ViT (T2T) (Yuan et al., 2021) was proposed in 2021. The architecture of T2T is shown in Figure 11 (a). T2T wants to train model in a more efficient way and avoid the reliance on a large-size dataset. They pointed out two problems in ViT. First, splitting each image into a sequence of tokens fails to capture key local structures like edges or lines. Second, the backbone of ViT is not suitable for vision tasks due to the redundancy in the attention module. From Figure 11 (b) , we can find that T2T can capture local structure features (green box), but ViT capture invalid feature maps (red box). Thus, they proposed a progressive tokenization module to replace the tokenization method in ViT. Their method helps to capture local structure information from neighbor tokens and achieve length reduction of tokens. They also compared different backbone from existing CNN architectures and found that “deep-narrow” architecture design with fewer channels but more layers in ViT is the best one.

![img](clip_image053.png)

Figure 11. T2T architecture and feature visualization. (Yuan et al., 2021)

Viewing images as a sequence of patches ignore the structure information between patches. To solve this problem, Transformer in Transformer (TNT), a pure transformer network, handles images in patch-level and pixel level. The architecture of TNT is shown in Figure 12. TNT uses two transformer blocks. The first one is an outer transformer block which is used to process patch embedding. The other one is an inner transformer block which is designed to caoture local features from pixel level. In a similar computational cost, TNT’s performance is 1.5% higher than that of DeiT on ImageNet. They also compared visualization of learned features between DeiT and TNT. Compared to DeiT, TNT can preserve better local information and have more diverse features. 

![img](clip_image054.png)

Figure 12. TNT block and TNT framework. (Han et al., 2021)

Bottleneck Transformers Network (BoTNet) is also a backbone architecture. It only modifies the final three bottleneck blocks of ResNet with global self-attention, which is shown in Figure 13. It has two main stages. In the first stage, they use convolution operations to learn high-level feature maps. In the second stage, they use global self-attention to aggregate information. Based on such design, it is easy to handle large images (1024×1024) in an efficient way. They pointed out that the designed block can be regarded as transformer block. BoTNet is quite similar to NLNet. The first difference is that BoTNet uses multi-head self-attention but NLNet does not always use it in the implementation. The second difference is that non-local block is used as an additional block, while bottleneck block is used to replace convolution block.

![img](clip_image055.png)

Figure 13. A comparison of ResNet bottleneck and bottleneck Transformer. (Srinivas et al., 2021)

## 3.2.2  Object detection

Compared to image classification, object detection requires models to predict object labels and their corresponding bounding boxes. Detection Transformer (DETR) is the first end-to-end transformer architecture used in object detection. It removes the many traditional hand-designed components like region proposal network (RPN) and non-maximal suppression (NMS). It also can predict all objects at once, which is different to RNN that predicts sequence elements one by one. DETR uses a set-based global loss which performs bipartite matching between predicted and ground-truth objects to train the model. It does not require any customized layers, which makes it easily reproduced in any framework. It achieves comparable results to Faster R-CNN (Ren et al., 2016). But it has two main limitations. First, it takes a long time to converge. This is because the attention module has uniform attention at the start of training and needs more epochs to update to optimal attention weights. Second, it easily fails to detect small objects due to limited feature spatial resolution. 

![img](clip_image056.png)

Figure 14. The architecture for DETR. (Carion et al., 2020)

To solve the drawbacks of DETR, Deformable DETR uses more efficient attention module that only attend on some key sampling points around the reference point. In this way, it greatly reduces the computational cost. Also, the deformable attention module cam fuse multi-scale features. It achieves better performance with 10 times less training epochs compared to the original DETR model.

Pyramid Vision Transformer (PVT) (W. Wang et al., 2021), a recent pure transformer backbone, also aims to capture multi-scale features maps for dense predictions. Compared to ViT, PVT is trained on dense partition of images (the patch size is 4) to get high resolution outputs. To prevent the high computational costs, PVT uses a progressive shrinking pyramid and adopts a spatial -reduction attention layer. They applied PVT in image classification, object detection, and sematic segmentation and achieved a competitive performance. PVT also can be easily combined with DETR. The combination model is called PVT+DETR, which is an entirely convolution-free network. It has better performance than DETR plus ResNet50. 

Adaptive Clustering Transformer (ACT) (M. Zheng et al., 2020) is also designed to solve DETR’s high computational cost in training and testing. ACT cluster the query features using Locality Sensitive Hashing (LSH) and use the prototype-key interaction to approximate the query-key relationship. ACT greatly reduce the computation cost while just reduce little accuracy. 

Sun et al. (Z. Sun et al., 2020) pointed out that the slow convergence of the original DETR model comes from the Hungarian loss and the Transformer cross-attention. They proposed two possible solutions: TSP-FCOS (Transformer-based Set Prediction with FCOS) and TSP-RCNN (Transformer-based Set Prediction with RCNN). These two models achieve better performance than the original DETR model.



## 3.2.3  Segmentation

Segmentation requires model to have pixel-level understanding, which is different from image classification and object detection. Video Instance Segmentation with Transformers (VisTR) (Y. Wang et al., 2020)  is an end-to-end parallel sequence prediction network. The pipeline of VisTR is shown in Figure 15. VisTR takes a sequence of images from a video as input and make instance segmentation prediction for each image directly.  VisTR is designed to learn pixel-level and instance-level similarity. It can track instances seamlessly and naturally. A CNN model is used to extract feature maps from a sequence of images. After adding with positional encoding, extracted feature maps will be fed into encoder and decoder. Finally, VisTR will output the final mask sequences by an instance sequence segmentation. 

![img](clip_image057.png)

Figure 15. The architecture for VisTR. (Y. Wang et al., 2020)

Point Transformer (Zhao et al., 2020) is a backbone architecture for 3D point cloud understanding tasks, including semantic scene segmentation, object part segmentation, and object classiﬁcation. They designed a Point Transformer layer which is invariant to permutation and cardinality. It outperforms the state-of-the-art model in S3DIS dataset for large -scale semantic scene segmentation. Axial-DeepLab (H. Wang, Zhu, Green, et al., 2020) is proposed to convert 2D self-attention as two 1D axial-attention layers which are applied to height and width sequentially. This design greatly reduces the computation and allow to capture non-local interactions easily. There is also a model called Attention-Based Transformers that is designed to instance segmentation of cells in microstructures. It uses long skip connections between the CNN encoder and CNN decoder. It achieves fast and accurate cell instance segmentation. 



# 4.  Classification and comparison of various methods

Transformer structure uses self-attention module to long-range information, while convolutions use hierarchical structure to learn high-level features. Generally, researchers try to explore architectures in these two different ways and take advantage from each other. Another important category is efficient transformer because transformer normally requires more data than standard CNN. 

 

| **Task**                      | **Category**                            | **Method**                                                  | **Characteristics**                                          |
| ----------------------------- | --------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| Image classification          | Transformer + global attention          | ViT   (Dosovitskiy  et al., 2020)                           | Only transformer encoder, patching embedding, and  positional encoding;  High computational cost, heavily rely on large-size dataset |
| Image classification          | Transformer + CNN                       | DeiT   (Touvron  et al., 2021)                              | Teacher-student strategy, distillation token, data  augmentation and regularization strategies;  Rely on the CNN model to be a teacher |
| Image classification          | Transformer + global attention          | T2T-ViT   (Yuan et  al., 2021)                              | Progressively structurize images to tokens, deep-narrow  structure, train from scratch on ImageNet |
| Image classification          | Transformer + global attention          | TNT   (Han et  al., 2021)                                   | Operate on patch-level and pixel-level, more diverse  feature extraction |
| Object detection              | Transformer + CNN                       | DETR   (Carion  et al., 2020)                               | End-to-end learning, set-based prediction, remove  hand-design modules;  Struggle on small objects, slow to converge |
| Object detection              | Transformer + CNN + efficient attention | Deformable DETR   (Zhu et  al., 2020)                       | Better performance on small objects, faster convergence  than DETR;  Two stage detector architecture, use data augmentation in  test time |
| Segmentation                  | Transformer + CNN                       | MaX-DeepLab (H. Wang,  Zhu, Adam, et al., 2020)             | ﬁrst end-to-end model for panoptic segmentation, mask  transformer, bipartite matching, dual-path architecture |
| Segmentation                  | Transformer + CNN                       | VisTR   (Y. Wang  et al., 2020)                             | instance sequence matching and segmentation, direct  end-to-end parallel sequence prediction |
| Image processing              | Transformer + CNN                       | Image processing transformer  (IPT) (H. Chen  et al., 2020) | The multi-head and multi-trail structure, generate a large  amount of corrupted image pairs images by add noising, rain streak, or down-sampling;  can be used in different tasks in one model after ﬁne-tuning |
| Image colorization            | Transformer + global attention          | Colorization Transformer (ColTran) (Kumar et  al., 2020)    | Conditional transformer layers, two parallel networks to upsample  low resolution images |
| Segmentation                  | Transformer + efficient attention       | Criss-Cross Network (CCNet)                                 | Criss-Cross Attention block, high efficiency and low  computation, The state-of-the-art performance, |
| NLP, architecture exploration | Transformer + efficient attention       | ConvBERT (Jiang et al., 2020)                               | Span-based dynamic  convolution, mixed attention block to capture both global and local dependencies |

 

# 5.  Future work

Transformer is still a young field and has different architectures proposed in almost every week. There are several important problems or difficulties that are worth to be noticed. 

The first one is to discover the mechanism of transformer. Since Google proposed transformer architecture in 2017, countless papers have been using this idea in different fields, from NLP to CV or to medical researches. But recently, Google also published a paper which points out that “Attention is not all you need: pure attention loses rank doubly exponentially with depth”. They decompose the outputs of self-attention layer into a sum of smaller terms and found that skip connections or multi-layer perceptron are vital to the final performance. We need to understand why and how transformer works. Only when we have a better understanding about that, we can know the direction of improvement and development. 

The second one is to build efficient vision transformer. There are several ways to accomplish that. For example, we can take advantages from convolution operations which has inductive biases, including translation invariance, weight sharing, scale invariance. Or we can build effective attention block to capture different scale information in a better way. Or we can discover the efficient to use training strategies (data augmentation or regularization) to avoid the reliance on large-size dataset. This problem is the core research direction for vision transformer for a long time since in practical we generally cannot find such large dataset. 

The third one is to reduce model complexity. Small and efficient model can be easily extended in different fields. Distillation maybe a choice to make transformer models more feasible in practice. Some useful techniques such as selective or sparse attention, low-rank matrix in attention, and efficient cluster attention may give some insights to solve this problem.

The fourth one is to explore more backbone architecture for visual inputs instead of directly using transformer structure in NLP tasks. The last one is to discover more models that can do multiple tasks in one model. 

# 6.  Conclusion

In this survey, I give a comprehensive review about vision transformer. I introduce the development of the depth exploration in CNN and go through the researches in channel and spatial attention. I also introduce the all related modules in transformer structure, including self-attention, MLP, FFNN, and layer normalization. Then the application of vision transformer and their classification and comparison are discussed. I also list some potential and useful future research directions at the end.

# 7.  Course feedbacks

This course is a great graduate-level class. There are several reasons for that:

\1.   **Rich course content.** From PCA to Bayesian, all important topics are covered in this 9-week course. It is amazing and really give me a good overview in this field. 

\2.   **Excellent lectures.** Even though I toke class by watching video, I can feel that professor try to explain concept in detail rather quickly go through various concepts and equations. For some hard concepts or math-heavy part, professor is always willing to draw some lines or circles in patient, which really help me to understand them.

\3.   **Novel and creative illustration**. Even though I think I have learned PCA from other course or videos, I am surprised to learn new things from professor’s explanation. More importantly, most of other courses only cover PCA and do not introduce LDA. Only after this class, I finally understand that when do I need to choose PCA or LDA. Another example is backpropagation. This is my first time to learn sensitivity in backpropagation, which really help me understand more than other courses. I never heard it, although I have learned some course or read some books before.

\4.   **Excellent reading recommendation**. Professor have kindly given us some reading list in each lecture to help us learn more things. They are really helpful and help me to understand deeply.

\5.   For every lecture, professor always pointed out the “**take away home message**”. Personally, I really love this part. This short message really helps me to focus on the most important part and avoid loss from some relative trivial concepts.

\6.   **The slides are clear and well-organized.** I really love this class’s slides. The have a lot of sentence and organized very well, which help to in the lecture and review.

# 8.  Reference

[1] Beal, J., Kim, E., Tzeng, E., Park, D. H., Zhai, A., and Kislyuk, D. (2020). Toward Transformer-Based Object Detection. *ArXiv:2012.09958 [Cs]*. http://arxiv.org/abs/2012.09958

[2] Bengio, Y., Simard, P., and Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE Transactions on Neural Networks*, *5*(2), 157–166. https://doi.org/10.1109/72.279181

[3] Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., … Amodei, D. (2020). *Language Models are Few-Shot Learners*. https://arxiv.org/abs/2005.14165v4

[4] Cao, Y., Xu, J., Lin, S., Wei, F., and Hu, H. (2019). *GCNet: Non-Local Networks Meet Squeeze-Excitation Networks and Beyond*. 0–0. https://openaccess.thecvf.com/content_ICCVW_2019/html/NeurArch/Cao_GCNet_Non-Local_Networks_Meet_Squeeze-Excitation_Networks_and_Beyond_ICCVW_2019_paper.html

[5] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., and Zagoruyko, S. (2020). End-to-End Object Detection with Transformers. *ArXiv:2005.12872 [Cs]*. http://arxiv.org/abs/2005.12872

[6] Chen, H., Wang, Y., Guo, T., Xu, C., Deng, Y., Liu, Z., Ma, S., Xu, C., Xu, C., and Gao, W. (2020). Pre-Trained Image Processing Transformer. *ArXiv:2012.00364 [Cs]*. http://arxiv.org/abs/2012.00364

[7] Chen, M., Radford, A., Child, R., Wu, J., Jun, H., Luan, D., and Sutskever, I. (2020). Generative Pretraining From Pixels. *International Conference on Machine Learning*, 1691–1703. http://proceedings.mlr.press/v119/chen20s.html

[8] Chen, X., Duan, Y., Houthooft, R., Schulman, J., Sutskever, I., and Abbeel, P. (2016). InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets. *ArXiv:1606.03657 [Cs, Stat]*. http://arxiv.org/abs/1606.03657

[9] Cheng, J., Dong, L., and Lapata, M. (2016). Long Short-Term Memory-Networks for Machine Reading. *ArXiv:1601.06733 [Cs]*. http://arxiv.org/abs/1601.06733

[10]         Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. (2009). Imagenet: A large-scale hierarchical image database. *2009 IEEE Conference on Computer Vision and Pattern Recognition*, 248–255.

[11]         Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *ArXiv:1810.04805 [Cs]*. http://arxiv.org/abs/1810.04805

[12]         Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ArXiv:2010.11929 [Cs]*. http://arxiv.org/abs/2010.11929

[13]         Glorot, X., and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics*, 249–256. http://proceedings.mlr.press/v9/glorot10a.html

[14]         Goodfellow, I. (2016). *Generative Adversarial Networks (GANs)*. 86.

[15]         Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning*. MIT Press.

[16]         Han, K., Xiao, A., Wu, E., Guo, J., Xu, C., and Wang, Y. (2021). Transformer in Transformer. *ArXiv:2103.00112 [Cs]*. http://arxiv.org/abs/2103.00112

[17]         He, K., Zhang, X., Ren, S., and Sun, J. (2015). Deep Residual Learning for Image Recognition. *ArXiv:1512.03385 [Cs]*. http://arxiv.org/abs/1512.03385

[18]         Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., and Adam, H. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *ArXiv:1704.04861 [Cs]*. http://arxiv.org/abs/1704.04861

[19]         Hu, J., Shen, L., Albanie, S., Sun, G., and Wu, E. (2017). Squeeze-and-Excitation Networks. *ArXiv:1709.01507 [Cs]*. http://arxiv.org/abs/1709.01507

[20]         Huang, G., Liu, Z., van der Maaten, L., and Weinberger, K. Q. (2016). Densely Connected Convolutional Networks. *ArXiv:1608.06993 [Cs]*. http://arxiv.org/abs/1608.06993

[21]         Huang, Z., Wang, X., Huang, L., Huang, C., Wei, Y., and Liu, W. (2019). Ccnet: Criss-cross attention for semantic segmentation. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 603–612.

[22]         Jiang, Z., Yu, W., Zhou, D., Chen, Y., Feng, J., and Yan, S. (2020). Convbert: Improving bert with span-based dynamic convolution. *ArXiv Preprint ArXiv:2008.02496*.

[23]         Krizhevsky, A., Sutskever, I., and Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger (Eds.), *Advances in Neural Information Processing Systems 25* (pp. 1097–1105). Curran Associates, Inc. http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

[24]         Kumar, M., Weissenborn, D., and Kalchbrenner, N. (2020, September 28). *Colorization Transformer*. International Conference on Learning Representations. https://openreview.net/forum?id=5NA1PinlGFu

[25]         LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., and Jackel, L. D. (1989). Backpropagation Applied to Handwritten Zip Code Recognition. *Neural Computation*, *1*(4), 541–551. https://doi.org/10.1162/neco.1989.1.4.541

[26]         Li, X., Wang, W., Hu, X., and Yang, J. (2019). Selective Kernel Networks. *ArXiv:1903.06586 [Cs]*. http://arxiv.org/abs/1903.06586

[27]         Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. (2020). *Fourier Neural Operator for Parametric Partial Differential Equations*. https://arxiv.org/abs/2010.08895v1

[28]         Lu Chi. (2020). Fast Fourier Convolution. *Neural Information Processing Systems*, 767–774. https://proceedings.neurips.cc/paper/1987/file/c4ca4238a0b923820dcc509a6f75849b-Paper.pdf

[29]         Luong, M.-T., Pham, H., and Manning, C. D. (2015). Effective Approaches to Attention-based Neural Machine Translation. *ArXiv:1508.04025 [Cs]*. http://arxiv.org/abs/1508.04025

[30]         Nair, V., and Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. *Proceedings of the 27th International Conference on International Conference on Machine Learning*, 807–814.

[31]         Park, J., Woo, S., Lee, J.-Y., and Kweon, I. S. (2018). BAM: Bottleneck Attention Module. *ArXiv:1807.06514 [Cs]*. http://arxiv.org/abs/1807.06514

[32]         Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *Undefined*. /paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe

[33]         Radford, Alec, Narasimhan, K., Salimans, T., and Sutskever, I. (2018). *Improving language understanding by generative pre-training*.

[34]         Redmon, J., and Farhadi, A. (2018). YOLOv3: An Incremental Improvement. *ArXiv:1804.02767 [Cs]*. http://arxiv.org/abs/1804.02767

[35]         Ren, S., He, K., Girshick, R., and Sun, J. (2016). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *ArXiv:1506.01497 [Cs]*. http://arxiv.org/abs/1506.01497

[36]         Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., and Fei-Fei, L. (2015). ImageNet Large Scale Visual Recognition Challenge. *ArXiv:1409.0575 [Cs]*. http://arxiv.org/abs/1409.0575

[37]         Simonyan, K., and Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ArXiv:1409.1556 [Cs]*. http://arxiv.org/abs/1409.1556

[38]         Srinivas, A., Lin, T.-Y., Parmar, N., Shlens, J., Abbeel, P., and Vaswani, A. (2021). Bottleneck Transformers for Visual Recognition. *ArXiv:2101.11605 [Cs]*. http://arxiv.org/abs/2101.11605

[39]         Srivastava, R. K., Greff, K., and Schmidhuber, J. (2015). Highway Networks. *ArXiv E-Prints*, *1505*, arXiv:1505.00387.

[40]         Sun, C., Shrivastava, A., Singh, S., and Gupta, A. (2017). Revisiting Unreasonable Effectiveness of Data in Deep Learning Era. *ArXiv:1707.02968 [Cs]*. http://arxiv.org/abs/1707.02968

[41]         Sun, Z., Cao, S., Yang, Y., and Kitani, K. (2020). Rethinking Transformer-based Set Prediction for Object Detection. *ArXiv:2011.10881 [Cs]*. http://arxiv.org/abs/2011.10881

[42]         Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A. (2014). Going Deeper with Convolutions. *ArXiv:1409.4842 [Cs]*. http://arxiv.org/abs/1409.4842

[43]         Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. (2016). *Rethinking the Inception Architecture for Computer Vision*. 2818–2826. http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html

[44]         Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., and Jégou, H. (2021). Training data-efficient image transformers & distillation through attention. *ArXiv:2012.12877 [Cs]*. http://arxiv.org/abs/2012.12877

[45]         Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is All you Need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.), *Advances in Neural Information Processing Systems 30* (pp. 5998–6008). Curran Associates, Inc. http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf

[46]         Wang, H., Zhu, Y., Adam, H., Yuille, A., and Chen, L.-C. (2020). MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers. *ArXiv:2012.00759 [Cs]*. http://arxiv.org/abs/2012.00759

[47]         Wang, H., Zhu, Y., Green, B., Adam, H., Yuille, A., and Chen, L.-C. (2020). Axial-deeplab: Stand-alone axial-attention for panoptic segmentation. *European Conference on Computer Vision*, 108–126.

[48]         Wang, P., Chen, P., Yuan, Y., Liu, D., Huang, Z., Hou, X., and Cottrell, G. (2018). Understanding Convolution for Semantic Segmentation. *2018 IEEE Winter Conference on Applications of Computer Vision (WACV)*, 1451–1460. https://doi.org/10.1109/WACV.2018.00163

[49]         Wang, W., Xie, E., Li, X., Fan, D.-P., Song, K., Liang, D., Lu, T., Luo, P., and Shao, L. (2021). Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions. *ArXiv:2102.12122 [Cs]*. http://arxiv.org/abs/2102.12122

[50]         Wang, X., Girshick, R., Gupta, A., and He, K. (2018). Non-local Neural Networks. *ArXiv:1711.07971 [Cs]*. http://arxiv.org/abs/1711.07971

[51]         Wang, Y., Xu, Z., Wang, X., Shen, C., Cheng, B., Shen, H., and Xia, H. (2020). End-to-End Video Instance Segmentation with Transformers. *ArXiv:2011.14503 [Cs]*. http://arxiv.org/abs/2011.14503

[52]         Woo, S., Park, J., Lee, J.-Y., and Kweon, I. S. (2018). *CBAM: Convolutional Block Attention Module*. 3–19. https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html

[53]         Yang, F., Yang, H., Fu, J., Lu, H., and Guo, B. (2020). Learning Texture Transformer Network for Image Super-Resolution. *ArXiv:2006.04139 [Cs]*. http://arxiv.org/abs/2006.04139

[54]         Yuan, L., Chen, Y., Wang, T., Yu, W., Shi, Y., Tay, F. E., Feng, J., and Yan, S. (2021). Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet. *ArXiv:2101.11986 [Cs]*. http://arxiv.org/abs/2101.11986

[55]         Zhang, D., Zhang, H., Tang, J., Wang, M., Hua, X., and Sun, Q. (2020). Feature Pyramid Transformer. *ArXiv:2007.09451 [Cs]*. http://arxiv.org/abs/2007.09451

[56]         Zhao, H., Jiang, L., Jia, J., Torr, P., and Koltun, V. (2020). Point Transformer. *ArXiv:2012.09164 [Cs]*. http://arxiv.org/abs/2012.09164

[57]         Zheng, M., Gao, P., Wang, X., Li, H., and Dong, H. (2020). End-to-End Object Detection with Adaptive Clustering Transformer. *ArXiv:2011.09315 [Cs]*. http://arxiv.org/abs/2011.09315

[58]         Zheng, S., Lu, J., Zhao, H., Zhu, X., Luo, Z., Wang, Y., Fu, Y., Feng, J., Xiang, T., Torr, P. H. S., and Zhang, L. (2020). Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers. *ArXiv:2012.15840 [Cs]*. http://arxiv.org/abs/2012.15840

[59]         Zhu, X., Su, W., Lu, L., Li, B., Wang, X., and Dai, J. (2020). Deformable DETR: Deformable Transformers for End-to-End Object Detection. *ArXiv:2010.04159 [Cs]*. http://arxiv.org/abs/2010.04159

 

 

