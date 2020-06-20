# DCGAN Experiments with Generative Output Layer Activation Functions

### Introduction

Deep Convolutional Generative Adversarial Networks (DCGANs) were developed by Radford, Metz, and Chintala (2016). They incorporate a generator neural network which learns how to distinguish fake objects from real ones while the discriminator neural network creates new content and tries to fool the generator. (TensorFlow). Both are deep learning neural 

#### Figure 1: Generator and Discriminator Neural Networks
![A diagram of a generator and discriminator](https://www.tensorflow.org/tutorials/generative/images/gan1.png)

networks that run simultaneously (TensorFlow). During training, the generator becomes better at creating realistic looking images while the discriminator becomes better at distinguishing between real and fake images. (Tensorflow).

#### Figure 2: Example of how the Generator and Discriminator work in Tandem
![A second diagram of a generator and discriminator](https://www.tensorflow.org/tutorials/generative/images/gan2.png)
 


According to Radford et al (2016), a stable DCGAN architecture: “1. Replaces any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator), 2) uses batch norm in both the generator and the discriminator, 3) Remove fully connected hidden layers for deeper architectures. 4) uses ReLU activation in generator for all layers except for the output, which uses Tanh, and 5) uses LeakyReLU activation in the discriminator for all layers” (p.3). Additionally, “no pre-processing was applied to training images besides scaling to the range of the tanh activation function [-1, 1]” (Radford et al, 2016). 





#### Figure 3: DCGAN Generator Architecture
![Deeper into DCGANs - Towards Data Science](https://miro.medium.com/max/2504/1*5ALjnfAqwcWbOsledTBXsw.png)
 

#### Activation Functions in Deep Learning Output Layers
According to Nwankpa et al (2018), “activation functions are functions used in neural networks to computes the weighted sum of input and biases to  decide  if  a  neuron  can  be  fired  or  not” and  they “manipulate  the  presented  data  through  some  gradient  processing  usually gradient descent and afterwards produce an output for the neural network, that contains the parameters in the data” (p.3). The activation function specifically in the output “perform classifications or predictions, with associated probabilities” (p.4). 
The hyperbolic tangent function known as tanh function, is a smoother zero-centered  function  whose  range  lies  between  -1  to1. The tanh function is preferred over the sigmoid function because if provides better performance in multi-layer neural networks (Nwankpa et al, 2018, p.7). The sigmoid is a non-linear logistic function that is used frequently in output layers predicting probabilities, specifically in shallow networks (Nwankpa et al, 2018, p.5). The exponential linear units (ELUs) is another type of activation function used to speed up the training of deep neural networks. They can alleviate the vanishing gradient problem by  using positive  values  and  also  improves  the  learning  characteristics. They also have negative  values  which  allows for  pushing  of  mean  unit  activation  closer  to  zero  thereby  reducing  computational  complexity  thereby  improving  learning speed (p.11). The scaled exponential  linear units (SELU)  is  another  variant  of  the  ELUs. The SELU was introduced as a self-normalizing neural network that induces self-normalizing properties. It has a close to zero mean and unit variance that converges towards zero mean and unit  variance  when  propagated  through  multiple  layers  during  network  training making  it  suitable  for  deep  learning application and with strong regularization, learns robust features efficiently (p.12).

Figure 4: Common Activation Functions and their Derivatives
![Comparing Activation Functions in Neural Networks](https://miro.medium.com/max/1215/1*3HV9Es0CMHUuLqBoG4XjYQ.png)
  

### Experiments
For this study, four different experiments were done using the DCGAN. The starter code for the DCGAN used for these was from TensorFlow (https://www.tensorflow.org/tutorials/generative/dcgan). The code was initiated in Google CoLab to specifically use the available GPU so that the running of the code was relatively done within 2 minutes or less. Specifically, I wanted to see the effects of changing the generative activation function at the output layer so see the specific outcomes. To do this, I had the DCGAN use the MNIST dataset. I first used the Tahn function, then the ELU, SELU, and Sigmoid activation functions respectfully. The codes for each of these can be found at https://github.com/Rothgargeert/DCGAN-Experiments-with-Generative-Output-Layer-Activation-Functions

### Dataset Used
The specific dataset for this experiment I used is the Modified National Institutes of Technology or MNIST dataset. This dataset was developed from two different data sets at the National Institute of Standards and Technology which specifically contain binary images of handwritten numbers (Siddique et al, 2019). 50% of the handwritten numbers are from Census Bureau employees while the other 50% are from high school students (Siddique et al, 2019). The dataset is frequently used by scientists to test the effectiveness of specific neural networks (Siddique et al, 2019). The dataset includes 60,000 images for training and 10,000 for testing. Each image is 28x28 pixels which can be flattened to a 28*28=784 dimensional vector which in turn are a binary value that can describe the intensity of the pixel (Siddique et al, 2019). 

### Brief Review of Previous Work/Research on Dataset
Siddique et al (2019) experimented on the MNIST testing which six different types of cases with different combinations of hidden layers. They found that of all the cases, the one that had the highest accuracy rate of 99.21%, used 15 epochs, 100 batches, with a structure including: Conv1, pool1, Conv2, pool2 with 2 dropouts. Chen et al (2018) tested four different neural network models on the MNIST dataset. These included: CNN, ResNet, DenseNet, and Capsnet (p.1). They found that Capsnet performed the best overall. The Capsnet architecture includes: 1 convolutional layer (256, 9x9 convolution kernels, stride of 1, and ReLU activation), PrimaryCaps(32 capsules), output (6632 eight-dimensional vector), and DigitCaps(10 digital capsules which represents the prediction of a number) (p.3). Harmon and Klabjan (2017) use MNIST to examine which activation ensembles work best in specific types of neural networks.

### Why Deep Learning Rather than Classical Techniques
According to Mahony et al (2019), deep learning (DL) is effective for difficult image processing problems such as image [colourization], classification, segmentation and detection. Traditional computer vision (CV) techniques are best for: robotics, augmented reality, automatic panorama stitching, virtual reality, and 3D modeling. Thus, DL is the best technique for the intended experiments here concerning the MNIST dataset and the effectiveness of DCGAN which focuses specifically on image classification.

### Experimental Findings
When I ran the code using the tahn function in the generative output layer, the generator/discriminator results were easy to read, realistically created images that resembled numbers that the DCGAN learned from the MNIST dataset. When I ran the ELU and SELU functions separately in the generative output layer, they produced easy to read, realistic looking number images, but they were slightly blurry. When I ran the sigmoid function in the generative output layer, the generative ouput layer began to work, but got “stuck” after a few seconds.

### Conclusion
Since the tahn, ELU, and SELU are able to accept not only positive, but negative numbers as well, it appears that they were all able to create realistic looking numbers clearly. However, the ELU and SELU created slightly blurry numbers which could mean that being zero-centered like the tahn function improves the images better because it aids in the backpropagation process (Nwankpa et al, 2018, p.7) than being non-zero centered like the ELU and SELU functions. The sigmoid function got “stuck” after a few seconds and did not produce any realistic looking numbers at all. This could be that since it does not accept negative numbers the function stopped responding because non-zero centered output is causing the gradient updates to propagate in different  directions (p.5). Also, the sigmoid function has other “drawbacks” including sharp, damp gradients during backpropagation from deeper hidden layers to the input layers, gradient saturation, and slow convergence (p.5).

### References

Chen, F., Chen, N., Mao, H., & Hu, H. (2018). Assessing four neural networks on handwritten 
digit recognition dataset (MNIST). Chuangxinban Journal of Computing, arXiv:1811.08278

Goodfellow, I.,  Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley,D., Ozair, S., Courville, A.,  
Bengio, Y. (2014). Generative adversarial networks. arXiv:1406.2661

Harmon & Klubjan (2017). Activation ensembles for neural networks. arXiv:1702.07790v1

Nwankpa, C.E., Ijomah, W., Gachagan, A., and Marshall, S. (2018). Activation functions: 
Comparison of trends in practice and research for deep learning. arXiv:1811.03378v1 

O’Mahony N. et al. (2020) Deep Learning vs. Traditional Computer Vision. In: Arai K., Kapoor S. 
(eds) Advances in Computer Vision. CVC 2019. Advances in Intelligent Systems and Computing, vol 943. Springer, Cham

Palvanov, A. & Cho, Y.I. (2018). Comparisons of deep learning algorithms for MNIST in real-time 
environment. International Journal of Fuzzy Logic and Intelligent Systems, 18, 2.

Radford, A., Metz, L., Shintala, S. (2016). Unsupervised representation learning with deep 
convolutional generative adversarial networks. arXiv:1511.06434v2 [cs.LG]

Siddique, F.,  Sakib, S., & Siddique, M.A.B. (2019). Recognition of handwritten digit using 
convolutional neural network in Python with Tensorflow and comparison of performance of hidden layers. 5th Annual Conference of Advances in Electrical Engineering. arXiv:1909.08490

