# MNENet
project for PRDL UU

a MEG-Based Task Classifier Inspired by EEGNet

# Introduction
I implement two models to categorize different MEG signals recorded while each participant performs one of the four tasks. The prediction model aims to not only recognize patterns among a single participant but also to generalize its predictive power to new data from additional participants.

# MODEL
The architecture used in this project for EEG classification is based on the architecture EEGNet. Although there exists a clear difference between recording electric signals (EEG) instead of magnetic signals (MEG) from the brain, EEG CNN have been thoroughly applied to the field of MEG signal classification.

Basically, this architecture comprises two blocks and a final classifier. Block 1 captures EEG signals at different frequencies using a two-step convolutional sequence. The first step applies 2D convolutions with filters of size (1, 64), followed by a Depthwise Convolution to learn spatial filters. Batch Normalization along the feature map dimension is done before applying the exponential linear unit (ELU) non-linearity. Then, Dropout techniques are applied for regularization. An average pooling layer reduces the signal sampling rate to 32Hz. Block 2 uses a Separable Convolution for further feature extraction with an average pooling layer for dimension reduction.

In the classification section, features are passed directly to a Soft-max Classification layer with N units, where N represents the number of classes, in this specific case four. It is worth noting that a dense layer for feature aggregation has been omitted to reduce model parameters. The entire model is trained using the Adam optimizer and categorical cross-entropy loss function. 

In this case, it has also been decided to perform a hyperparametric adjustment of the main parameters of this architecture in order to adapt the model to our dataset and obtain the best results. However, in this case, it was decided to do one part manually and the other part automatically using Keras Bayesian tuner. This was due to the limited resources and computing power of the personal GPU. 

The hyperparameters selected manually were the filter size and the type of loss function. The filter size determines the receptive field, influencing the scale of features captured during each convolution. Larger filters provide a global perspective, while smaller filters focus on local details. The choice impacts the number of parameters, computational efficiency, and the network's ability to handle translations. On the other hand, the loss function was chosen based on the type of multi-class classification problem it is being addressed since categorical cross-entropy loss instead of a binary cross-entropy loss was more suitable for this type of approach.

The hyperparameters selected automatically were the learning rate, the type of optimizer, the regularizer value in the first Conv2D layer, and the type of activation function employed in both the first and second convolutional blocks. Another hyperparameter that is usually not included in other studies is the value of the Dropouts that are placed at the end of the first and second blocks. 

In order to prevent overfitting, other techniques such as dropouts are also widely implemented. Dropouts randomly cuts some neurons in the model, therefore, the model will try to extract more general characteristics rather than only focusing in the very intrinsic details of a single dataset. Hence, this enhances the generalizability of the model. However, if too many neurons are suddenly dropped at the same time, it can cause the model to become extremely generic which prevents it from learning important characteristics that are needed in this classification task.

Moreover, another common approach is the usage of regularization which is controlled by the regularizer value. This number informs about the strength of the penalty imposed to an increase in the value of the weights. In this case, if this value is too large, the model might not be able to efficiently learn the intrinsic characteristics of the data. Nonetheless, if this value is too small, then the model might overfit which prevents the generalizability of the design. 
