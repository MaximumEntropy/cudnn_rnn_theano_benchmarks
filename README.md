# cudnn-rnn-benchmarks

All benchmarks are reported for a host with the following specifications :
    
    * NVIDIA GeForce GTX TITAN X GPU

    * Intel(R) Xeon(R) CPU X5650  @ 2.67GHz

    * CUDA 7.5, cudnnv5005

These benchmarks are aimed at understanding the performance gains with using the cuDNN RNN implementation (https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/) in theano.

The benchmarks are evaluated similar to https://github.com/glample/rnn-benchmarks that compares RNN implementations in different deep learning frameworks. Results will be integrated into the above repository eventually.

Note: Results on regular RNNs cannot be compared as is between the two repositories as this benchmark uses the new theano GPU backend libgpuarray https://github.com/Theano/libgpuarray and different hardware specifications.

The Recurrent Networks take as input a 3D Tensor `batch_size x seq_length x hidden_size`
and output all hidden states, compute a MSE loss at each step and compute the gradients of error with respect to each parameter.
The `hidden_size` specifies the size of the output and input layer of the networks.

The code of the scripts we ran are available.
The code for the regular theano RNN implementations were borrowed from the rnn-benchmarks repository.

The reported `Train` time is the average time needed to run (forward, backward) for a single training example, the smaller the better.

A more exhaustive grid search will be done soon.

Note: The compile times, although not reported are much faster for the cuDNN implementation. 

## LSTM - cuDNN LSTM vs FastLSTM in rnn.py

This LSTM implementation used for these benchmarks does not use peephole connections between cell and gates.

## Depth 1

### Batch Size 32 x Seq Len 30

#### Hidden Size 128

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 255.2 | 89.9 |
| cuDNN Theano | 136.3 | 42.8 |

#### Hidden Size 512

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 581.0 | 172.7 |
| cuDNN Theano | 243.4 | 113.5 |

#### Hidden Size 1024

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 1051.8 | 322.9 |
| cuDNN Theano | 638.4 | 193.9 |

### Batch Size 128 x Seq Len 30

#### Hidden Size 128

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 231.0 | 74.0 |
| cuDNN Theano | 46.4 | 20.9 |


#### Hidden Size 512

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 471.4 | 149.7 |
| cuDNN Theano | 110.9 | 37.9 |

#### Hidden Size 1024

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 886.3 | 289.1 |
| cuDNN Theano | 204.3 | 70.8 |

## Depth 3

### Batch Size 128 x Seq Len 30

#### Hidden Size 512

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 742.1 | 419.7 |
| cuDNN Theano | 228.0 | 75.2 |


#### Hidden Size 1024

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 1405.1 | 813.0 |
| cuDNN Theano | 748.7 | 236.6 |

### Batch Size 128 x Seq Len 200

#### Hidden Size 512

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 2031.3 | 1121.5 |
| cuDNN Theano | 1436.3 | 465.1 |

#### Hidden Size 1024

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 5339.6 | 3161.8 |
| cuDNN Theano | 4719.5 | 1553.2 |
