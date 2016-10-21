# cudnn-rnn-benchmarks

All benchmarks are reported for a host with the following specifications :
    * NVIDIA GeForce GTX TITAN GPU
    * Intel(R) Core(TM) i7 CPU 950 @ 3.07GHz
    * CUDA 7.5, cudnnv5005

These benchmarks are aimed at understanding the performance gains with using the cuDNN RNN implementation (https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/) in theano.

The benchmarks are evaluated similar to https://github.com/glample/rnn-benchmarks that compares RNN implementations in different deep learning frameworks. Results will be integrated into the above repository eventually.

Note: Results on regular RNNs cannot be compared as is between the two repositories as this benchmark uses the new theano GPU backend libgpuarray https://github.com/Theano/libgpuarray and different hardware specifications.

The Recurrent Networks take as input a 3D Tensor `batch_size x seq_length x hidden_size`
and output the last hidden state, compute a MSE loss and compute the gradients of error with respect to each parameter.
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
| Theano | 345.2 | 83.4 |
| cuDNN Theano | 188.1 | 78.2 |

#### Hidden Size 512

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 647.5 | 165.0 |
| cuDNN Theano | 391.8 | 144.6 |

#### Hidden Size 1024

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 1583.1 | 446.4 |
| cuDNN Theano | 929.6 | 339.1 |

### Batch Size 128 x Seq Len 30

#### Hidden Size 128

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 92.1 | 21.7 |
| cuDNN Theano | 45.7 | 20.7 |


#### Hidden Size 512

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 361.8 | 82.3 |
| cuDNN Theano | 158.2 | 49.4 |

#### Hidden Size 1024

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 874.5 | 234.1 |
| cuDNN Theano | 429.7 | 138.2 |

## Depth 3

### Batch Size 128 x Seq Len 30

#### Hidden Size 512

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 1069.3 | 565.7 |
| cuDNN Theano | 451.7 | 126.8 |


#### Hidden Size 1024

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 2656.2 | 1517.3 |
| cuDNN Theano | 1544.5 | 474.6 |

### Batch Size 128 x Seq Len 200

#### Hidden Size 512

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 3446.4 | 1998.8 |
| cuDNN Theano | 2907.6 | 841.0 |

#### Hidden Size 1024

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano | 9788.9 | 5878.7 |
| cuDNN Theano | 9983.6 | 3128.4 |
