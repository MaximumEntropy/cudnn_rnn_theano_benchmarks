# cudnn-rnn-benchmarks

All benchmarks are reported for a host with the following specifications :
    
    * NVIDIA GeForce GTX TITAN X GPU

    * Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz

    * CUDA 8.0, cudnnv5105

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

## LSTM - cuDNN LSTM and GRU vs FastLSTM in rnn.py

This LSTM implementation used for these benchmarks does not use peephole connections between cell and gates.

## Depth 1

### Batch Size 32 x Seq Len 30

#### Hidden Size 128

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano LSTM | 204.5 | 57.1 |
| cuDNN Theano LSTM | 118.8 | 59.5 |
| cuDNN Theano GRU | 117.4 | 57.6 |

#### Hidden Size 512

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano LSTM | 530.9 | 148.1 |
| cuDNN Theano LSTM | 223.2 | 102.4 |
| cuDNN Theano GRU | 184.6 | 77.6 |

#### Hidden Size 1024

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano LSTM | 1102.0 | 294.0 |
| cuDNN Theano LSTM | 601.8 | 161.1 |
| cuDNN Theano GRU | 394.8 | 136.2 |

### Batch Size 128 x Seq Len 30

#### Hidden Size 128

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano LSTM | 200.8 | 52.8 |
| cuDNN Theano LSTM | 33.4 | 15.0 |
| cuDNN Theano GRU | 32.2 | 14.4 |


#### Hidden Size 512

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano LSTM | 491.0 | 138.2 |
| cuDNN Theano LSTM | 100.8 | 31.7 |
| cuDNN Theano GRU | 83.3 | 26.5 |

#### Hidden Size 1024

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano LSTM | 1000.1 | 291.8 |
| cuDNN Theano LSTM | 221.2 | 69.0 |
| cuDNN Theano GRU | 181.3 | 59.1 |

## Depth 3

### Batch Size 128 x Seq Len 30

#### Hidden Size 512

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano LSTM | 778.3 | 418.3 |
| cuDNN Theano LSTM | 244.9 | 70.2 |
| cuDNN Theano GRU | 197.1 | 55.7 |


#### Hidden Size 1024

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano LSTM | 1592.8 | 882.7 |
| cuDNN Theano LSTM | 820.6 | 256.8 |
| cuDNN Theano GRU | 639.5 | 195.2 |

### Batch Size 128 x Seq Len 200

#### Hidden Size 512

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano LSTM | 2196.6 | 1168.1 |
| cuDNN Theano LSTM | 1539.5 | 485.9 |
| cuDNN Theano GRU | 1253.8 | 386.4 |

#### Hidden Size 1024

| Version | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano LSTM | 5711.1 | 3427.9 |
| cuDNN Theano LSTM | 5342.5 | 1692.1 |
| cuDNN Theano GRU | 4163.4 | 1274.5 |
