# NN Time Sequence Prediction
## Environment
- OS  
Ubuntu 16.04.3 LTS (Xenial Xerus)　

- GPU  
NVIDIA Tesla K40　

- CUDA  
cuda driver 375.74　   
cuda 8.0   

- Python  
Python 3.6.2　   
Pytorch 0.3.0

I used a [qrnn library of Salesforce](https://github.com/salesforce/pytorch-qrnn) in my code (./models/torchqrnn/*).
```
@article{bradbury2016quasi,
  title={{Quasi-Recurrent Neural Networks}},
  author={Bradbury, James and Merity, Stephen and Xiong, Caiming and Socher, Richard},
  journal={International Conference on Learning Representations (ICLR 2017)},
  year={2017}
}
```

## Usage
```
python main.py [-h] [--model {all,rnn,lstm,qrnn,cnn}] [--epoch EPOCH]
               [--batch_size BATCH_SIZE] [--n_iter N_ITER] [--seq_len SEQ_LEN]
               [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS]
               [--result_path RESULT_PATH] [--cuda CUDA]

optional arguments:
  -h, --help                        show this help message and exit
  --model {all,rnn,lstm,qrnn,cnn}   The type of model (default: all)
  --epoch EPOCH                     The number of epochs to run (default: 300)
  --batch_size BATCH_SIZE           The number of batch (default: 200)
  --n_iter N_ITER                   The number of iteration (default: 5)
  --seq_len SEQ_LEN                 The length of sequence (default: 50)
  --hidden_size HIDDEN_SIZE         The number of features in the hidden state h (default: 20)
  --num_layers NUM_LAYERS           The number of layers (default: 2)
  --result_path RESULT_PATH         Result path (default: ./result)
  --cuda CUDA                       set CUDA_VISIBLE_DEVICES (default: None)
```

**example**   
If you set `--model all`(default), all models (rnn, lstm, cnn, qrnn) are used.
If you cannot use GPU, set `--cuda None`(default).
```
python main.py --cuda 0 --model all
```

You can also run one model.
```
python main.py --cuda 0 --model rnn 
```

## Datasets
I used summed up sin wave.

![](https://latex.codecogs.com/gif.latex?y&space;=&space;\sum_{i=1}^{3}&space;\sin(a_i\pi&space;x)&space;&plus;&space;\epsilon)


You can generate the datasets as follows:
```
python generate_data.py [-h] [--sample SAMPLE] [--path PATH]
                        [--filename FILENAME]

optional arguments:
  -h, --help           show this help message and exit
  --sample SAMPLE      The number of samples (default: 10000)
  --path PATH          Data path (default: ./data)
  --filename FILENAME  File name (default: data.pkl)
```

## Result
|Train & Test Error|Time|
|:-:|:-:|
|![error.png](https://github.com/takatex/time_series/blob/master/result/error.png)|![error.png](https://github.com/takatex/time_series/blob/master/result/time.png)|

Prediction results are as follows:

|RNN|LSTM|
|:-:|:-:|
|![](https://github.com/takatex/time_series/blob/master/result/rnn/data4.png)|![](https://github.com/takatex/time_series/blob/master/result/lstm/data4.png)|
|CNN|QRNN|
|![](https://github.com/takatex/time_series/blob/master/result/cnn/data4.png)|![](https://github.com/takatex/time_series/blob/master/result/qrnn/data4.png)|


## Thesis
### Overall
- [Recent Advances in Recurrent Neural Networks](https://arxiv.org/abs/1801.01078)

### RNN
- [Finding Structure in Time](http://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1/abstract;jsessionid=1EF2F2E583DE0106FE6FC879B42FCD93.f02t01)
- [SERIAL ORDER: A PARALLEL DISTRIBUTED PROCESSING APPROACH](http://cseweb.ucsd.edu/~gary/PAPER-SUGGESTIONS/Jordan-TR-8604.pdf)

### LSTM
- [LONG SHORT-TERM MEMORY](http://www.bioinf.jku.at/publications/older/2604.pdf)
- [Learning to Forget: Continual Prediction with LSTM](https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf)
- [Recurrent Nets that Time and Count](https://www.researchgate.net/publication/3857862_Recurrent_nets_that_time_and_count)

### CNN（TDNN）
- [Phoneme Recognition Using Time-Delay Neural Networks](http://www.cs.toronto.edu/~fritz/absps/waibelTDNN.pdf)
- [Review of TDNN Architectures for Speech Recognition](http://isl.anthropomatik.kit.edu/pdf/Sugiyama1991.pdf)
- [TDNN architecture for efficient modeling of longtemporal contexts](http://www.danielpovey.com/files/2015_interspeech_multisplice.pdf)

### QRNN
- [Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576)
