# Time-series Analysis
## Environment
Ubuntu 16.04.3 LTS (Xenial Xerus)

Tesla K40

cuda driver 375.74

cuda 

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

## Usege
```
main.py [-h] [--model {rnn,lstm,qrnn,cnn}] [--all ALL] [--epoch EPOCH]
               [--batch_size BATCH_SIZE] [--n_iter N_ITER] [--seq_len SEQ_LEN]
               [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS]
               [--result_path RESULT_PATH] [--cuda CUDA]

optional arguments:
  -h, --help                    show this help message and exit
  --model {rnn,lstm,qrnn,cnn}   The type of model (default: rnn)
  --all ALL                     run all model types (default: False)
  --epoch EPOCH                 The number of epochs to run (default: 300)
  --batch_size BATCH_SIZE       The number of batch (default: 200)
  --n_iter N_ITER               The number of iteration (default: 5)
  --seq_len SEQ_LEN             The length of sequence (default: 50)
  --hidden_size HIDDEN_SIZE     The number of features in the hidden state h (default: 20)
  --num_layers NUM_LAYERS       The number of layers (default: 2)
  --result_path RESULT_PATH     Result path (default: ./result)
  --cuda CUDA                   set CUDA_VISIBLE_DEVICES (default: None)
```

## Thesis
### Overall
- [Recent Advances in Recurrent Neural Networks](https://arxiv.org/abs/1801.01078)

### RNN

### LSTM
- [LONG SHORT-TERM MEMORY](http://www.bioinf.jku.at/publications/older/2604.pdf)
- [Recurrent Nets that Time and Count](ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf)

### CNN（TDNN）
- [Phoneme Recognition Using Time-Delay Neural Networks](http://www.cs.toronto.edu/~fritz/absps/waibelTDNN.pdf)
- [Review of TDNN Architectures for Speech Recognition](http://isl.anthropomatik.kit.edu/pdf/Sugiyama1991.pdf)
- [TDNN architecture for efficient modeling of longtemporal contexts](http://www.danielpovey.com/files/2015_interspeech_multisplice.pdf)

### QRNN
- [Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576)
