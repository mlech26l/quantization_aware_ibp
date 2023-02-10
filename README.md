# Quantization-aware Interval Bound Propagation for Training Certifiably Robust Quantized Neural Networks

![Paper](https://arxiv.org/pdf/2211.16187.pdf)

**Requirements**

- python3.8
- Tensorflow 2.8

## MNIST and Fashion-MNIST training

```bash
python3 train_ibp_mnist.py --epochs 1000 --dataset mnist --model bigmnist --wd 5e-5
python3 train_ibp_mnist.py --epochs 1000 --dataset fashion --model deep --wd 5e-5
```

## CIFAR-10 training

```bash
python3 train_ibp_cifar.py --epochs 1000 --model cnn --wd 1e-4
python3 train_ibp_cifar.py --epochs 1000 --model cnn --wd 5e-5
python3 train_ibp_cifar.py --epochs 1000 --model cnn --wd 1e-5
```

## Running Algorithm 1

```bash
python3 eval_ibp_mnist.py --dataset mnist --model bigmnist --eps 1 --timeout 20
python3 eval_ibp_mnist.py --dataset mnist --model bigmnist --eps 4 --timeout 20
python3 eval_ibp_mnist.py --dataset fashion --model deep --eps 1 --timeout 20
python3 eval_ibp_mnist.py --dataset fashion --model deep --eps 4 --timeout 20
python3 eval_ibp_cifar.py --model cnn --eps 1 --timeout 20
python3 eval_ibp_cifar.py --model cnn --eps 4 --timeout 20
```

```bib
@article{lechner2022quantization,
  title={Quantization-aware Interval Bound Propagation for Training Certifiably Robust Quantized Neural Networks},
  author={Lechner, Mathias and {\v{Z}}ikeli{\'c}, {\DJ}or{\dj}e and Chatterjee, Krishnendu and Henzinger, Thomas A and Rus, Daniela},
  journal={arXiv preprint arXiv:2211.16187},
  year={2022}
}
```