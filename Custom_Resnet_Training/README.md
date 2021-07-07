# S11_Super_Convergence
Super Convergence Using One Cycle LR policy on Custom Resnet and Ciphar10 dataset

In this assignment, CIPHAR-10 dataset is trained using Custom ResNet architecture, Data Augmentation(Padding, Cutout,Random Crop, Horizontal Flip) and One Cycle Learning Rate policy and achieved 91.05% test accuracy in 24 epochs. Developed APIs so as to load data, train, test and show results.

Custom ResNet architecture is as below:
Layer1 -
    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    Add(X, R1)
    
Layer 2 -
    Conv 3x3 [256k]
    MaxPooling2D
    BN
    ReLU
    
Layer 3 -
    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    Add(X, R2)
    
MaxPooling with Kernel Size 4

FC Layer 

SoftMax

Following changes are done:

1. Added Data Augmentations (Albumentatons)

i. Padding (4)

ii. RandomCrop

iii. Cutout

iv. Horizontal Flip

2. Plotted the Cyclic LR over iterations

3. Performed LR Range Test for various maximum LR values for test accuracy in 5 epochs

4. Used One Cycle learning strategey for scheduling learning rates

Got best test accuracy(23rd epoch) : 91.05%

Logs are as below:

  0%|          | 0/98 [00:00<?, ?it/s]

EPOCH: 1

/content/models/CustomResNet.py:77: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(out)
/usr/local/lib/python3.6/dist-packages/torch/optim/lr_scheduler.py:122: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
Loss=1.3158868551254272 Batch_id=97 Accuracy=40.32: 100%|██████████| 98/98 [01:39<00:00,  1.02s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 1.4000, Accuracy: 5455/10000 (54.55%)

Test Accuracy: 54.55 has increased. Saving the model
EPOCH: 2

Loss=0.985808789730072 Batch_id=97 Accuracy=56.92: 100%|██████████| 98/98 [01:39<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 1.0920, Accuracy: 6232/10000 (62.32%)

Test Accuracy: 62.32 has increased. Saving the model
EPOCH: 3

Loss=1.187458872795105 Batch_id=97 Accuracy=66.13: 100%|██████████| 98/98 [01:39<00:00,  1.02s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 1.7496, Accuracy: 5917/10000 (59.17%)

EPOCH: 4

Loss=1.0295690298080444 Batch_id=97 Accuracy=69.68: 100%|██████████| 98/98 [01:39<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 1.0893, Accuracy: 6820/10000 (68.20%)

Test Accuracy: 68.2 has increased. Saving the model
EPOCH: 5

Loss=0.6749715209007263 Batch_id=97 Accuracy=74.11: 100%|██████████| 98/98 [01:38<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.7453, Accuracy: 7531/10000 (75.31%)

Test Accuracy: 75.31 has increased. Saving the model
EPOCH: 6

Loss=0.6637411713600159 Batch_id=97 Accuracy=76.27: 100%|██████████| 98/98 [01:38<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.8636, Accuracy: 7511/10000 (75.11%)

EPOCH: 7

Loss=0.5989397764205933 Batch_id=97 Accuracy=81.58: 100%|██████████| 98/98 [01:38<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.6287, Accuracy: 8044/10000 (80.44%)

Test Accuracy: 80.44 has increased. Saving the model
EPOCH: 8

Loss=0.4172368347644806 Batch_id=97 Accuracy=83.72: 100%|██████████| 98/98 [01:38<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.6174, Accuracy: 8014/10000 (80.14%)

EPOCH: 9

Loss=0.46466636657714844 Batch_id=97 Accuracy=85.95: 100%|██████████| 98/98 [01:38<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.5447, Accuracy: 8400/10000 (84.00%)

Test Accuracy: 84.0 has increased. Saving the model
EPOCH: 10

Loss=0.3375796973705292 Batch_id=97 Accuracy=86.99: 100%|██████████| 98/98 [01:38<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.4644, Accuracy: 8538/10000 (85.38%)

Test Accuracy: 85.38 has increased. Saving the model
EPOCH: 11

Loss=0.2578215003013611 Batch_id=97 Accuracy=88.97: 100%|██████████| 98/98 [01:38<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.4134, Accuracy: 8687/10000 (86.87%)

Test Accuracy: 86.87 has increased. Saving the model
EPOCH: 12

Loss=0.34829118847846985 Batch_id=97 Accuracy=90.12: 100%|██████████| 98/98 [01:38<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.4614, Accuracy: 8658/10000 (86.58%)

EPOCH: 13

Loss=0.219451904296875 Batch_id=97 Accuracy=90.78: 100%|██████████| 98/98 [01:38<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3679, Accuracy: 8852/10000 (88.52%)

Test Accuracy: 88.52 has increased. Saving the model
EPOCH: 14

Loss=0.25894445180892944 Batch_id=97 Accuracy=92.35: 100%|██████████| 98/98 [01:39<00:00,  1.02s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3522, Accuracy: 8918/10000 (89.18%)

Test Accuracy: 89.18 has increased. Saving the model
EPOCH: 15

Loss=0.23112605512142181 Batch_id=97 Accuracy=93.17: 100%|██████████| 98/98 [01:39<00:00,  1.02s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3666, Accuracy: 8882/10000 (88.82%)

EPOCH: 16

Loss=0.17997615039348602 Batch_id=97 Accuracy=93.87: 100%|██████████| 98/98 [01:39<00:00,  1.02s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3526, Accuracy: 8908/10000 (89.08%)

EPOCH: 17

Loss=0.16690823435783386 Batch_id=97 Accuracy=94.38: 100%|██████████| 98/98 [01:39<00:00,  1.02s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3185, Accuracy: 9040/10000 (90.40%)

Test Accuracy: 90.4 has increased. Saving the model
EPOCH: 18

Loss=0.13394024968147278 Batch_id=97 Accuracy=95.28: 100%|██████████| 98/98 [01:39<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3053, Accuracy: 9057/10000 (90.57%)

Test Accuracy: 90.57 has increased. Saving the model
EPOCH: 19

Loss=0.1790371686220169 Batch_id=97 Accuracy=95.45: 100%|██████████| 98/98 [01:39<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3057, Accuracy: 9059/10000 (90.59%)

Test Accuracy: 90.59 has increased. Saving the model
EPOCH: 20

Loss=0.12321416288614273 Batch_id=97 Accuracy=95.63: 100%|██████████| 98/98 [01:39<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3055, Accuracy: 9057/10000 (90.57%)

EPOCH: 21

Loss=0.09736372530460358 Batch_id=97 Accuracy=95.80: 100%|██████████| 98/98 [01:38<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3034, Accuracy: 9084/10000 (90.84%)

Test Accuracy: 90.84 has increased. Saving the model
EPOCH: 22

Loss=0.09842408448457718 Batch_id=97 Accuracy=95.91: 100%|██████████| 98/98 [01:39<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3012, Accuracy: 9093/10000 (90.93%)

Test Accuracy: 90.93 has increased. Saving the model
EPOCH: 23

Loss=0.11666856706142426 Batch_id=97 Accuracy=96.17: 100%|██████████| 98/98 [01:38<00:00,  1.01s/it]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.2990, Accuracy: 9105/10000 (91.05%)

Test Accuracy: 91.05 has increased. Saving the model
EPOCH: 24

Loss=0.14368516206741333 Batch_id=97 Accuracy=96.25: 100%|██████████| 98/98 [01:39<00:00,  1.01s/it]


Test set: Average loss: 0.2989, Accuracy: 9091/10000 (90.91%)




