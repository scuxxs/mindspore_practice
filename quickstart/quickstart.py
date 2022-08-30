from mindvision.dataset import Mnist
from mindvision.classification.models import lenet
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindvision.engine.callback import LossMonitor

download_train = Mnist(path="./mnist", split="train", batch_size=32, shuffle=True, resize=32, download=True)
download_eval = Mnist(path="./mnist", split="test", batch_size=32, resize=32, download=True)

dataset_train = download_train.run()
dataset_eval = download_eval.run()

network = lenet(num_classes=10)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)

config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
ckpoint = ModelCheckpoint(prefix="lenet", directory="./lenet", config=config_ck)

model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'acc'})
model.train(1, dataset_train, callbacks=[ckpoint, LossMonitor(0.01)])

acc=model.eval(dataset_eval)
print("{}".format(acc))
