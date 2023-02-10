import tensorflow as tf
import numpy as np
import os


class IFGSM:
    def __init__(self, iterations=1, fixed_epsilon=None):
        self.iterations = iterations
        self.fixed_epsilon = fixed_epsilon

    @tf.function
    def __call__(self, model, x, y, input_range, epsilon):
        eps = epsilon if self.fixed_epsilon is None else self.fixed_epsilon
        for i in range(self.iterations):
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = tf.reduce_mean(model.loss(y, model(x)))
            # Get the gradients of the loss w.r.t to the input image.
            gradient = tape.gradient(loss, x)
            gradient = tf.sign(gradient)
            gradient = gradient * (eps / tf.constant(self.iterations, dtype=tf.float32))

            x = x + gradient
            min_v, max_v = input_range
            if not min_v is None:
                x = tf.maximum(x, min_v)
            if not max_v is None:
                x = tf.minimum(x, max_v)
        return x


class GradientMonitor(tf.keras.callbacks.Callback):
    def __init__(
        self, model, monitor_batch, log_dir, fixed_epsilon=None, log_interval=1
    ):
        self.model = model
        self.log_interval = log_interval
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.monitor_batch_x, self.monitor_batch_y = monitor_batch
        self.ifgsm = IFGSM(1, fixed_epsilon=fixed_epsilon)

    def _append_gradients(self, grad_dict, prefix, grads):
        for i, g in enumerate(grads):
            grad_dict[f"{prefix}_{i:02d}"] = g.numpy()

    def measure(self, epoch):

        ce_loss, ce_grads = self.model.get_crossentropy_loss_and_grads(
            self.monitor_batch_x, self.monitor_batch_y, attack_method=None
        )
        adv_loss, adv_grads = self.model.get_crossentropy_loss_and_grads(
            self.monitor_batch_x, self.monitor_batch_y, attack_method=self.ifgsm
        )
        ibp_loss, ibp_grads = self.model.get_ibp_loss_and_grads(
            self.monitor_batch_x, self.monitor_batch_y, attack_method=None
        )
        grad_dict = {
            "ce_loss": ce_loss.numpy(),
            "adv_loss": adv_loss.numpy(),
            "ibp_loss": ibp_loss.numpy(),
        }
        self._append_gradients(grad_dict, "ce", ce_grads)
        self._append_gradients(grad_dict, "adv", adv_grads)
        self._append_gradients(grad_dict, "ibp", ibp_grads)
        np.savez(os.path.join(self.log_dir, f"epoch_{epoch:04d}.npz"), **grad_dict)

    def on_train_begin(self, logs=None):
        self.measure(0)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_interval == 0:
            self.measure(epoch + 1)


class BoundViolationLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred, sample_weight=None):
        # print("Called bound violation loss! y_pred: ", str(y_pred))
        loss = -tf.reduce_mean(0.5 * (1.0 - tf.sign(y_pred)) * y_pred, axis=-1)
        if not sample_weight is None:
            loss *= sample_weight
        return loss


class SparseBoundViolationLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(SparseBoundViolationLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        # print(
        #     "Called sparse bound violation loss! y_pred: ",
        #     str(y_pred),
        #     " y_true: ",
        #     str(y_true),
        # )
        mask = tf.one_hot(y_true, y_pred.shape[-1])
        # print("mask.shape",str(mask.shape))
        loss_item = 0.5 * (1.0 - tf.sign(y_pred)) * y_pred
        # print("loss_item.shape",str(mask.shape))
        loss = -tf.reduce_mean(loss_item * mask, axis=-1)
        if not sample_weight is None:
            loss *= sample_weight
        return loss


def second_to_fancy_str(elapsed):
    elapsed_mins = elapsed // 60
    seconds = int(elapsed % 60)
    mins = int(elapsed_mins % 60)
    hours = int(elapsed_mins // 60)
    return "{:02d}:{:02d}:{:02d} [h:m:s]".format(hours, mins, seconds)


def obj_to_seconds(obj):
    raise NotImplementedError("")
