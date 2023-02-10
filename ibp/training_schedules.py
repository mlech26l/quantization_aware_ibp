import tensorflow as tf


class LossFactorSchedule:
    def __init__(self):
        pass

    def call(self, steps):
        raise NotImplementedError("Abstract class")


class ConstantSchedule(LossFactorSchedule):
    def __init__(self, constant_value):
        super().__init__()
        self.constant_value = tf.constant(constant_value, dtype=tf.float32)

    def call(self, steps):
        return self.constant_value


class LinearLossFactorSchedule(LossFactorSchedule):
    def __init__(self, start_factor, end_factor, total_train_steps):
        super().__init__()
        self.total_train_steps = tf.constant(total_train_steps, dtype=tf.int32)
        self.increments = tf.constant(end_factor - start_factor, dtype=tf.float32)
        self.offset = tf.constant(start_factor, dtype=tf.float32)

    def call(self, steps):
        progress = tf.cast(steps, tf.float32) / tf.cast(
            self.total_train_steps, tf.float32
        )
        progress = tf.clip_by_value(progress, 0, 1)
        return self.offset + self.increments * progress


# class PiecewiseLinearLossFactorSchedule(LossFactorSchedule):
#     def __init__(self, factors):
#         self.factors = factors
#         self.is_build = False

#     def call(self, steps, total_train_steps):
#         if not self.is_build:

#         progress = steps / total_train_steps
#         progress = tf.clip_by_value(progress, 0, 1)

#         return self.offset + self.increments * progress


def make_schedule(schedule):
    if schedule is None:
        return ConstantSchedule(0.0)
    elif isinstance(schedule, LossFactorSchedule):
        return schedule
    else:
        return ConstantSchedule(schedule)