import tensorflow as tf
import numpy as np
from enum import Enum
import time
from tensorflow.python.keras.utils import losses_utils
from tqdm import tqdm
from ibp.training_schedules import make_schedule
import ibp.utils as ibp_utils
import ibp.building as ibp_build
import ibp.decision_procedure as ibp_decision

assert tf.__version__.startswith("2."), "Please upgrade to TensorFlow 2.x"


class QNNBoundPropModel:
    def __init__(self, ff_model):

        # Normal model
        self.ff_model = ff_model

        self._is_built = False
        self._build_training_vars = False

    def pretrain(self, *args, **kwargs):
        """
        Simple alias for model.fit
        """
        self.ff_model.fit(*args, **kwargs)

    def call_and_collect(self, lb, ub):
        last_hidden = self._pre_logit_bound_model([lb, ub])
        return last_hidden

    @property
    def number_of_classes(self):
        return int(self.ff_model.output_dim)

    def get_decision_procedure(self, lb, ub, allowed_classes):
        return ibp_decision.DecisionProcedure(
            self.ff_model,
            self._pre_logit_bound_model,
            self._output_layer,
            lb,
            ub,
            allowed_classes,
        )

    def _create_local_vars(self):
        # Always create all local vars, even if we don't need all of them
        self._regularize_internal_bounds_factor = tf.Variable(
            initial_value=0.0,
            trainable=False,
            name="regularization_loss_factor",
            dtype=tf.float32,
        )

        self._robustness_bound_margin = tf.Variable(
            initial_value=0.0,
            trainable=False,
            name="robustness_bound_margin",
            dtype=tf.float32,
        )
        self._robustness_loss_factor = tf.Variable(
            initial_value=0.0,
            trainable=False,
            name="robustness_loss_factor",
            dtype=tf.float32,
        )
        self._robustness_epsilon = tf.Variable(
            initial_value=0.0,
            trainable=False,
            name="robustness_epsilon",
            dtype=tf.float32,
        )

    def compile(
        self,
        optimizer,
        allowed_classes=None,
        robustify_samples=None,
        robustness_input_min_max=None,
        regularize_internal_bounds=None,
    ):
        """
        @allowed_classes: Either of type
            list:           Single specification provided as set of allowed classes
            nested-list:    Multiple specifications provides as a list of set of allowed classes.
                            Which spec should be used/optimized depend on the label
        """
        # Check if parameters are valid
        robustify_samples_arg = [None, "all", "correct_only"]
        if not robustify_samples in robustify_samples_arg:
            raise ValueError(
                "Unknown roustify_samples argumnet '{}', exptected one of '{}'".format(
                    robustify_samples, str(robustify_samples_arg)
                )
            )
        # Create locals vars and store compliation parameters
        self._allowed_classes = allowed_classes
        self._robustify_samples = robustify_samples
        self._regularize_internal_bounds = regularize_internal_bounds
        self._allowed_classes = allowed_classes
        if robustness_input_min_max is None:
            # Make tuple of lower and upper limits
            robustness_input_min_max = (None, None)
        self._robustness_input_min_max = robustness_input_min_max
        self._create_local_vars()

        self.optimizer = optimizer

        # Base bound prop model
        (
            self._pre_logit_bound_model,
            self._output_layer,
            self._reg_loss,
        ) = ibp_build.build_bound_model(self.ff_model)

        # Create robustness bound prop model out of base model
        allowed_classes = [[i] for i in range(self.ff_model.output_dim)]
        robust_bound_model = ibp_build.add_spec_list_to_bound_model(
            self._pre_logit_bound_model,
            self._output_layer,
            allowed_classes,
            reduce_max=False,
            bound_margin=self._robustness_bound_margin,
        )
        self.robustness_bound_model = tf.keras.Model(
            inputs=robust_bound_model.inputs,
            outputs=[robust_bound_model.outputs[0], self._reg_loss],
        )
        self.robust_loss = ibp_utils.SparseBoundViolationLoss()

        self._sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        # Initialize variables needed for scheduling during training
        self._current_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self._total_train_loss = tf.Variable(0, dtype=tf.float32, trainable=False)
        self._total_robust_loss = tf.Variable(0, dtype=tf.float32, trainable=False)
        self._total_reg_loss = tf.Variable(0, dtype=tf.float32, trainable=False)

        self._is_built = True

    def get_robustness_input(self, x, epsilon):
        lb = x - epsilon
        ub = x + epsilon
        if not self._robustness_input_min_max is None:
            input_min, input_max = self._robustness_input_min_max
            if not input_min is None:
                lb = tf.maximum(lb, tf.constant(input_min, dtype=tf.float32))
            if not input_max is None:
                ub = tf.minimum(ub, tf.constant(input_max, dtype=tf.float32))
        return (lb, ub)

    # def get_crossentropy_loss_and_grads(self, x, y, w=None, attack_method=None):
    #     perturbed_x = x
    #     if not attack_method is None:
    #         perturbed_x = attack_method(
    #             self.model,
    #             x,
    #             y,
    #             self._robustness_input_min_max,
    #             self._robustness_epsilon,
    #         )
    #
    #     with tf.GradientTape() as tape:
    #         # Training loss
    #         logits = self.model(perturbed_x)
    #         train_loss = self.model.loss(y, logits, sample_weight=w)
    #         train_loss = tf.reduce_mean(train_loss)
    #     grads = tape.gradient(train_loss, self.model.trainable_variables)
    #     return train_loss, grads

    # def get_ibp_loss_and_grads(self, x, y, w=None, attack_method=None):
    # with tf.GradientTape() as tape:
    #     # Training loss
    #     # Bound prop for optimizing robustness is enabled
    #     x_eps = self.get_robustness_input(x, self._robustness_epsilon)
    #     robust_bounds, reg_loss = self.robustness_bound_model(x_eps)
    #     robust_loss = self.robust_loss(y, robust_bounds, sample_weight=w)
    #     if self._robustify_samples == "correct_only":
    #         # Only optimize robustness bounds on samples that are
    #         # classified correctly
    #         perturbed_x = x
    #         if not attack_method is None:
    #             perturbed_x = attack_method(
    #                 self.model,
    #                 x,
    #                 y,
    #                 self._robustness_input_min_max,
    #                 self._robustness_epsilon,
    #             )
    #         logits = self.model(perturbed_x)
    #
    #         predictions = tf.argmax(logits, axis=-1)
    #         correct_mask = tf.cast(
    #             tf.equal(predictions, tf.cast(y, dtype=tf.int64)),
    #             dtype=tf.float32,
    #         )
    #         # Mask robust_loss with inverted 0/1-loss
    #         robust_loss = robust_loss * correct_mask
    #     robust_loss = tf.reduce_mean(robust_loss)
    # grads = tape.gradient(robust_loss, self.model.trainable_variables)
    # return robust_loss, grads

    @tf.function
    def compute_loss_and_grads(self, x, y, w=None, attack_method=None):
        # Running attack method should not be backpropagated
        perturbed_x = x
        if not attack_method is None:
            perturbed_x = attack_method(
                self.ff_model,
                x,
                y,
                self._robustness_input_min_max,
                self._robustness_epsilon,
            )

        with tf.GradientTape() as tape:
            # Training loss
            logits = self.ff_model(perturbed_x)
            train_loss = self.ff_model.loss(y, logits, sample_weight=w)
            train_loss = tf.reduce_mean(train_loss)
            total_loss = train_loss

            # Robustness loss
            robust_loss = tf.constant(-1, dtype=tf.float32)
            reg_loss = tf.constant(-1, dtype=tf.float32)
            if not self._robustify_samples is None:
                print("TRAIN ON ROBUST MODE")
                # Bound prop for optimizing robustness is enabled
                x_eps = self.get_robustness_input(x, self._robustness_epsilon)
                robust_bounds, reg_loss = self.robustness_bound_model(x_eps)
                # print("train_step traced")
                # print("robust_bounds shape: ",str(robust_bounds.shape))
                # print("y shape: ",str(y.shape))
                robust_loss = self.robust_loss(y, robust_bounds, sample_weight=w)
                if self._robustify_samples == "correct_only":
                    # Only optimize robustness bounds on samples that are
                    # classified correctly
                    predictions = tf.argmax(logits, axis=-1)
                    correct_mask = tf.cast(
                        tf.equal(predictions, tf.cast(y, dtype=tf.int64)),
                        dtype=tf.float32,
                    )
                    # Mask robust_loss with inverted 0/1-loss
                    robust_loss = robust_loss * correct_mask

                robust_loss = tf.reduce_mean(robust_loss)
                total_loss = total_loss + self._robustness_loss_factor * robust_loss
                if self._regularize_internal_bounds:
                    total_loss += reg_loss * self._regularize_internal_bounds_factor

            # Update metric
            # TODO: Is step might be optional, but shouldn't cause any bugs
            self._sparse_categorical_accuracy(y, logits)
        grads = tape.gradient(total_loss, self.ff_model.trainable_variables)
        return total_loss, train_loss, robust_loss, reg_loss, grads

    @tf.function
    def train_step(self, x, y, w=None, attack_method=None):
        (
            total_loss,
            train_loss,
            robust_loss,
            reg_loss,
            grads,
        ) = self.compute_loss_and_grads(x, y, w, attack_method)
        self.optimizer.apply_gradients(zip(grads, self.ff_model.trainable_variables))
        return train_loss, robust_loss, reg_loss

    def write_checkpoint(self, path):
        ckpt = tf.train.Checkpoint(
            step=self._current_step, optimizer=self.optimizer, net=self.ff_model
        )
        ckpt.write(path)
        # ckpt.save(path)

    def restore_checkpoint(self, path):
        ckpt = tf.train.Checkpoint(
            step=self._current_step, optimizer=self.optimizer, net=self.ff_model
        )
        ckpt.read(path).assert_existing_objects_matched()
        # ckpt.restore(path).assert_existing_objects_matched()
        # ckpt.restore(path).assert_consumed()

    @tf.function
    def eval_step(self, x, y, w=None, attack_method=None):

        # Running attack method should not be backpropagated
        perturbed_x = x
        if not attack_method is None:
            perturbed_x = attack_method(
                self.ff_model,
                x,
                y,
                self._robustness_input_min_max,
                self._robustness_epsilon,
            )

        # Training loss
        logits = self.ff_model(perturbed_x)
        # train_loss = self.model.loss(y,logits,sample_weight=w)
        train_loss = self.ff_model.loss(y, logits)
        train_loss = tf.reduce_mean(train_loss)
        total_loss = train_loss

        # Robustness loss
        robust_loss = tf.constant(-1, dtype=tf.float32)
        reg_loss = tf.constant(-1, dtype=tf.float32)
        if not self._robustify_samples is None:
            x_eps = self.get_robustness_input(x, self._robustness_epsilon)
            robust_bounds, reg_loss = self.robustness_bound_model(x_eps)
            robust_loss = self.robust_loss(y, robust_bounds, sample_weight=w)
            if self._robustify_samples == "correct_only":
                predictions = tf.argmax(logits, axis=-1)
                correct_mask = tf.cast(
                    tf.equal(predictions, tf.cast(y, dtype=tf.int64)), dtype=tf.float32
                )
                # Mask robust_loss with inverted 0/1-loss
                robust_loss = robust_loss * correct_mask
            robust_loss = tf.reduce_mean(robust_loss)
            total_loss = total_loss + self._robustness_loss_factor * robust_loss
            if self._regularize_internal_bounds:
                total_loss += reg_loss * self._regularize_internal_bounds_factor

        # Update metric
        self._sparse_categorical_accuracy(y, logits)

        return train_loss, robust_loss, reg_loss

    def _evaluate(self, dataset):
        self._sparse_categorical_accuracy.reset_states()
        total_loss = 0
        # Run a validation loop at the end of each epoch.
        for step, batch in enumerate(dataset):
            x, y = batch
            logits = self.ff_model(x)
            loss = self.ff_model.loss(y, logits)
            loss = tf.reduce_mean(loss)

            # Update metric
            self._sparse_categorical_accuracy(y, logits)

            total_loss += loss
        metrics = [metric.result() for metric in self.metrics]
        metrics.insert(0, total_loss / (step + 1))
        return metrics

    def evaluate(self, dataset, steps=None, verbose=None):
        if not self._is_built:
            raise ValueError(
                "BoundProp model is not compiled yet. Call .compile() with optimizer and spec loss"
            )

        # Initialize progressbar
        callbacks = []
        metrics_names = [
            "loss",
            "accuracy",
            "robust_loss",
            "reg_loss",
            "val_loss",
            "val_accuracy",
            "val_robust_loss",
            "val_reg_loss",
        ]
        if verbose:
            progbar = tf.keras.callbacks.ProgbarLogger(
                count_mode="steps", stateful_metrics=list(metrics_names).append("loss")
            )
            progbar.set_params(
                {"epochs": 1, "steps": steps, "verbose": 1, "metrics": metrics_names}
            )
            callbacks.append(progbar)
        # Convert safety domains into python iterator

        # Begin eval
        for c in callbacks:
            c.on_test_begin()

        # Reset metrics
        self._sparse_categorical_accuracy.reset_states()

        self._total_train_loss.assign(0)
        self._total_robust_loss.assign(0)
        self._total_reg_loss.assign(0)

        # Train on the mini-batches
        _debug_list = []
        for step, batch in enumerate(dataset):
            for c in callbacks:
                c.on_test_batch_begin(batch)

            x, y = batch

            # Everything from here on will be traced
            train_loss, robust_loss, reg_loss = self.eval_step(x, y, None)

            self._total_train_loss.assign_add(train_loss)
            self._total_robust_loss.assign_add(robust_loss)
            self._total_reg_loss.assign_add(reg_loss)
            if step < 5:
                _debug_list.append(robust_loss.numpy())

            logs = {}
            logs["accuracy"] = self._sparse_categorical_accuracy.result()
            logs["loss"] = self._total_train_loss / tf.cast(step + 1, dtype=tf.float32)
            logs["robust_loss"] = self._total_robust_loss / tf.cast(
                step + 1, dtype=tf.float32
            )
            logs["reg_loss"] = self._total_reg_loss / tf.cast(
                step + 1, dtype=tf.float32
            )
            for c in callbacks:
                c.on_test_batch_end(batch, logs)

            if (not steps is None) and step >= steps:
                break
        # print("Valid Robustness debug list: ",str(_debug_list))
        # print("Valid Robustness debug mask: ",str(_debug_mask))

        for c in callbacks:
            c.on_test_end(logs)
        return logs

    def _update_schedules(self, current_steps, all_schedules):
        for schedule, tf_vars in all_schedules:
            new_value = schedule.call(current_steps)
            tf_vars.assign(new_value)

    def _print_scheduled_variables(self, current_epoch, total_epochs):
        print("Schedule of epoch {}/{}:".format(current_epoch, total_epochs))
        print(
            "   Robust loss factor: {:0.5f}".format(
                self._robustness_loss_factor.numpy()
            )
        )
        print(
            "   Robust epsilon    : {:0.5f}".format(
                self._robustness_epsilon.numpy(),
            )
        )
        print(
            "   Robust margin     : {:0.5f}".format(
                self._robustness_bound_margin.numpy()
            )
        )
        print(
            "   Reg loss factor   : {:0.5f}".format(
                self._regularize_internal_bounds_factor.numpy()
            )
        )
        print()

    # https://www.tensorflow.org/guide/keras/train_and_evaluate
    def fit(
        self,
        train_data,
        attack_method=None,
        epochs=1,
        verbose_schedule=None,
        steps_per_epoch=None,
        valid_data=None,
        validation_steps=None,
        robustness_epsilon=None,
        robustness_loss_factor=None,
        robustness_bound_margin=None,
        regularization_factor=None,
        callbacks=None,
    ):

        if not self._is_built:
            raise ValueError(
                "BoundProp model is not compiled yet. Call .compile() with optimizer and spec loss"
            )
        # Initialize progressbar
        callbacks = callbacks or []
        train_metrics_name = [
            "loss",
            "accuracy",
            "robust_loss",
            "reg_loss",
        ]
        metrics_names = [s for s in train_metrics_name]
        metrics_names.extend("val_" + s for s in train_metrics_name)

        progbar = tf.keras.callbacks.ProgbarLogger(
            count_mode="steps", stateful_metrics=metrics_names
        )
        progbar.set_params(
            {
                "epochs": epochs,
                "steps": steps_per_epoch,
                "verbose": 1,
                "metrics": metrics_names,
            }
        )
        callbacks.append(progbar)
        # Convert safety domains into python iterator

        # Unify constants, tupes, and schedules into schedules
        robustness_epsilon = make_schedule(robustness_epsilon)
        robustness_loss_factor = make_schedule(robustness_loss_factor)
        regularization_factor = make_schedule(regularization_factor)
        robustness_bound_margin = make_schedule(robustness_bound_margin)
        # Assign schedules to variables
        schedule_assignment = [
            (robustness_epsilon, self._robustness_epsilon),
            (robustness_loss_factor, self._robustness_loss_factor),
            (regularization_factor, self._regularize_internal_bounds_factor),
            (robustness_bound_margin, self._robustness_bound_margin),
        ]
        # Reset variables
        self._current_step.assign(0)

        # Begin training
        for c in callbacks:
            c.on_train_begin()
        for epoch in range(epochs):
            for c in callbacks:
                c.on_epoch_begin(epoch)

            # Reset metrics
            self._sparse_categorical_accuracy.reset_states()
            self._total_train_loss.assign(0)
            self._total_robust_loss.assign(0)
            self._total_reg_loss.assign(0)

            _debug_list = []
            # Train on the mini-batches
            step = 0
            for batch in train_data:
                step += 1
                for c in callbacks:
                    c.on_train_batch_begin(step)

                w = None
                x, y = batch
                # Optimize the model

                self._update_schedules(self._current_step, schedule_assignment)
                self._current_step.assign_add(1)
                # Everything from here on will be traced
                train_loss, robust_loss, reg_loss = self.train_step(
                    x, y, w, attack_method
                )

                self._total_train_loss.assign_add(train_loss)
                self._total_robust_loss.assign_add(robust_loss)
                self._total_reg_loss.assign_add(reg_loss)
                logs = {}
                logs["accuracy"] = self._sparse_categorical_accuracy.result()
                logs["loss"] = self._total_train_loss / tf.cast(step, dtype=tf.float32)
                logs["robust_loss"] = self._total_robust_loss / tf.cast(
                    step, dtype=tf.float32
                )
                logs["reg_loss"] = self._total_reg_loss / tf.cast(
                    step, dtype=tf.float32
                )
                if step < 5:
                    _debug_list.append(robust_loss.numpy())

                for c in callbacks:
                    c.on_train_batch_end(step, logs)

                if (not steps_per_epoch is None) and step >= steps_per_epoch:
                    break
            if steps_per_epoch is None:
                steps_per_epoch = step
                progbar.set_params(
                    {
                        "epochs": epochs,
                        "steps": steps_per_epoch,
                        "verbose": 1,
                        "metrics": metrics_names,
                    }
                )

            if not valid_data is None:
                valid_metrics = self.evaluate(valid_data)
                for m in train_metrics_name:
                    logs["val_" + m] = valid_metrics[m]
            for c in callbacks:
                c.on_epoch_end(epoch, logs)
            if verbose_schedule:
                self._print_scheduled_variables(epoch, epochs)
            # print("Current schedules: ")
            # print("spec_loss_factor: ",str(self.spec_loss_factor))
            # print("robustness_loss_factor: ",str(self.robustness_loss_factor))
            # print("robustness_epsilon: ",str(self.robustness_epsilon))
            # print("Train Robustness debug list: ",str(_debug_list))
            # print("Train Robustness debug mask: ",str(_debug_mask))
        for c in callbacks:
            c.on_train_end()

    def evaluate_robustness(
        self, dataset, epsilon, total_samples=None, per_sample_timeout=20
    ):

        stats = {
            "processed": 0,
            "correct": 0,
            "robust": 0,
            "timeouts": 0,
            "pbar": None,
            "total_count": None,
        }

        if not total_samples is None:
            stats["pbar"] = tqdm(total=total_samples)
        else:
            stats["pbar"] = tqdm(total=1)

        for i in range(self.number_of_classes):
            stats = self._evaluate_class_robustness(
                dataset, epsilon, i, stats, per_sample_timeout
            )

        standard_accuracy = stats["correct"] / stats["processed"]
        robust_accuracy = stats["robust"] / stats["processed"]
        timeouts_percentage = stats["timeouts"] / stats["processed"]
        return standard_accuracy, robust_accuracy, timeouts_percentage

    def _evaluate_class_robustness(
        self, dataset, epsilon, class_id, stats, per_sample_timeout=20
    ):
        filtered_dataset = ibp_decision.filter_dataset_by_class(
            dataset, class_id
        ).batch(1)

        eps = tf.constant(epsilon, dtype=tf.float32)
        dp = self.get_decision_procedure(0, 255, [class_id])

        for batch in filtered_dataset:
            x, y = batch[0], batch[1]

            stats["processed"] += 1
            orig_prediction = tf.argmax(self.ff_model(x)[0])
            if orig_prediction == class_id:
                stats["correct"] += 1

                lb, ub = self.get_robustness_input(x, eps)
                # robust_bounds = self.robustness_bound_model((lb,ub))
                # robust_loss = self.robust_loss(tf.constant(class_id,dtype=tf.uint8),robust_bounds)
                # print("robust_bounds: {}".format(robust_bounds.numpy()))
                # print("robust_loss: {}".format(robust_loss.numpy()))

                # dp_bounds = dp.bound_domains(lb,ub)
                # print("dp_bounds: {}".format(str(dp_bounds.numpy())))
                # print("eps: ",str(eps))
                # print("lb: ",str(lb))
                # print("ub: ",str(ub))
                dp.reset(lb, ub)
                result = dp.check_sat(timeout=per_sample_timeout)
                # if (
                #     result != DecisionResult.SAT
                #     and result != DecisionResult.UNSAT
                #     and result != DecisionResult.TIMEOUT
                # ):
                #     raise ValueError(
                #         "An dp error occured during solve. ({})".format(result)
                #     )

                if result == ibp_decision.DecisionResult.UNSAT:
                    stats["robust"] += 1
                elif result == ibp_decision.DecisionResult.TIMEOUT:
                    stats["timeouts"] += 1
                    # print("Timeout!!!")
                    # print("Last bound: ",str(dp._latest_bound))
                    # print("Last attack score: ",str(dp._latest_attack_score))
                elif result == ibp_decision.DecisionResult.SAT:
                    attack = dp.get_conterexample(batch=True)
                    # print("SAT!!!")
                    # diff = np.max(tf.abs(x-attack))
                    # print("l-infty: {} pixels".format(str(255.0*diff)))
                    # pred = tf.argmax(self.model(attack)[0])
                    # print("Orig pred: {}, attack pred: {}, label: {}".format(str(orig_prediction),str(pred),str(y[0])))
                    # print("Last bound: ",str(dp._latest_bound))
                    # print("Last attack score: ",str(dp._latest_attack_score))
                elif result == ibp_decision.DecisionResult.NUMERICAL_ERROR:
                    print("Numerical error, counting example as non-robust")
                else:
                    raise ValueError("Unexpected return type '{}'".format(str(result)))

            stats["pbar"].set_description(
                "Class [{:d}/{:d}] acc: {:0.2f}%, robust: {:0.2f}% ({:0.2f}% timeouts)".format(
                    class_id + 1,
                    self.number_of_classes,
                    100 * stats["correct"] / stats["processed"],
                    100 * stats["robust"] / stats["processed"],
                    100 * stats["timeouts"] / stats["processed"],
                ),
                refresh=True,
            )
            if not stats["total_count"] is None:
                stats["pbar"].update(1)
            else:
                stats["pbar"].update()  # may trigger a refresh
        return stats