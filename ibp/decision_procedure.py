import tensorflow as tf
import numpy as np
from enum import Enum
import time
from tqdm import tqdm
import ibp.utils as ibp_utils
import ibp.building as ibp_build
import ibp.depr_layers as ibp_layers


def try_convert_to_tensor(value):
    if not type(value) is tf.Tensor:
        value = tf.convert_to_tensor(value, dtype=tf.float32)
    return value


def convert_to_flat_tensor(value):
    value = try_convert_to_tensor(value)
    if len(value.shape) > 1:
        value = tf.reshape(value, [value.shape[1]])
    return value


class DecisionResult(Enum):
    SAT = 1
    UNSAT = 2
    UNKNOWN = 3
    TIMEOUT = 4
    ERROR = 5
    NUMERICAL_ERROR = 6


ABD_DEFAULT_CONFIG = {
    "split_heuristic": "longest_axis",  # "abs_gradient" | "longest axis" | "hybrid_grad_length"
    "attack_iters": 20,
    "attack_lr": 0.1,
    "attack_sign": True,
    "attack_lr_relative_of_domain": True,
    "numerical_eps": 0.0,
}


def filter_dataset_by_class(dataset, class_id):
    # Note: tf.reshape(y,[]) converts Tensor of shape (1,) to scalar tensor
    if len(dataset.element_spec) == 3:
        # Samples of this dataset are weighted
        return dataset.filter(
            lambda x, y, w: tf.math.equal(tf.reshape(y, []), class_id)
        )
    else:
        # Just features and labels
        return dataset.filter(lambda x, y: tf.math.equal(tf.reshape(y, []), class_id))


class DecisionProcedure:
    def __init__(
        self, ff_model, pre_logit_bound_model, output_layer, lb, ub, allowed_classes
    ):
        """
        @allowed_classes: List of allowed output classes
        """
        self.allowed_classes = allowed_classes
        self.ff_model = ff_model
        self.bound_model = ibp_build.add_spec_list_to_bound_model(
            pre_logit_bound_model,
            output_layer,
            allowed_classes,
            reduce_max=True,
            bound_margin=0.0,
        )
        # self.bound_model.summary()
        # self._pure_bound_model = ibp_build.build_pure_bound_model(
        #     pre_logit_bound_model, output_layer
        # )
        self.input_size = int(ff_model._input_shape[-1])
        self.output_size = int(ff_model.output_dim)

        self.initial_domains = [
            (convert_to_flat_tensor(lb), convert_to_flat_tensor(ub))
        ]
        self._config = ABD_DEFAULT_CONFIG
        self._create_attack_canvas()
        self._debug_doms_list = []

    def _create_attack_canvas(self):
        batch_size = int(self._num_forbidden_classes)
        # print("Batch size: ",str(batch_size))
        # print("lb: ",str(lb))
        # print("ub: ",str(ub))
        self._canvas = tf.Variable(
            tf.zeros([batch_size, self.input_size], dtype=tf.float32), trainable=False
        )

    def append_domain(self, lb, ub):
        self.initial_domains.append(
            (convert_to_flat_tensor(lb), convert_to_flat_tensor(ub))
        )

    def reset(self, lb, ub):
        self.initial_domains = [
            (convert_to_flat_tensor(lb), convert_to_flat_tensor(ub))
        ]

    def _reset_solver(self):
        # Copy list
        self.domains = list(self.initial_domains)
        self.sat_witness = None
        self._latest_attack_score = -1
        self._latest_bound = -1
        self._pruned_domains = 0
        self._processed_domains = 0

    def _run_step(self):
        if len(self.domains) == 0:
            return DecisionResult.UNSAT

        self._processed_domains += 1
        lb, ub = self._pop_domain()
        # Atttack (traced)
        counterexample, attack_score = self.attack_domain(lb, ub)
        self._latest_attack_score = float(attack_score.numpy())
        # print("Attack score: ",str(attack_score))
        # pred = self.model.predict(np.expand_dims(counterexample,axis=0))[0]
        # print("Attack class: {}".format(np.argmax(pred)))

        # Bound (traced)
        bounds = self.bound_domains(lb, ub)
        self._latest_bound = float(bounds.numpy())
        # print("bounds: ",str(bounds))

        if attack_score < 0:
            # pred = self.model.predict(np.expand_dims(counterexample,axis=0))[0]
            # print("Attack class: {} (label {})".format(np.argmax(pred),self.allowed_classes[0]))
            # print("Attack score : {} (less than 0 is attack)".format(str(self._latest_attack_score)))
            # print("Defense bound: {} (> 0 is safe)".format(str(self._latest_bound)))
            # print("SAT")
            self.sat_witness = counterexample
            return DecisionResult.SAT

        # pure_bounds = self.get_pure_bounds(lb,ub)
        # print("output layer lower bound:\n",str(pure_bounds[0]))
        # print("output layer upper bound:\n",str(pure_bounds[1]))
        if bounds < 0:
            # raise ValueError("DEBUG: This should not happend with eps=0")
            # Divide domain (traced)
            (lb_1, ub_1), (lb_2, ub_2), min_dim = self._split_domain(lb, ub)
            min_dim = float(min_dim)
            if min_dim < self._config["numerical_eps"]:
                print(
                    "Warning: Numerical precision limit reached: {}, latest attack score: {:}, defense bound {}".format(
                        str(min_dim),
                        str(self._latest_attack_score),
                        str(self._latest_bound),
                    )
                )
                # print("lb: ", str(lb))
                # print("lb: ", str(ub))
                # print("lb_1: ", str(lb_1))
                # print("ub_1: ", str(ub_1))
                # print("lb_2: ", str(lb_2))
                # print("ub_2: ", str(ub_2))
                # print("min_dim: ", str(min_dim))
                return DecisionResult.NUMERICAL_ERROR
            # self._debug_doms_list.append((lb_1.numpy(),ub_1.numpy()))
            # self._debug_doms_list.append((lb_2.numpy(),ub_2.numpy()))
            # print("Spliting domain lb: {}, ub: {}".format(str(lb.numpy()),str(ub.numpy())))
            # print("   Bound : {:0.5f}".format(self._latest_bound))
            # print("   Attack: {:0.5f}".format(self._latest_attack_score))
            # print("   Min dimension: {}".format(str(min_dim)))
            # print("   new domain 1: lb: "+str(lb_1.numpy())+", ub: "+str(ub_1.numpy()))
            # print("   new domain 2: lb: "+str(lb_2.numpy())+", ub: "+str(ub_2.numpy()))
            # print("#######################################")
            self.domains.append((lb_1, ub_1))
            self.domains.append((lb_2, ub_2))
        else:
            # Domain safe ->
            self._pruned_domains += 1
            if len(self.domains) == 0:
                return DecisionResult.UNSAT

        return DecisionResult.UNKNOWN

    def get_conterexample(self, batch=None):
        if batch:
            return tf.reshape(self.sat_witness, shape=[1, -1])
        return self.sat_witness

    def _pop_domain(self):
        return self.domains.pop(0)

    @tf.function
    def _split_domain(self, lb, ub):
        largest_dim = self._select_index_to_split(lb, ub)
        one_hot = tf.one_hot(largest_dim, self.input_size)
        center = 0.5 * (ub + lb)

        # Split [lb,ub] into [lb,center] and [center,ub]
        # upper bounds of domain 1 is the middle
        ub_1 = ub * (1 - one_hot) + center * one_hot
        # lower bounds of domain 2 is the middle
        lb_2 = lb * (1 - one_hot) + center * one_hot

        lb_1 = lb
        ub_2 = ub

        min_dim = tf.minimum(tf.reduce_min(ub_1 - lb_1), tf.reduce_min(ub_2 - lb_2))

        return (lb_1, ub_1), (lb_2, ub_2), min_dim

    @tf.function
    def _select_index_to_split(self, lb, ub):
        if self._config["split_heuristic"] == "longest_axis":
            return tf.argmax(ub - lb)
        elif self._config["split_heuristic"] == "hybrid_grad_length":
            # Create batch of size 1
            axis_len = ub - lb

            lb = tf.expand_dims(lb, axis=0)
            ub = tf.expand_dims(ub, axis=0)
            abs_grad = self.get_bound_to_input_gradient(lb, ub)[0]

            axis_len /= tf.reduce_max(axis_len)
            abs_grad /= tf.reduce_max(abs_grad)
            score = axis_len * abs_grad
            index = tf.argmax(score)
            # print("inex: ",str(index))
            return index
        elif self._config["split_heuristic"] == "abs_gradient":
            lb = tf.expand_dims(lb, axis=0)
            ub = tf.expand_dims(ub, axis=0)
            abs_grad = self.get_bound_to_input_gradient(lb, ub)[0]
            abs_grad /= tf.reduce_max(abs_grad)
            score = abs_grad
            index = tf.argmax(score)
            return index
        else:
            raise ValueError("Unknown split heuristic")

    def check_sat(self, timeout=None, verbose_interval=None):
        self._reset_solver()

        current_result = DecisionResult.UNKNOWN
        start_time = time.time()
        last_logged = time.time()
        current_result = self._run_step()  # Run at least one step even if timeout=0
        while current_result == DecisionResult.UNKNOWN:
            if (not timeout is None) and time.time() - start_time > timeout:
                return DecisionResult.TIMEOUT
            current_result = self._run_step()

            if (
                not verbose_interval is None
            ) and time.time() - last_logged > verbose_interval:
                last_logged = time.time()
                percent_done = 100.0 - self._remaining_search_space()
                print(
                    "Runtime {}, percent done: {:0.2f}%,\n   - Domains: {} open, {} pruned\n   - Bounds: [defend: {:0.4f}, attack: {:0.4f}]".format(
                        ibp_utils.second_to_fancy_str(time.time() - start_time),
                        percent_done,
                        len(self.domains),
                        self._pruned_domains,
                        self._latest_bound,
                        self._latest_attack_score,
                    )
                )
        # print("Done. Runtime: {}, {:d} domains processed ({:d} pruned)".format(second_to_fancy_str(time.time()-start_time),self._processed_domains,self._pruned_domains))
        return current_result

    def _remaining_search_space(self):

        remaining = 0
        divisor = 1e-8
        for lb, ub in self.initial_domains:
            divisor += np.prod(ub - lb)
        for lb, ub in self.domains:
            remaining += np.prod((ub - lb + 1e-8) / divisor)

        remaining_percent = 100 * remaining
        return remaining_percent

    """ Create an additional last layer that implements targeted attacks """

    @tf.function
    def _add_attack_layer(self, logits):
        allowed = self.allowed_classes
        forbidden = [i for i in range(self.output_size) if not i in allowed]

        allowed_mask = np.zeros([self.output_size], dtype=np.float32)
        for yes in allowed:
            allowed_mask[yes] = 1
        # Element-wise multiplication with allowed_mask
        masked_logits = logits * tf.constant(allowed_mask)

        # HACK: make sure 0 created through the masking is not the maximum
        masked_logits -= (1 - allowed_mask) * tf.reduce_sum(tf.abs(logits))
        top_allowed = tf.reduce_max(masked_logits, axis=1)

        # Each forbidden class has the potential to be greater than all allowed classes
        attack_matrices = []
        for no in forbidden:
            M = np.zeros([self.output_size], dtype=np.float32)
            M[no] = 1
            attack_matrices.append(M)

        M = np.stack(attack_matrices, axis=0)
        assert M.shape[0] == len(forbidden)
        # Targeted attacks
        attack_heads = tf.reduce_sum(logits * tf.constant(M), axis=1)

        # Top allowed - attack candidate
        attack_score = top_allowed - attack_heads
        return attack_score

    @tf.function
    def attack_domain(self, lb, ub):
        batch_size = int(self._num_forbidden_classes)
        # print("Batch size: ",str(batch_size))
        # print("lb: ",str(lb))
        # print("ub: ",str(ub))
        # Clear canvas
        self._canvas.assign(tf.random.uniform([batch_size, self.input_size], lb, ub))

        attacks, scores = self.attack_inplace(self._canvas, lb, ub)
        best_i = tf.argmin(scores)
        return attacks[best_i], scores[best_i]

    @tf.function
    def get_bound_to_input_gradient(self, lb, ub):
        with tf.GradientTape() as tape_ub:
            with tf.GradientTape() as tape_lb:
                tape_ub.watch(ub)
                tape_lb.watch(lb)
                bound = self.bound_model([lb, ub])

        # Get the gradients of the loss w.r.t to the input image.
        gradient_lb = tf.abs(tape_lb.gradient(bound, lb))
        gradient_ub = tf.abs(tape_ub.gradient(bound, ub))
        mean_gradient = 0.5 * (gradient_lb + gradient_ub)
        return mean_gradient

    @property
    def _num_forbidden_classes(self):
        return self.output_size - len(self.allowed_classes)

    @tf.function
    def bound_domains(self, lb, ub):
        lb = tf.expand_dims(lb, axis=0)
        ub = tf.expand_dims(ub, axis=0)
        return self.bound_model([lb, ub])

    # @tf.function
    # def get_pure_bounds(self, lb, ub):
    #     lb = tf.expand_dims(lb, axis=0)
    #     ub = tf.expand_dims(ub, axis=0)
    #     return self._pure_bound_model([lb, ub])

    @tf.function
    def get_attack_score(self, x):
        if len(x.shape) == 1:
            # Create batch of size 1
            x = tf.expand_dims(x, axis=0)
        attacked_prediction = self.ff_model(x)
        attack_scores = self._add_attack_layer(attacked_prediction)
        return attack_scores

    # Evaluates the network at random locations (=weak attack)
    # @tf.function
    # def attack_inplace(self,x,lower_bound,upper_bound):
    #     attacked_prediction = self.attack_model(x)*self.attack_matrices
    #     attack_scores = tf.reduce_sum(attacked_prediction,axis=1)

    #     return x,attack_scores

    # Runs IFGSM attack to produce an attack (=strong attack)
    @tf.function
    def attack_inplace(self, x, lower_bound, upper_bound):
        for i in range(self._config["attack_iters"]):
            with tf.GradientTape() as tape:
                tape.watch(x)
                prediction = self.ff_model(x)
                attack_score = self._add_attack_layer(prediction)

            # Get the gradients of the loss w.r.t to the input image.
            gradient = tape.gradient(attack_score, x)
            if self._config["attack_sign"]:
                gradient = tf.sign(gradient)
            if self._config["attack_lr_relative_of_domain"]:
                gradient = gradient * (upper_bound - lower_bound)
            gradient_step = -self._config["attack_lr"] * gradient
            x.assign_add(gradient_step)  # We want to descent the loss
            new_x = tf.clip_by_value(x, lower_bound, upper_bound)
            x.assign(new_x)
        attacked_prediction = self.ff_model(x)
        attack_scores = self._add_attack_layer(attacked_prediction)

        return x, attack_scores