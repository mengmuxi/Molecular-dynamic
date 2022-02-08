# Copyright 2019 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A graph neural network based model to predict particle mobilities.

The architecture and performance of this model is described in our publication:
"Unveiling the predictive power of static structure in glassy systems".
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import functools

from graph_nets import graphs
from graph_nets import modules as gn_modules
from graph_nets import utils_tf

import sonnet as snt
import tensorflow as tf
from typing import Any, Dict, Text, Tuple, Optional


def make_graph_from_static_structure(
    positions,
    types,
    box,
    edge_threshold):
  """Returns graph representing the static structure of the glass.

  Each particle is represented by a node in the graph. The particle type is
  stored as a node feature.
  Two particles at a distance less than the threshold are connected by an edge.
  The relative distance vector is stored as an edge feature.

  Args:   
    positions: particle positions with shape [n_particles, 3].粒子位置
    types: particle types with shape [n_particles].粒子类型
    box: dimensions of the cubic box that contains the particles with shape [3].盒子尺寸
    edge_threshold: particles at distance less than threshold are connected by
      an edge.距离小于阈值的粒子由边连接
  """
  # Calculate pairwise relative distances between particles: shape [n, n, 3].计算粒子之间的距离
  cross_positions = positions[tf.newaxis, :, :] - positions[:, tf.newaxis, :]
  # Enforces periodic boundary conditions.强制执行周期性边界条件
  box_ = box[tf.newaxis, tf.newaxis, :]
  cross_positions += tf.cast(cross_positions < -box_ / 2., tf.float32) * box_#tf.cast(x,dtype)将x（如果是Tensor）强制转换为dtype。
  cross_positions -= tf.cast(cross_positions > box_ / 2., tf.float32) * box_
  # Calculates adjacency matrix in a sparse format (indices), based on the given
  # distances and threshold.根据给定的距离和阈值，以稀疏格式（索引）计算邻接矩阵
  distances = tf.norm(cross_positions, axis=-1)#计算范数
  indices = tf.where(distances < edge_threshold)#返回值，是condition中元素为True对应的索引
 
  # Defines graph.
  nodes = types[:, tf.newaxis]
  senders = indices[:, 0]
  receivers = indices[:, 1]
  edges = tf.gather_nd(cross_positions, indices)
  va=[[0.03]]
  return graphs.GraphsTuple(
      nodes=tf.cast(nodes, tf.float32),
      n_node=tf.reshape(tf.shape(nodes)[0], [1]),
      edges=tf.cast(edges, tf.float32),
      n_edge=tf.reshape(tf.shape(edges)[0], [1]),
      globals=tf.cast(va, dtype=tf.float32),

      receivers=tf.cast(receivers, tf.int32),
      senders=tf.cast(senders, tf.int32)
      )


def apply_random_rotation(graph):
  """Returns randomly rotated graph representation.

  The rotation is an element of O(3) with rotation angles multiple of pi/2.
  This function assumes that the relative particle distances are stored in
  the edge features.

  Args:
    graph: The graphs tuple as defined in `graph_nets.graphs`.
  """
  # Transposes edge features, so that the axes are in the first dimension.
  # Outputs a tensor of shape [3, n_particles].
  xyz = tf.transpose(graph.edges)
  # Random pi/2 rotation(s)
  permutation = tf.random.shuffle(tf.constant([0, 1], dtype=tf.int32))
  xyz = tf.gather(xyz, permutation)
  # Random reflections.
  symmetry = tf.random_uniform([2], minval=0, maxval=2, dtype=tf.int32)
  symmetry = 1 - 2 * tf.cast(tf.reshape(symmetry, [2, 1]), tf.float32)
  xyz = xyz * symmetry
  edges = tf.transpose(xyz)
  return graph.replace(edges=edges)


class GraphBasedModel(snt.AbstractModule):
  """Graph based model which predicts particle mobilities from their positions.

  This network encodes the nodes and edges of the input graph independently, and
  then performs message-passing on this graph, updating its edges based on their
  associated nodes, then updating the nodes based on the input nodes' features
  and their associated updated edge features.
  This update is repeated several times.
  Afterwards the resulting node embeddings are decoded to predict the particle
  mobility.
  """

  def __init__(self,
               n_recurrences,
               mlp_sizes,
               mlp_kwargs = None,
               name='Graph'):
    """Creates a new GraphBasedModel object.

    Args:
      n_recurrences: the number of message passing steps in the graph network.图网络中消息传递步骤的数量
      mlp_sizes: the number of neurons in each layer of the MLP.MLP每一层神经元数量
      mlp_kwargs: additional keyword aguments passed to the MLP.传递给 MLP 的附加关键字参数
      name: the name of the Sonnet module.
    """
    super(GraphBasedModel, self).__init__(name=name)
    self._n_recurrences = n_recurrences

    if mlp_kwargs is None:
      mlp_kwargs = {}

    model_fn = functools.partial(
        snt.nets.MLP,
        output_sizes=mlp_sizes,
        activate_final=True,
        **mlp_kwargs)

    final_model_fn = functools.partial(
        snt.nets.MLP,
        output_sizes=mlp_sizes+(1,),
        activate_final=False,
        **mlp_kwargs)

    with self._enter_variable_scope():
      self._encoder = gn_modules.GraphIndependent(
          node_model_fn=model_fn,
          edge_model_fn=model_fn,
          global_model_fn=model_fn)

      if self._n_recurrences > 0:
        self._propagation_network = gn_modules.GraphNetwork(
            node_model_fn=model_fn,
            edge_model_fn=model_fn,
            # We do not use globals, hence we just pass the identity function.
            global_model_fn=model_fn,
            reducer=tf.unsorted_segment_sum)
      self._decoder = gn_modules.GraphIndependent(
          node_model_fn=model_fn,
          edge_model_fn=model_fn,
          global_model_fn=final_model_fn)

  def _build(self, graphs_tuple):
    """Connects the model into the tensorflow graph.

    Args:
      graphs_tuple: input graph tensor as defined in `graphs_tuple.graphs`.

    Returns:
      tensor with shape [n_particles] containing the predicted particle
      mobilities.
    """
    encoded = self._encoder(graphs_tuple)
    outputs = encoded

    for _ in range(self._n_recurrences):
      # Adds skip connections.
      inputs = utils_tf.concat([outputs, encoded], axis=-1)
      outputs = self._propagation_network(inputs)

    decoded = self._decoder(outputs)
    return tf.squeeze(decoded.globals, axis=-1)
