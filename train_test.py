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

"""Tests for train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow.compat.v1 as tf

import train


class TensorflowTrainTest(tf.test.TestCase):

  def test_apply_model(self):
    """Tests if we can apply a model to a small test dataset."""
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'ETA0w0.4',
                                   't0f50.ckpt')#‘Eta0.1’文件夹下 跟训练模型的文件名一样，
    file_pattern = os.path.join(os.path.dirname(__file__), 'pickle_N1000_L5_ETA0w0.4/test',
                                'Con100eta0N1000L5w0.4_1000')#存放数据集，要改名η
    predictions = train.apply_model(checkpoint_path=checkpoint_path,
                                    file_pattern=file_pattern,
                                    time_index=0)#改
    data = train.load_data(file_pattern, 0)#改
    targets = data[0].targets
    # correlation_value = np.corrcoef(predictions[0], targets)[0, 1]
    print('predictions:',predictions)
    print('targets:',targets) 
    # self.assertGreater(correlation_value, 0.5)


if __name__ == '__main__':
  tf.test.main()
