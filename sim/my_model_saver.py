import os

import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.ops import variables
from tensorflow.python.lib.io import file_io
from tensorflow.python.util import compat
from tensorflow.python.saved_model import constants


class MyModelSaver(tf.saved_model.builder.SavedModelBuilder):
    def update(self, sess):
        # Create the variables sub-directory, if it does not exist.
        variables_dir = os.path.join(
            compat.as_text(self._export_dir),
            compat.as_text(constants.VARIABLES_DIRECTORY))
        if not file_io.file_exists(variables_dir):
            file_io.recursive_create_dir(variables_dir)

        variables_path = os.path.join(
            compat.as_text(variables_dir),
            compat.as_text(constants.VARIABLES_FILENAME))

        saver = tf_saver.Saver(
            variables._all_saveable_objects(),  # pylint: disable=protected-access
            sharded=True,
            write_version=saver_pb2.SaverDef.V2,
            allow_empty=True)

        saver.save(sess, variables_path, write_meta_graph=False, write_state=True)