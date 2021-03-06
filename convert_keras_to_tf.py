import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.tools.graph_transforms import TransformGraph
from pathlib import Path
from absl import app
from absl import flags
from absl import logging
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json, model_from_yaml
from yolo import YOLO

K.set_learning_phase(0)
FLAGS = flags.FLAGS

flags.DEFINE_string('input_model', None, 'Path to the input model.')
flags.DEFINE_string('output_model', None, 'Path where the converted model will '
                                          'be stored.')
flags.DEFINE_boolean('save_graph_def', False,
                     'Whether to save the graphdef.pbtxt file which contains '
                     'the graph definition in ASCII format.')
flags.DEFINE_boolean('quantize', False,
                     'If set, the resultant TensorFlow graph weights will be '
                     'converted from float into eight-bit equivalents. See '
                     'documentation here: '
                     'https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms')
flags.DEFINE_boolean('output_meta_ckpt', False,
                     'If set to True, exports the model as .meta, .index, and '
                     '.data files, with a checkpoint file. These can be later '
                     'loaded in TensorFlow to continue training.')

flags.mark_flag_as_required('input_model')
flags.mark_flag_as_required('output_model')

def main(args):
    # If output_model path is relative and in cwd, make it absolute from root
    output_model = FLAGS.output_model
    if str(Path(output_model).parent) == '.':
        output_model = str((Path.cwd() / output_model))

    output_fld = Path(output_model).parent
    output_model_name = Path(output_model).name
    output_model_stem = Path(output_model).stem
    output_model_pbtxt_name = output_model_stem + '.pbtxt'

    # Create output directory if it does not exist
    Path(output_model).parent.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(model_path=FLAGS.input_model)
    sess = yolo.sess
    image_data = yolo.yolo_model.input
    boxes = tf.identity(yolo.boxes, name="boxes")
    scores = tf.identity(yolo.scores, name="scores")
    classes = tf.identity(yolo.classes, name="classes")
    converted_output_node_names = ["boxes", "scores", "classes"]
    logging.info("Input image data %s", image_data)
    logging.info("Input image shape: %s", yolo.input_image_shape)
    logging.info('Converted output node names are: %s', str(converted_output_node_names))

    if FLAGS.output_meta_ckpt:
        saver = tf.train.Saver()
        saver.save(sess, str(output_fld / output_model_stem))

    if FLAGS.save_graph_def:
        tf.train.write_graph(sess.graph.as_graph_def(), str(output_fld),
                             output_model_pbtxt_name, as_text=True)
        logging.info('Saved the graph definition in ascii format at %s',
                     str(Path(output_fld) / output_model_pbtxt_name))

    transforms = ['remove_nodes(op=Identity)', \
         'merge_duplicate_nodes', \
         'strip_unused_nodes',
         'fold_constants(ignore_errors=true)',
         'fold_batch_norms']
    if FLAGS.quantize:
        transforms += [
             'quantize_weights',
             'quantize_node']
    transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [],
                                           converted_output_node_names,
                                           transforms)
    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        transformed_graph_def,
        converted_output_node_names)
    graph_io.write_graph(constant_graph, str(output_fld), output_model_name,
                         as_text=False)
    logging.info('Saved the freezed graph at %s',
                 str(Path(output_fld) / output_model_name))


if __name__ == "__main__":
    app.run(main)
