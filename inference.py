from src import *
from cross_validation import generate_cross_validation_table


def inference(root_uuid, prefix, model_name, inference_function, coupled_labels, 
              use_manual_counts, nn_hop_length, n_windows_for_dense_inference, n_events_per_dense_window):
    uuids = []
    for i in range(0, 5):
        uuid = f'{root_uuid}/{i}'
        print(uuid)
        uuids.append(uuid)
        config = load_config_locally(uuid)
        config.inference_function = inference_function
        config.coupled_labels = coupled_labels
        config.use_manual_counts = use_manual_counts
        config.n_windows_for_dense_inference = n_windows_for_dense_inference
        config.n_events_per_dense_window = n_events_per_dense_window
        if nn_hop_length is not None:
            config.nn_hop_length = nn_hop_length
            config.n_samples_in_nn_hop = int(nn_hop_length * config.sr)
        tst_datapool = DataPool(config.testing_files, config)
        validate_and_save(uuid, tst_datapool, prefix, Part.WHOLE, model_name, config=config)

    generate_cross_validation_table(uuids, model_name=model_name, prefix=prefix)


root_uuid = '035_RX100_resized_128_audio_image_augmentation_bs_512'
prefix = 'tst'

doubled = False
dense = True

use_manual_counts = False
coupled_labels = None
nn_hop_length = None
n_events_per_dense_window = None

if doubled:
    prefix += '_doubled'
    inference_function = InferenceFunction.DOUBLED
    n_windows_for_dense_inference = 2
    n_events_per_dense_window = 6
elif dense:
    prefix += '_dense'
    inference_function = InferenceFunction.DENSE
    n_windows_for_dense_inference = 6
    n_events_per_dense_window = 2
else:
    prefix += '_structured'
    coupled_labels = [['n_incoming', 'n_outgoing'], ['n_CAR', 'n_NOT_CAR']]
    # coupled_labels = [['n_CAR', 'n_NOT_CAR']]
    # coupled_labels = [['n_incoming', 'n_outgoing']]
    inference_function = InferenceFunction.STRUCTURED

if use_manual_counts:
    prefix += '_manual'

model_name = 'rvce'

inference(
    root_uuid,
    prefix,
    model_name,
    inference_function,
    coupled_labels,
    use_manual_counts,
    nn_hop_length, 
    n_windows_for_dense_inference,
    n_events_per_dense_window)