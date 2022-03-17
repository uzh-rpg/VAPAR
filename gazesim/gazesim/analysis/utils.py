import pandas as pd

from tensorboard.backend.event_processing import event_accumulator


def extract_tensorboard_data(data_dir):
    # taken from: https://gist.github.com/willwhitney/9cecd56324183ef93c2424c9aa7a31b4#file-load_tf-py-L12
    accumulator = event_accumulator.EventAccumulator(data_dir, size_guidance={event_accumulator.SCALARS: 0})
    accumulator.Reload()
    data_frames = {}
    scalar_names = accumulator.Tags()["scalars"]

    for n in scalar_names:
        data_frames[n] = pd.DataFrame(accumulator.Scalars(n), columns=["wall_time", "step", n])
        data_frames[n].drop("wall_time", axis=1, inplace=True)
        data_frames[n] = data_frames[n].set_index("step")

    return pd.concat([v for k, v in data_frames.items()], axis=1)

