from src import *

FILES = [
    ["08-OUT-ALL", 1],
    ["09-IN-ALL", 1],
    ["11-IN-ALL", 1],
    ["05-IN-ALL", 1],
    ["06-IN-ALL", 1],
]

UUID = "042_large_dataset_1000/0"

for file in FILES:
    model, config = load_model_locally(UUID, model_name="rvce", device="cuda:1")
    video = Video(file, config)

    from_time, till_time = video.get_from_till_time(Part.WHOLE)

    validate_video(
        video,
        model,
        from_time=from_time,
        till_time=till_time,
        return_probs=True,
        tqdm=tqdm,
        verbose=True,
    )
