from typing import List
from src import *
import subprocess


def append_summary(dict):
    for k, v in dict.items():
        if k == "file":
            dict[k].append("")
        else:
            v = np.array(v).astype(float)
            stats = f"{v.mean():.3f} ± {v.std():.3f}"
            dict[k].append(stats)


def generate_summary_table(files, prefix="tst"):
    root_uuid = files[0].split("/")[1]
    table = []
    header = []

    for file in files:
        results = np.genfromtxt(
            file,
            delimiter=",",
            skip_footer=1,
            dtype=str,
        )
        header = results[0]
        results = results[1:]
        results = np.atleast_2d(results)
        table.extend(results)
    table = np.array(table).T

    dict = {}
    for i in range(len(header)):
        column_name = header[i]
        column = table[i].tolist()
        dict[column_name] = column

    append_summary(dict)

    save_dict_txt(f"outputs/{root_uuid}/{prefix}_structured_predictor*.txt", dict)
    save_dict_csv(f"outputs/{root_uuid}/{prefix}_structured_predictor*.csv", dict)


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def setup_globals(_config):
    global config
    config = Config(_config)


def train_models_parallel(config) -> None:
    args = " ".join(sys.argv[1:])

    for head in aslist(config.structured_predictor_heads):
        system_calls = []
        for split in aslist(config.structured_predictor_splits):
            for reg in aslist(config.structured_predictor_regs):
                call = (
                    f"python3 train_structured_predictor.py {args} "
                    f"structured_predictor_splits={split} "
                    f"structured_predictor.reg={reg} "
                    f"structured_predictor.head={head} "
                    f"structured_predictor.combine_trn_and_val=False "
                )
                system_calls.append(call)

    system_calls = " & ".join(system_calls)

    # run processes in parallel
    subprocess.call(system_calls, shell=True)


def train_final_models_parallel(head_split_reg) -> None:
    args = " ".join(sys.argv[1:])

    system_calls.append(call)

    for head, split, reg in head_split_reg:
        call = (
            f"python3 train_structured_predictor.py {args} "
            f"structured_predictor_splits={split} "
            f"structured_predictor.reg={reg} "
            f"structured_predictor.head={head} "
            f"structured_predictor.combine_trn_and_val=True "
        )
        system_calls.append(call)

    system_calls = " & ".join(system_calls)

    # run processes in parallel
    subprocess.call(system_calls, shell=True)


def find_best_regularization_constant() -> List:
    head_split_reg = []
    for head in aslist(config.structured_predictor_heads):
        for split in aslist(config.structured_predictor_splits):
            best_reg = None
            best_rvce = np.inf
            for reg in aslist(config.structured_predictor_regs):
                file = f"outputs/{config.uuid}/{split}/results_structured_predictor/val_{head}_{reg}_structured_predictor.csv"
                rvce = np.loadtxt(
                    file,
                    delimiter=",",
                    usecols=0,
                    dtype=str,
                )[-1]
                rvce = float(rvce.split(" ± ")[0])

                if rvce < best_rvce:
                    best_rvce = rvce
                    best_reg = reg

            head_split_reg.append([head, split, best_reg])
            print(
                f"for head: {head} and split: {split}, best rvce: {best_rvce} with reg: {best_reg}"
            )
    return head_split_reg


def generate_results(head_split_reg) -> None:
    files_trn = []
    files_val = []
    files_tst = []

    for head, split, reg in head_split_reg:
        file_trn = f"outputs/{config.uuid}/{split}/results_structured_predictor/trn_{head}_{reg}_structured_predictor.csv"
        file_val = f"outputs/{config.uuid}/{split}/results_structured_predictor/val_{head}_{reg}_structured_predictor.csv"
        file_tst = f"outputs/{config.uuid}/{split}/results_structured_predictor/tst_{head}_{reg}_structured_predictor.csv"

        files_trn.append(file_trn)
        files_val.append(file_val)
        files_tst.append(file_tst)

    generate_summary_table(files_trn, "trn")
    generate_summary_table(files_val, "val")
    generate_summary_table(files_tst, "tst")


if __name__ == "__main__":
    setup_globals()

    # train_models_parallel(config)

    head_split_reg = find_best_regularization_constant()

    # train_final_models_parallel()

    generate_results(head_split_reg)
