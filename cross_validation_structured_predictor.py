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


def generate_summary_table(files, prefix="tst", is_final=False):
    root_uuid = files[0].split("/")[1]
    table = []
    header = []

    for file in files:
        results = pd.read_csv(file, skipfooter=1)
        header = results.columns
        results = results.values
        results = np.atleast_2d(results)
        table.extend(results)
    table = np.array(table).T

    dict = {}
    for i in range(len(header)):
        column_name = header[i]
        column = table[i].tolist()
        dict[column_name] = column

    append_summary(dict)

    suffix = "*" if is_final else ""

    save_dict_txt(
        f"outputs/{root_uuid}/results/{prefix}_structured_predictor{suffix}.txt", dict
    )
    save_dict_csv(
        f"outputs/{root_uuid}/results/{prefix}_structured_predictor{suffix}.csv", dict
    )


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def setup_globals(_config):
    global config
    config = Config(_config)


def train_models_parallel(config) -> None:
    args = " ".join(sys.argv[1:])

    programs = []

    for head in aslist(config.structured_predictor_heads):
        for split in aslist(config.structured_predictor_splits):
            for reg in aslist(config.structured_predictor_regs):
                program = (
                    f"python train_structured_predictor.py {args} "
                    f"structured_predictor_splits={split} "
                    f"structured_predictor.reg={reg} "
                    f"structured_predictor.head={head} "
                    f"structured_predictor.combine_trn_and_val=False "
                )
                instance = subprocess.Popen(program, shell=True)
                programs.append(instance)

    # run programs in parallel
    for p in programs:
        p.wait()


def train_final_models_parallel(head_split_reg) -> None:
    args = " ".join(sys.argv[1:])

    programs = []

    for head, (splits_regs) in head_split_reg.items():
        for split, reg in splits_regs:
            program = (
                f"python train_structured_predictor.py {args} "
                f"structured_predictor_splits={split} "
                f"structured_predictor.reg={reg} "
                f"structured_predictor.head={head} "
                f"structured_predictor.combine_trn_and_val=True "
            )
            instance = subprocess.Popen(program, shell=True)
            programs.append(instance)

    # run programs in parallel
    for p in programs:
        p.wait()


def find_best_regularization_constant(config) -> List:
    head_split_reg = defaultdict(list)
    for head in aslist(config.structured_predictor_heads):
        for split in aslist(config.structured_predictor_splits):
            best_reg = None
            best_rvce = np.inf
            for reg in aslist(config.structured_predictor_regs):
                file = f"outputs/{config.uuid}/{split}/results_structured_predictor/val_{head}_{reg}_structured_predictor.csv"
                rvce = pd.read_csv(file)["rvce"].tail(1).item().split(" ")[0]
                rvce = float(rvce)
                if rvce < best_rvce:
                    best_rvce = rvce
                    best_reg = reg

            head_split_reg[head].append([split, best_reg])
            print(
                f"for head: {head} and split: {split}, best val rvce: {best_rvce} with reg: {best_reg}"
            )
    return head_split_reg


def generate_results(head_split_reg, is_final=False) -> None:
    subfolder = "/*" if is_final else ""

    for head, (splits_regs) in head_split_reg.items():
        files_trn = []
        files_val = []
        files_tst = []

        for split, reg in splits_regs:
            path = (
                f"outputs/{config.uuid}/{split}/results_structured_predictor{subfolder}"
            )

            files_trn.append(f"{path}/trn_{head}_{reg}_structured_predictor.csv")
            files_val.append(f"{path}/val_{head}_{reg}_structured_predictor.csv")
            files_tst.append(f"{path}/tst_{head}_{reg}_structured_predictor.csv")

        generate_summary_table(files_trn, f"trn/{head}", is_final)
        generate_summary_table(files_val, f"val/{head}", is_final)
        generate_summary_table(files_tst, f"tst/{head}", is_final)


if __name__ == "__main__":
    setup_globals()

    train_models_parallel(config)

    head_split_reg = find_best_regularization_constant(config)

    generate_results(head_split_reg, is_final=False)

    train_final_models_parallel(head_split_reg)

    generate_results(head_split_reg, is_final=True)
