#!/bin/sh

python3 create_dataset_for_structured_learning.py --split_number=0 --part=trn &
python3 create_dataset_for_structured_learning.py --split_number=1 --part=trn &
python3 create_dataset_for_structured_learning.py --split_number=2 --part=trn &
python3 create_dataset_for_structured_learning.py --split_number=3 --part=trn &
python3 create_dataset_for_structured_learning.py --split_number=4 --part=trn &

python3 create_dataset_for_structured_learning.py --split_number=0 --part=val &
python3 create_dataset_for_structured_learning.py --split_number=1 --part=val &
python3 create_dataset_for_structured_learning.py --split_number=2 --part=val &
python3 create_dataset_for_structured_learning.py --split_number=3 --part=val &
python3 create_dataset_for_structured_learning.py --split_number=4 --part=val &

python3 create_dataset_for_structured_learning.py --split_number=0 --part=tst &
python3 create_dataset_for_structured_learning.py --split_number=1 --part=tst &
python3 create_dataset_for_structured_learning.py --split_number=2 --part=tst &
python3 create_dataset_for_structured_learning.py --split_number=3 --part=tst &
python3 create_dataset_for_structured_learning.py --split_number=4 --part=tst &