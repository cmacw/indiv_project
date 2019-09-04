from PoseEstimation import PoseEstimation
import itertools
import torch
import os
import glob


def test_one_set(dataset, model, radial=True):
    trainset_info = {"path": "Train", "dataset_name": dataset, "cam_id": 0,
                     "image_name": "image_t_{}_cam_{}.png",
                     "pos_file_name": "cam_pos.csv",
                     "ndata": 10000, "epochs": 40, "batch_size": 32}

    testset_info = {"path": "Real_Test_Webcam", "dataset_name": dataset, "cam_id": 0,
                    "image_name": "{}.png",
                    "pos_file_name": "webcam_pos.csv",
                    "ndata": 261}

    trainer = PoseEstimation(trainset_info, radial=radial)
    trainer.net.load_state_dict(torch.load(model))
    trainer.load_test_set(testset_info, radial=radial, webcam_test=True)
    trainer.evaluation()


os.chdir("datasets/Set06")
model_state = glob.glob('Real_Test_Webcam/*.pt')

for state in model_state:
    print("\n"+state)
    test_one_set("Test", state)
