from PoseEstimation import PoseEstimation
import torch
import os


def run_all_sets(datasets):
    for dataset in datasets:
        run_one_set(dataset)


def run_one_set(dataset):
    # Train one set only
    trainset_info = {"path": "Train", "dataset_name": dataset, "cam_id": 0,
                     "image_name": "image_t_{}_cam_{}.png",
                     "pos_file_name": "cam_pos.csv",
                     "ndata": 10000, "epochs": 30, "batch_size": 32}

    testset_info = {"path": "Test", "dataset_name": dataset + "_test", "cam_id": 0,
                    "image_name": "image_t_{}_cam_{}.png",
                    "pos_file_name": "cam_pos_test.csv",
                    "ndata": 1000}

    trainer = PoseEstimation(trainset_info)
    # Recover parameters. CHECK BEFORE RUN!!
    # trainer.net.load_state_dict(torch.load("Train/realistic_un_results/mdl_realistic_un_eph_50_bs_128.pt"))
    # trainer.net.load_state_dict(torch.load("Debug/Train/mdl_01_fixed_eph_30_bs_32.pt"))
    trainer.load_test_set(testset_info)

    trainer.train(show_fig=True, save_output=True, eval_eph=True)
    # trainer.evaluation()

if __name__ == '__main__':
    os.chdir("datasets/Set05")
    datasets = ["random_mj", "random_un", "realistic_mj", "realistic_un"]

    # run_one_set("realistic_un")
    run_all_sets(datasets)

    debug = ["01_fixed", "02_radial", "03_2D", "04_3D", "05_3D"]
    # run_one_set("04_3D")
    # run_one_set("05_3D")
    # run_all_sets(debug)
