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
                     "ndata": 10000, "epochs": 25, "batch_size": 32}

    testset_info = {"path": "Test", "dataset_name": dataset + "_test", "cam_id": 0,
                    "image_name": "image_t_{}_cam_{}.png",
                    "pos_file_name": "cam_pos_test.csv",
                    "ndata": 1000}

    trainer = PoseEstimation(trainset_info)
    # Recover parameters. CHECK BEFORE RUN!!
    # trainer.net.load_state_dict(torch.load("Train/mdl_realistic_un_eph_25_btcsz_4.pt"))
    trainer.load_test_set(testset_info)
    trainer.train(show_fig=True, save_output=True, eval_eph=True)

    # Make a seperate object for evaluation
    # tester = PoseEstimation(testset_info)
    # tester.load_test_set(testset_info)
    # tester.net.load_state_dict(torch.load("Train/mdl_realistic_un_eph_25_btcsz_32.pt"))
    # tester.evaluation()


if __name__ == '__main__':
    os.chdir("datasets/Set02")
    datasets = ["random_mj", "random_un", "realistic_mj", "realistic_un"]

    run_one_set("realistic_un")
    # run_all_sets(datasets)
