import os


if __name__ == "__main__":
    python_path = "/home/yangzheng/miniconda3/envs/mindspore-2/bin/python"

    python_file = "/home/yangzheng/layout-diffusion-mindspore/scripts/image_train_for_layout.py "

    total_process = "-n {}".format(8)

    command_str = 'mpirun {0} {1} {2}'.format(total_process, python_path, python_file)
    print(command_str)

    os.system(command_str)