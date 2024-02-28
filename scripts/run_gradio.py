import os

if __name__ == "__main__":
    python_path = "/home/yangzheng/miniconda3/envs/mindspore-2/bin/python"

    python_file = "/home/yangzheng/layout-diffusion-mindspore/scripts/launch_gradio_app.py "

    total_process = "-n {}".format(1)

    command_str = 'mpirun {0} {1} {2}'.format(total_process, python_path, python_file)
    print(command_str)

    os.system(command_str)
