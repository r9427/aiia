import platform
import shutil
from pathlib import Path

from cx_Freeze import setup


BASE_DIR = Path(__file__).resolve().parent
system = platform.system().lower()
platform_dict = {
    "windows": {"format": "zip", "script": "startup.bat"},
    "linux": {"format": "xztar", "script": "startup.sh"},
    "darwin": {"format": "gztar", "script": "startup.sh"}
}

if system not in platform_dict:
    raise Exception("Unsupported platform: {}".format(system))

def remove_dir(target_path):
    if target_path.exists():
        shutil.rmtree(target_path)
    # target_path.mkdir()

def get_version(path):
    with open(path, "r") as file:
        content = file.read()
    return content.strip()

# walnut-1.0.1-linux.tar.gz
app_name = "walnut-controller"
version = get_version(BASE_DIR.joinpath('version.txt'))
output_name = "{}-{}-{}".format(app_name, version, system)
build_dir = 'build'
build_path = BASE_DIR.joinpath(build_dir)
remove_dir(build_path)

# Dependencies are automatically detected, but they might need fine-tuning.
build_exe_options = {
    "build_exe": "{}/{}".format(build_dir, output_name),
    "excludes": [
        "app/test/*",
        "dev"
    ],
    "include_files": [
        "version.txt",
        ("config/default_application.ini", "config/application.ini"),
        ("ssl/placeholder.txt", "ssl/placeholder.txt"),
        platform_dict[system]["script"],
        # "config",
        # "ssl"
    ]
}

print("========== start to prepare files ==========")

setup(
    name="shellController",
    version="1.0.1",
    description="my smart shell 2",
    options={"build_exe": build_exe_options},
    executables=[{"script": "app/main.py", "target_name": "app"}],
)

print("========== start to compress files ==========")

final_file_name = shutil.make_archive(
    base_name='{}/{}'.format(build_dir, output_name),
    # format='zip',
    format=platform_dict[system]["format"],
    root_dir=build_dir,
    # root_dir="{}/{}".format(build_dir, output_name),
    base_dir=output_name
)

remove_dir(build_path.joinpath(output_name))

print("========== finish building {} ==========".format(final_file_name))