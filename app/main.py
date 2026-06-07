# from agents.agent1.entry import run as run1
from agents.agent2.entry import run as run2
from util.Util import Util


def main():
    print("Hello from aiia!")
    base_path = Util.get_base_path("pyproject2.toml")
    print(f"Base path: {base_path}")

    run2()


if __name__ == "__main__":
    main()
