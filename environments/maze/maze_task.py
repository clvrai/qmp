from pathlib import Path

from .maze import MazeTask


def load_init_goals(file_name):
    with open(Path(__file__).resolve().parent / "asset" / (file_name + ".csv")) as f:
        f.readline()
        lines = f.readlines()
    init_goal_loc = [[float(c) for c in l.strip().split(",")] for l in lines]
    return [
        tuple(l[i:i+2] for i in range(0, len(l), 2))  # locs are in 2D (x, y)
        for l in init_goal_loc
    ]


class Tasks:
    _file_selector = {}
    _default_num_tasks = None
    _task_selector = {}

    @classmethod
    def get_tasks(cls, num_tasks=None, task_indices=None, train=False):
        if num_tasks is None:
            num_tasks = cls._default_num_tasks
        tasks = cls._task_selector[num_tasks if train else 0]
        if task_indices is not None:
            tasks = [tasks[i] for i in task_indices]
        return tasks


class Size20Seed0Tasks(Tasks):
    _file_selector = {
        40: "size20/flat_40t_train_goals",
        20: "size20/flat_20t_train_goals",
        10: "size20/flat_10t_train_goals",
        0: "size20/flat_test_goals",
    }
    _default_num_tasks = 10
    _task_selector = {
        k: [MazeTask(*locs) for locs in load_init_goals(v)]
        for k, v in _file_selector.items()
    }


class MediumTasks(Tasks):  # maze_spec = MEDIUM_MAZE
    _file_selector = {
        3: "medium/medium_3t_train_goals",
        0: "medium/medium_3t_train_goals",
    }
    _default_num_tasks = 3
    _task_selector = {
        k: [MazeTask(*locs) for locs in load_init_goals(v)]
        for k, v in _file_selector.items()
    }


class LargeTasks(Tasks):  # maze_spec = LARGE_MAZE
    _file_selector = {
        20: "large/large_20t_train_goals",
        10: "large/large_10t_train_goals",
        5: "large/large_5t_train_goals",
        0: "large/large_10t_train_goals",
    }
    _default_num_tasks = 10
    _task_selector = {
        k: [MazeTask(*locs) for locs in load_init_goals(v)]
        for k, v in _file_selector.items()
    }


class MT10Tasks(Tasks):  # maze_spec = LARGE_MAZE
    _file_selector = {
        10: "mt10/mt10_train_goals",
        0: "mt10/mt10_train_goals",
    }
    _default_num_tasks = 10
    _task_selector = {
        k: [MazeTask(*locs) for locs in load_init_goals(v)]
        for k, v in _file_selector.items()
    }


def get_tasks(spec, num_tasks=None, task_indices=None, train=True):
    if spec in [20, "20"]:
        cls = Size20Seed0Tasks
    elif spec in ["MT10"]:
        cls = MT10Tasks
    elif spec in ["MEDIUM_MAZE"]:
        cls = MediumTasks
    elif spec in ["LARGE_MAZE"]:
        cls = LargeTasks
    else:
        raise ValueError("Unsupported maze spec: {}".format(spec))
    return cls.get_tasks(num_tasks, task_indices, train)
