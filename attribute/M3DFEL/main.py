from options import Options
from solver import Solver


def main():
    """Run the whole training process through solver
    Change the options according to your situation, especially workers, gpu_ids, batch_size and epochs
    """
    args = Options().parse()
    solver = Solver(args)
    solver.run()


if __name__ == '__main__':
    main()
