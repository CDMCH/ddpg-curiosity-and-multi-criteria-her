from mpi4py import MPI


def demo(agents, eval_rollout_worker, demo_video_recording_name):

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        while True:
            eval_rollout_worker.generate_rollouts(render_override=True, reset_on_success_overrride=False,
                                                  demo_video_recording_name=demo_video_recording_name)
