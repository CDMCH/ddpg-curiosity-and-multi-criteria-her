import os
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from mpi4py import MPI


class DynamicsLossMapper:

    def __init__(self, working_dir, sample_env):
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        self.record_write_max_steps = 4000

        if MPI.COMM_WORLD.Get_rank() == 0:
            sample_env.save_heatmap_picture(os.path.join(self.working_dir, 'level.png'))

        self.losses_locations_record = None
        self.record_write_steps_recorded = 0

    def set_record_write(self, prefix):

        self.flush_record_write()
        self.record_write_prefix = prefix
        self.record_write_file_number = 0

        return True

    def flush_record_write(self, create_new_record=True):
        if self.losses_locations_record is not None and self.record_write_steps_recorded > 0:
            write_file = os.path.join(self.working_dir, "{}_{}".format(self.record_write_prefix,
                                                                            self.record_write_file_number))
            np.save(write_file, self.losses_locations_record[:self.record_write_steps_recorded])
            self.record_write_file_number += 1
            self.record_write_steps_recorded = 0

        if create_new_record:
            self.losses_locations_record = np.empty(shape=(self.record_write_max_steps, 3), dtype=np.float32)

    def log_losses_at_locations(self, losses, locations):
        if self.losses_locations_record is not None:
            for loss, location in zip(losses, locations):
                self.losses_locations_record[self.record_write_steps_recorded] = [loss, *location]
                self.record_write_steps_recorded += 1

                if self.record_write_steps_recorded >= self.record_write_max_steps:
                    self.flush_record_write()

    def generate_dynamics_loss_map_from_npy_records(self, file_prefix, delete_records=False):

        file_names = [file_name for file_name in os.listdir(self.working_dir)
                      if file_name.endswith(".npy") and file_name.startswith(file_prefix)]

        losses_locations_records = np.concatenate([np.load(os.path.join(self.working_dir, file_name)) for file_name in file_names],
                                          axis=0)

        max_heatmap_samples = 10000

        losses_locations_records = losses_locations_records[np.random.choice(len(losses_locations_records),
                                                             min(max_heatmap_samples, len(losses_locations_records)),
                                                             replace=False)]

        # print("\n\n\nlosses_locations_records:\n{}\n".format(losses_locations_records))
        # exit(0)

        # Add a minuscule amount of location variation in case agent doesn't move on a certain axis.

        # print("z min: {} max: {}".format(min(z), max(z)))

        losses_locations_records[:, 1:] = losses_locations_records[:, 1:] * 100 + (np.random.randn(*losses_locations_records[:, 1:].shape) / 1000)

        losses_locations_records = losses_locations_records.swapaxes(0, 1)

        z = losses_locations_records[0, :]

        idx = z.argsort()
        x, y, z = losses_locations_records[1, idx], losses_locations_records[2, idx], z[idx]

        plt.figure(figsize=(3, 3))
        plt.scatter(x, y, c=z, s=80, edgecolors='', cmap=plt.cm.jet, alpha=0.5)
        plt.colorbar()
        plt.xlim(0, 100)
        plt.ylim(0, 100)

        im = plt.imread(os.path.join(self.working_dir, 'level.png'))
        plt.imshow(im, extent=[0, 100, 0, 100], aspect='auto')
        plt.axis('equal')
        plt.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title("Dynamics Loss " + file_prefix)

        heatmap_image_path = os.path.join(self.working_dir, "{}_dynamics_loss.png".format(file_prefix))
        plt.savefig(heatmap_image_path, transparent=False, bbox_inches='tight', pad_inches=0)

        plt.close()

        if delete_records:
            for file_name in file_names:
                try:
                    os.remove(os.path.join(self.working_dir, file_name))
                except OSError:
                    pass

        return heatmap_image_path

    def __del__(self):
        self.flush_record_write(create_new_record=False)