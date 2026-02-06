import law
import os

law.contrib.load("root")  # enable ROOT contrib


class MergeNanos(law.Task):

    def output(self):
        return law.LocalFileTarget(
            "/data/dust/user/markusla/public/hh2bbww/mcproduction/samples/GF_HHH_c3_0_d4_0/Run3Summer24/GF_HHH_Run3Summer24_merged.root"  # noqa E501
        )

    def run(self):

        def get_local_lfns():
            base_dir = "/data/dust/user/matsch/mcproduction/samples/GF_HHH_c3_0_d4_0/Run3Summer24"
            lfns = []

            for batch in ["batch_0", "batch_1"]:
                batch_dir = os.path.join(base_dir, batch)

                for fname in sorted(os.listdir(batch_dir)):
                    if fname.endswith(".root"):
                        lfns.append(f"{base_dir}/{batch}/{fname}")

            return lfns

        input_paths = get_local_lfns()

        output = self.output()
        output.parent.touch()  # ensure output directory exists

        law.root.hadd_task(
            self,
            input_paths,
            output.abspath,
            local=True,
            hadd_args=["-O", "-f501"],
            cascade_size=100,
        )
