# import law
# import os

# law.contrib.load("root")  # enable ROOT contrib


# class MergeNanos(law.Task):

#     # target size per merged file (≈7.5 GB)
#     target_size_gb = 2.4

#     def output(self):
#         # base name (suffix _XXX.root will be added)
#         return law.LocalFileTarget(
#             "/data/dust/user/markusla/public/hh2bbww/mcproduction/samples/"
#             "GF_HHH_c3_0_d4_0/Run3Summer24/GF_HHH_Run3Summer24_merged" # noqa
#         )

#     def run(self):

#         # ---------------------------------------------------------
#         # collect input ROOT files
#         # ---------------------------------------------------------
#         def get_local_lfns():
#             base_dir = (
#                 "/data/dust/user/matsch/mcproduction/samples/"
#                 "GF_HHH_c3_0_d4_99/Run3Summer24"
#             )
#             lfns = []

#             for batch in ["batch_0"]:
#                 batch_dir = os.path.join(base_dir, batch)

#                 for fname in sorted(os.listdir(batch_dir)):
#                     if fname.endswith(".root"):
#                         lfns.append(os.path.join(batch_dir, fname))

#             return lfns

#         # ---------------------------------------------------------
#         # group files by cumulative size
#         # ---------------------------------------------------------
#         def chunk_by_size(files, target_size_bytes):
#             chunks = []
#             current_chunk = []
#             current_size = 0

#             for f in files:
#                 size = os.path.getsize(f)

#                 if current_chunk and current_size + size > target_size_bytes:
#                     chunks.append(current_chunk)
#                     current_chunk = []
#                     current_size = 0

#                 current_chunk.append(f)
#                 current_size += size

#             if current_chunk:
#                 chunks.append(current_chunk)

#             return chunks

#         # ---------------------------------------------------------
#         # main logic
#         # ---------------------------------------------------------
#         input_paths = get_local_lfns()

#         target_size_bytes = self.target_size_gb * 1024**3
#         chunks = chunk_by_size(input_paths, target_size_bytes)

#         output_base = self.output()
#         output_base.parent.touch()

#         self.logger.info(f"Merging into {len(chunks)} output files")

#         for i, chunk in enumerate(chunks):
#             out_file = f"{output_base.abspath}_{i:03d}.root"

#             self.logger.info(
#                 f" → output {i:03d}: {len(chunk)} files"  # noqa
#             )

#             law.root.hadd_task(
#                 self,
#                 chunk,
#                 out_file,
#                 local=True,
#                 hadd_args=["-O", "-f501"],
#             )

import law
import os
import luigi

law.contrib.load("root")  # enable ROOT contrib


class MergeNanos(law.Task):

    # required parameter (no default)
    sample = luigi.Parameter(
        description="Sample name, e.g. GF_HHH_c3_0_d4_0",
    )

    # target size per merged file (≈2.4 GB)
    target_size_gb = 2.4

    def output(self):
        return law.LocalFileTarget(
            f"/data/dust/user/markusla/public/hh2bbww/mcproduction/samples/"
            f"{self.sample}/Run3Summer24/{self.sample}_merged",
        )

    def run(self):

        # ---------------------------------------------------------
        # collect input ROOT files
        # ---------------------------------------------------------
        def get_local_lfns():
            base_dir = (
                f"/data/dust/user/matsch/mcproduction/samples/"
                f"{self.sample}/Run3Summer24"
            )
            lfns = []

            for batch in ["batch_0"]:
                batch_dir = os.path.join(base_dir, batch)

                for fname in sorted(os.listdir(batch_dir)):
                    if fname.endswith(".root"):
                        lfns.append(os.path.join(batch_dir, fname))

            return lfns

        # ---------------------------------------------------------
        # group files by cumulative size
        # ---------------------------------------------------------
        def chunk_by_size(files, target_size_bytes):
            chunks = []
            current_chunk = []
            current_size = 0

            for f in files:
                size = os.path.getsize(f)

                if current_chunk and current_size + size > target_size_bytes:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0

                current_chunk.append(f)
                current_size += size

            if current_chunk:
                chunks.append(current_chunk)

            return chunks

        # ---------------------------------------------------------
        # main logic
        # ---------------------------------------------------------
        input_paths = get_local_lfns()

        target_size_bytes = self.target_size_gb * 1024**3
        chunks = chunk_by_size(input_paths, target_size_bytes)

        output_base = self.output()
        output_base.parent.touch()

        self.logger.info(f"Merging into {len(chunks)} output files")

        for i, chunk in enumerate(chunks):
            out_file = f"{output_base.abspath}_{i:03d}.root"
            self.logger.info(
                f" → output {i:03d}: {len(chunk)} files",
            )

            law.root.hadd_task(
                self,
                chunk,
                out_file,
                local=True,
                hadd_args=["-O", "-f501"],
            )
