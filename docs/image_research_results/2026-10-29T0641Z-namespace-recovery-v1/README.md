# OmniAID metadata compatibility recovery

The original GPU verification failed on `argparse.Namespace` metadata after official asset import and exact safe tensor conversion succeeded. The failed queue and receipt remain preserved.

The scoped fix retains weights-only loading and permits only the SHA-checked official checkpoint’s statically verified stdlib Namespace metadata. Actual CPU inspection read 2804 tensor entries successfully; no images or GPU were used. No detector performance result is present in this snapshot.

The independent recovery queue skips only the completed conversion and uses fresh verification evidence. Model weights, architectures, data, roles and consistency tolerances remain unchanged. `snapshot.json` binds each copied source and hash.
