#!/usr/bin/env python3
"""Runner script that captures screenshots from Warp examples.

Patches matplotlib to save figures to files instead of displaying them,
then runs the example as __main__ via runpy.
"""

import os
import runpy
import sys
import traceback

# Set matplotlib to non-interactive backend BEFORE any imports
os.environ["MPLBACKEND"] = "Agg"

SCREENSHOT_DIR = os.path.dirname(os.path.abspath(__file__))


def patch_matplotlib(example_name):
    """Patch matplotlib to save figures instead of showing them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _original_show = plt.show
    _fig_counter = [0]

    def patched_show(*args, **kwargs):
        figs = [plt.figure(i) for i in plt.get_fignums()]
        for fig in figs:
            _fig_counter[0] += 1
            outpath = os.path.join(
                SCREENSHOT_DIR,
                f"{example_name}_fig{_fig_counter[0]}.png",
            )
            fig.savefig(outpath, dpi=150, bbox_inches="tight")
            print(f"[SCREENSHOT] Saved: {outpath}")
        plt.close("all")

    plt.show = patched_show

    # Also patch matplotlib.image.imsave to redirect output
    import matplotlib.image as mpimg
    _original_imsave = mpimg.imsave

    def patched_imsave(fname, arr, **kwargs):
        basename = os.path.basename(fname)
        outpath = os.path.join(SCREENSHOT_DIR, basename)
        _original_imsave(outpath, arr, **kwargs)
        print(f"[SCREENSHOT] Saved imsave: {outpath}")

    mpimg.imsave = patched_imsave

    # Patch animation saving
    import matplotlib.animation as animation
    _OrigArtistAnimation = animation.ArtistAnimation
    _OrigFuncAnimation = animation.FuncAnimation

    class PatchedArtistAnimation(_OrigArtistAnimation):
        def save(self, filename, *args, **kwargs):
            basename = os.path.basename(filename)
            outpath = os.path.join(SCREENSHOT_DIR, basename)
            super().save(outpath, *args, **kwargs)
            print(f"[SCREENSHOT] Saved animation: {outpath}")

    class PatchedFuncAnimation(_OrigFuncAnimation):
        def save(self, filename, *args, **kwargs):
            basename = os.path.basename(filename)
            outpath = os.path.join(SCREENSHOT_DIR, basename)
            super().save(outpath, *args, **kwargs)
            print(f"[SCREENSHOT] Saved animation: {outpath}")

    animation.ArtistAnimation = PatchedArtistAnimation
    animation.FuncAnimation = PatchedFuncAnimation


def patch_pil(example_name):
    """Patch PIL Image.save to redirect to screenshot dir."""
    try:
        from PIL import Image
        _original_save = Image.Image.save

        def patched_save(self, fp, *args, **kwargs):
            if isinstance(fp, str):
                basename = os.path.basename(fp)
                outpath = os.path.join(SCREENSHOT_DIR, basename)
                _original_save(self, outpath, *args, **kwargs)
                print(f"[SCREENSHOT] Saved PIL: {outpath}")
            else:
                _original_save(self, fp, *args, **kwargs)

        Image.Image.save = patched_save
    except ImportError:
        pass


def main():
    if len(sys.argv) < 2:
        print("Usage: run_example.py <module.path> [extra_args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    example_name = module_path.split(".")[-1]

    # Remove our script name, keep module path and extra args
    # runpy will see sys.argv[0] as the module
    sys.argv = sys.argv[1:]

    print(f"\n{'='*60}")
    print(f"Running: {module_path}")
    print(f"{'='*60}")

    # Apply patches before running the example
    patch_matplotlib(example_name)
    patch_pil(example_name)

    try:
        runpy.run_module(module_path, run_name="__main__", alter_sys=True)
        print(f"[OK] {module_path} completed successfully")
        return True
    except SystemExit as e:
        if e.code == 0 or e.code is None:
            print(f"[OK] {module_path} completed (SystemExit 0)")
            return True
        else:
            print(f"[ERROR] {module_path} exited with code {e.code}")
            return False
    except Exception as e:
        print(f"[ERROR] {module_path} failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
