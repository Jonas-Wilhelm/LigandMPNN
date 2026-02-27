"""Download LigandMPNN model weights into the package's model_params/ directory."""

import argparse
import os
import sys
import urllib.request

from ligandmpnn._paths import get_model_params_dir

# All available model weight URLs
WEIGHTS = {
    # Original ProteinMPNN weights
    "proteinmpnn_v_48_002.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_002.pt",
    "proteinmpnn_v_48_010.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_010.pt",
    "proteinmpnn_v_48_020.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt",
    "proteinmpnn_v_48_030.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_030.pt",
    # LigandMPNN with num_edges=32; atom_context_num=25
    "ligandmpnn_v_32_005_25.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_005_25.pt",
    "ligandmpnn_v_32_010_25.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt",
    "ligandmpnn_v_32_020_25.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_25.pt",
    "ligandmpnn_v_32_030_25.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_030_25.pt",
    # Per residue label membrane ProteinMPNN
    "per_residue_label_membrane_mpnn_v_48_020.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/per_residue_label_membrane_mpnn_v_48_020.pt",
    # Global label membrane ProteinMPNN
    "global_label_membrane_mpnn_v_48_020.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/global_label_membrane_mpnn_v_48_020.pt",
    # SolubleMPNN
    "solublempnn_v_48_002.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_002.pt",
    "solublempnn_v_48_010.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_010.pt",
    "solublempnn_v_48_020.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt",
    "solublempnn_v_48_030.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_030.pt",
    # LigandMPNN for side-chain packing
    "ligandmpnn_sc_v_32_002_16.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_sc_v_32_002_16.pt",
}


def _download(url: str, dest: str) -> None:
    """Download a single file with a progress indicator."""
    filename = os.path.basename(dest)
    print(f"  Downloading {filename} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print("done")
    except Exception as e:
        print(f"FAILED: {e}")
        raise


def main_cli():
    parser = argparse.ArgumentParser(
        description="Download LigandMPNN model weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        default=None,
        help="Specific weight filenames to download (e.g. proteinmpnn_v_48_020.pt). "
             "If omitted, downloads the default set of weights needed for typical usage.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help="Download ALL available model weights.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download weights even if they already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_weights",
        help="List all available weight filenames and exit.",
    )
    args = parser.parse_args()

    if args.list_weights:
        print("Available model weights:")
        for name in sorted(WEIGHTS):
            print(f"  {name}")
        return

    model_dir = get_model_params_dir()
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model weights directory: {model_dir}")

    # Determine which weights to download
    if args.download_all:
        to_download = dict(WEIGHTS)
    elif args.weights:
        to_download = {}
        for w in args.weights:
            if w not in WEIGHTS:
                print(f"ERROR: Unknown weight '{w}'. Run with --list to see options.", file=sys.stderr)
                sys.exit(1)
            to_download[w] = WEIGHTS[w]
    else:
        # Default set: one of each model type needed for typical inference
        default_names = [
            "proteinmpnn_v_48_020.pt",
            "ligandmpnn_v_32_010_25.pt",
            "ligandmpnn_v_32_020_25.pt",
            "per_residue_label_membrane_mpnn_v_48_020.pt",
            "global_label_membrane_mpnn_v_48_020.pt",
            "solublempnn_v_48_020.pt",
            "ligandmpnn_sc_v_32_002_16.pt",
        ]
        to_download = {k: WEIGHTS[k] for k in default_names}

    downloaded = 0
    skipped = 0
    for name, url in to_download.items():
        dest = os.path.join(model_dir, name)
        if os.path.exists(dest) and not args.force:
            print(f"  Skipping {name} (already exists, use --force to re-download)")
            skipped += 1
            continue
        _download(url, dest)
        downloaded += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped.")


if __name__ == "__main__":
    main_cli()
