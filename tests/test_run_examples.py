"""
Pytest suite that runs every example from run_examples.sh and sc_examples.sh,
then checks that expected output files were written and are not empty.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "inputs"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run(args: list[str], out_folder: str) -> Path:
    """Run ligandmpnn-run with the given args, return the output folder Path."""
    out = ROOT / out_folder
    if out.exists():
        shutil.rmtree(out)
    cmd = ["ligandmpnn-run"] + args
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (
        f"ligandmpnn-run failed (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    return out


def _assert_outputs(out: Path, expect_seqs: bool = True, expect_backbones: bool = True,
                    expect_stats: bool = False, expect_packed: bool = False,
                    min_files: int = 1, min_packed_files: int = 1):
    """Assert that the output folder has the expected non-empty files."""
    assert out.exists(), f"Output folder does not exist: {out}"

    if expect_seqs:
        seqs = list((out / "seqs").glob("*.fa"))
        assert len(seqs) >= min_files, f"Expected ≥{min_files} .fa files in {out}/seqs, got {len(seqs)}"
        for f in seqs:
            assert f.stat().st_size > 0, f"Empty file: {f}"

    if expect_backbones:
        bb = list((out / "backbones").glob("*.pdb"))
        assert len(bb) >= min_files, f"Expected ≥{min_files} .pdb files in {out}/backbones, got {len(bb)}"
        for f in bb:
            assert f.stat().st_size > 0, f"Empty file: {f}"

    if expect_stats:
        stats = list((out / "stats").glob("*.pt"))
        assert len(stats) >= 1, f"Expected stats .pt files in {out}/stats"
        for f in stats:
            assert f.stat().st_size > 0, f"Empty file: {f}"

    if expect_packed:
        packed = list((out / "packed").glob("*.pdb"))
        assert len(packed) >= min_packed_files, f"Expected ≥{min_packed_files} .pdb files in {out}/packed, got {len(packed)}"
        for f in packed:
            assert f.stat().st_size > 0, f"Empty file: {f}"



# ===========================================================================
#  run_examples.sh  (tests 1-33)
# ===========================================================================

class TestRunExamples:
    """Each test mirrors a numbered example from run_examples.sh."""

    # 1 – default
    def test_01_default(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_default"], "outputs/test_default")
        _assert_outputs(out)

    # 2 – temperature
    def test_02_temperature(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--temperature", "0.05",
                     "--out_folder", "./outputs/test_temperature"], "outputs/test_temperature")
        _assert_outputs(out)

    # 3 – random seed
    def test_03_random_seed(self):
        out = _run(["--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_random_seed"], "outputs/test_random_seed")
        _assert_outputs(out)

    # 4 – verbose 0
    def test_04_verbose(self):
        out = _run(["--seed", "111", "--verbose", "0",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_verbose"], "outputs/test_verbose")
        _assert_outputs(out)

    # 5 – save_stats
    def test_05_save_stats(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_save_stats",
                     "--save_stats", "1"], "outputs/test_save_stats")
        _assert_outputs(out, expect_stats=True)

    # 6 – fixed_residues + bias_AA
    def test_06_fix_residues(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_fix_residues",
                     "--fixed_residues", "C1 C2 C3 C4 C5 C6 C7 C8 C9 C10",
                     "--bias_AA", "A:10.0"], "outputs/test_fix_residues")
        _assert_outputs(out)

    # 7 – redesigned_residues + bias_AA
    def test_07_redesign_residues(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_redesign_residues",
                     "--redesigned_residues", "C1 C2 C3 C4 C5 C6 C7 C8 C9 C10",
                     "--bias_AA", "A:10.0"], "outputs/test_redesign_residues")
        _assert_outputs(out)

    # 8 – batch_size
    def test_08_batch_size(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_batch_size",
                     "--batch_size", "3",
                     "--number_of_batches", "5"], "outputs/test_batch_size")
        # 3 * 5 = 15 backbone PDBs, 1 .fa
        _assert_outputs(out, min_files=1)

    # 9 – global bias
    def test_09_global_bias(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--bias_AA", "W:3.0,P:3.0,C:3.0,A:-3.0",
                     "--out_folder", "./outputs/test_global_bias"], "outputs/test_global_bias")
        _assert_outputs(out)

    # 10 – per-residue bias
    def test_10_per_residue_bias(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--bias_AA_per_residue", "./inputs/bias_AA_per_residue.json",
                     "--out_folder", "./outputs/test_per_residue_bias"], "outputs/test_per_residue_bias")
        _assert_outputs(out)

    # 11 – global omit
    def test_11_global_omit(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--omit_AA", "CDFGHILMNPQRSTVWY",
                     "--out_folder", "./outputs/test_global_omit"], "outputs/test_global_omit")
        _assert_outputs(out)

    # 12 – per-residue omit
    def test_12_per_residue_omit(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--omit_AA_per_residue", "./inputs/omit_AA_per_residue.json",
                     "--out_folder", "./outputs/test_per_residue_omit"], "outputs/test_per_residue_omit")
        _assert_outputs(out)

    # 13 – symmetry
    def test_13_symmetry(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_symmetry",
                     "--symmetry_residues", "C1,C2,C3|C4,C5|C6,C7",
                     "--symmetry_weights", "0.33,0.33,0.33|0.5,0.5|0.5,0.5"], "outputs/test_symmetry")
        _assert_outputs(out)

    # 14 – homo-oligomer
    def test_14_homooligomer(self):
        out = _run(["--model_type", "ligand_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/4GYT.pdb",
                     "--out_folder", "./outputs/test_homooligomer",
                     "--homo_oligomer", "1",
                     "--number_of_batches", "2"], "outputs/test_homooligomer")
        _assert_outputs(out)

    # 15 – file_ending
    def test_15_file_ending(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_file_ending",
                     "--file_ending", "_xyz"], "outputs/test_file_ending")
        # Files will have _xyz suffix
        assert out.exists()
        fa_files = list((out / "seqs").glob("*_xyz.fa"))
        assert len(fa_files) >= 1, f"Expected *_xyz.fa files, got {fa_files}"
        pdb_files = list((out / "backbones").glob("*_xyz.pdb"))
        assert len(pdb_files) >= 1, f"Expected *_xyz.pdb files, got {pdb_files}"

    # 16 – zero_indexed
    def test_16_zero_indexed(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_zero_indexed",
                     "--zero_indexed", "1",
                     "--number_of_batches", "2"], "outputs/test_zero_indexed")
        _assert_outputs(out)

    # 17 – chains_to_design
    def test_17_chains_to_design(self):
        out = _run(["--model_type", "ligand_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/4GYT.pdb",
                     "--out_folder", "./outputs/test_chains_to_design",
                     "--chains_to_design", "A,B"], "outputs/test_chains_to_design")
        _assert_outputs(out)

    # 18 – parse_these_chains_only
    def test_18_parse_these_chains_only(self):
        out = _run(["--model_type", "ligand_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/4GYT.pdb",
                     "--out_folder", "./outputs/test_parse_these_chains_only",
                     "--parse_these_chains_only", "A,B"], "outputs/test_parse_these_chains_only")
        _assert_outputs(out)

    # 19 – ligand_mpnn default
    def test_19_ligandmpnn_default(self):
        out = _run(["--model_type", "ligand_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_ligandmpnn_default"], "outputs/test_ligandmpnn_default")
        _assert_outputs(out)

    # 20 – ligand_mpnn custom checkpoint
    def test_20_ligandmpnn_v_32_005_25(self):
        out = _run(["--checkpoint_ligand_mpnn", "ligandmpnn/model_params/ligandmpnn_v_32_005_25.pt",
                     "--model_type", "ligand_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_ligandmpnn_v_32_005_25"], "outputs/test_ligandmpnn_v_32_005_25")
        _assert_outputs(out)

    # 21 – ligand_mpnn no context
    def test_21_ligandmpnn_no_context(self):
        out = _run(["--model_type", "ligand_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_ligandmpnn_no_context",
                     "--ligand_mpnn_use_atom_context", "0"], "outputs/test_ligandmpnn_no_context")
        _assert_outputs(out)

    # 22 – ligand_mpnn side-chain context
    def test_22_ligandmpnn_side_chain_context(self):
        out = _run(["--model_type", "ligand_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_ligandmpnn_sc_context",
                     "--ligand_mpnn_use_side_chain_context", "1",
                     "--fixed_residues", "C1 C2 C3 C4 C5 C6 C7 C8 C9 C10"], "outputs/test_ligandmpnn_sc_context")
        _assert_outputs(out)

    # 23 – soluble_mpnn
    def test_23_soluble_mpnn(self):
        out = _run(["--model_type", "soluble_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_soluble_mpnn"], "outputs/test_soluble_mpnn")
        _assert_outputs(out)

    # 24 – global_label_membrane_mpnn
    def test_24_global_label_membrane_mpnn(self):
        out = _run(["--model_type", "global_label_membrane_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_global_membrane",
                     "--global_transmembrane_label", "0"], "outputs/test_global_membrane")
        _assert_outputs(out)

    # 25 – per_residue_label_membrane_mpnn
    def test_25_per_residue_label_membrane_mpnn(self):
        out = _run(["--model_type", "per_residue_label_membrane_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_per_residue_membrane",
                     "--transmembrane_buried", "C1 C2 C3 C11",
                     "--transmembrane_interface", "C4 C5 C6 C22"], "outputs/test_per_residue_membrane")
        _assert_outputs(out)

    # 26 – fasta_seq_separation
    def test_26_fasta_seq_separation(self):
        out = _run(["--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_fasta_sep",
                     "--fasta_seq_separation", ":"], "outputs/test_fasta_sep")
        _assert_outputs(out)

    # 27 – pdb_path_multi
    def test_27_pdb_path_multi(self):
        out = _run(["--pdb_path_multi", "./inputs/pdb_ids.json",
                     "--out_folder", "./outputs/test_pdb_path_multi",
                     "--seed", "111"], "outputs/test_pdb_path_multi")
        _assert_outputs(out, min_files=2)  # multiple PDBs

    # 28 – fixed_residues_multi
    def test_28_fixed_residues_multi(self):
        out = _run(["--pdb_path_multi", "./inputs/pdb_ids.json",
                     "--fixed_residues_multi", "./inputs/fix_residues_multi.json",
                     "--out_folder", "./outputs/test_fixed_residues_multi",
                     "--seed", "111"], "outputs/test_fixed_residues_multi")
        _assert_outputs(out, min_files=2)

    # 29 – redesigned_residues_multi
    def test_29_redesigned_residues_multi(self):
        out = _run(["--pdb_path_multi", "./inputs/pdb_ids.json",
                     "--redesigned_residues_multi", "./inputs/redesigned_residues_multi.json",
                     "--out_folder", "./outputs/test_redesigned_residues_multi",
                     "--seed", "111"], "outputs/test_redesigned_residues_multi")
        _assert_outputs(out, min_files=2)

    # 30 – omit_AA_per_residue_multi
    def test_30_omit_AA_per_residue_multi(self):
        out = _run(["--pdb_path_multi", "./inputs/pdb_ids.json",
                     "--omit_AA_per_residue_multi", "./inputs/omit_AA_per_residue_multi.json",
                     "--out_folder", "./outputs/test_omit_AA_multi",
                     "--seed", "111"], "outputs/test_omit_AA_multi")
        _assert_outputs(out, min_files=2)

    # 31 – bias_AA_per_residue_multi
    def test_31_bias_AA_per_residue_multi(self):
        out = _run(["--pdb_path_multi", "./inputs/pdb_ids.json",
                     "--bias_AA_per_residue_multi", "./inputs/bias_AA_per_residue_multi.json",
                     "--out_folder", "./outputs/test_bias_AA_multi",
                     "--seed", "111"], "outputs/test_bias_AA_multi")
        _assert_outputs(out, min_files=2)

    # 32 – ligand_mpnn_cutoff_for_score
    def test_32_ligand_mpnn_cutoff_for_score(self):
        out = _run(["--model_type", "ligand_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--ligand_mpnn_cutoff_for_score", "6.0",
                     "--out_folder", "./outputs/test_cutoff_score"], "outputs/test_cutoff_score")
        _assert_outputs(out)

    # 33 – insertion code
    def test_33_insertion_code(self):
        out = _run(["--seed", "111",
                     "--pdb_path", "./inputs/2GFB.pdb",
                     "--out_folder", "./outputs/test_insertion_code",
                     "--redesigned_residues", "B82 B82A B82B B82C",
                     "--parse_these_chains_only", "B"], "outputs/test_insertion_code")
        _assert_outputs(out)


# ===========================================================================
#  sc_examples.sh  (side-chain packing tests)
# ===========================================================================

class TestSideChainExamples:
    """Each test mirrors a numbered example from sc_examples.sh."""

    # SC-1 – side chain packing (fast, 0 extra packs)
    def test_sc_01_default_fast(self):
        out = _run(["--model_type", "ligand_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_sc_default_fast",
                     "--pack_side_chains", "1",
                     "--number_of_packs_per_design", "0",
                     "--pack_with_ligand_context", "1"], "outputs/test_sc_default_fast")
        _assert_outputs(out, expect_packed=False)

    # SC-2 – side chain packing (4 packs)
    def test_sc_02_default(self):
        out = _run(["--model_type", "ligand_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_sc_default",
                     "--pack_side_chains", "1",
                     "--number_of_packs_per_design", "4",
                     "--pack_with_ligand_context", "1"], "outputs/test_sc_default")
        _assert_outputs(out, expect_packed=True, min_packed_files=4)

    # SC-3 – fixed residues for packing
    def test_sc_03_fixed_residues(self):
        out = _run(["--model_type", "ligand_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_sc_fixed",
                     "--pack_side_chains", "1",
                     "--number_of_packs_per_design", "4",
                     "--pack_with_ligand_context", "1",
                     "--fixed_residues", "C6 C7 C8 C9 C10 C11 C12 C13 C14 C15",
                     "--repack_everything", "0"], "outputs/test_sc_fixed")
        _assert_outputs(out, expect_packed=True, min_packed_files=4)

    # SC-4 – fixed residues, full repack
    def test_sc_04_fixed_residues_full_repack(self):
        out = _run(["--model_type", "ligand_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_sc_full_repack",
                     "--pack_side_chains", "1",
                     "--number_of_packs_per_design", "4",
                     "--pack_with_ligand_context", "1",
                     "--fixed_residues", "C6 C7 C8 C9 C10 C11 C12 C13 C14 C15",
                     "--repack_everything", "1"], "outputs/test_sc_full_repack")
        _assert_outputs(out, expect_packed=True, min_packed_files=4)

    # SC-5 – no ligand context for packing
    def test_sc_05_no_context(self):
        out = _run(["--model_type", "ligand_mpnn",
                     "--seed", "111",
                     "--pdb_path", "./inputs/1BC8.pdb",
                     "--out_folder", "./outputs/test_sc_no_context",
                     "--pack_side_chains", "1",
                     "--number_of_packs_per_design", "4",
                     "--pack_with_ligand_context", "0"], "outputs/test_sc_no_context")
        _assert_outputs(out, expect_packed=True, min_packed_files=4)
