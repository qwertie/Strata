"""Checks model-file arguments without loading weights or starting an engine."""
from strata_common.paged_launch import translate_args


def test_ready_pack_preserves_separate_dense_embedding_and_ple_files():
    args = ["--pack", "nvfp4/pack", "--native", "dense.gguf",
            "--native-dense-gguf", "dense.gguf", "--native-dense-gguf", "other.gguf",
            "--embd-gguf", "embedding.gguf", "--ple-gguf", "ple.gguf"]
    assert translate_args({"exe": "strata-q38.exe"}, args) == args


def test_canonical_pack_preserves_source_model_for_native_projections():
    args = ["--pack", "packs/q2_0", "--native", "q2_0-00001-of-00002.gguf",
            "--ple-gguf", "q2_0-00002-of-00002.gguf"]
    assert translate_args({"exe": "strata-q38.exe"}, args) == args
