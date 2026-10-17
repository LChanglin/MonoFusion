# Verified checkpoint downloads

The updated compatibility checkpoints are publicly downloadable from [Google Drive](https://drive.google.com/drive/folders/1ebfjezbP6rDpzH5ofFLMWLl50YtE8-kT). Both listed files were downloaded anonymously on October 17, 2026, matched the GPU-validated release byte-for-byte, and passed `strict=True` loading (387 tensors each).

Place the files directly in `pretrained/` and use the updated `submission.py`; **no conversion is required**. It selects `legacy_scale` from the embedded migration metadata. Use `--num_iterations 1` for single-pass inference or `--num_iterations 4` for the recurrent configuration.

| File | Bytes | Download |
| --- | ---: | --- |
| `monofusion_kitti.pth.tar` | 84,831,022 | [Download](https://drive.google.com/uc?export=download&id=1Zifo-hFbQbGptdq5TRGignfWFhFZNtOY) |
| `monofusion_r_kitti.pth.tar` | 84,831,872 | [Download](https://drive.google.com/uc?export=download&id=1I2Uv9tU4_aHhTsW7hfi-61WQe1yfwOld) |

## Verify your downloads

The machine-readable [manifest](checkpoints.json) records the exact release verified here. From the repository root, the bundled helper downloads the two inference packages and checks their size and SHA-256 before saving:

```bash
python skills/monofusion-reproduce/scripts/download_checkpoints.py --repo .
```

Matching existing files are reused; mismatched files are preserved and reported rather than overwritten. A changed upstream checksum requires a fresh release audit.

```text
64073f9b4a6bbc14e6daa55a3a6b1c83e4427c0dda418d8bc116b7558bbf1564  monofusion_kitti.pth.tar
f99a4dc5beb5f16eb0ee63793d0f7b030dccd91d324dc38efd66dd32f5764a54  monofusion_r_kitti.pth.tar
```

## Scope and limitations

- Both packages contain the **same trained tensors**. Their complete-file hashes differ because filenames/provenance metadata differ. They are not independently verified training variants.
- These files are not drop-in weights for the original training scripts.
- The tensors passed 40-pair GPU inference and historical-image comparisons at 1 and 4 iterations. This release audit does not reproduce the paper's KITTI/FlyingThings3D benchmark scores.
- Full scene-flow export requires external disparity inputs; flow/expansion inference needs only the image pair.

See the [reproduction skill](../skills/monofusion-reproduce/SKILL.md), [main quick start](../README.md#evaluation--inference), and [conversion instructions for older files](../migration/USAGE.txt).
