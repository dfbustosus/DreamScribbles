from __future__ import annotations

from huggingface_hub import snapshot_download

from dreamscribbles.config import get_settings


def main() -> None:
    s = get_settings()
    print(f"Downloading base model: {s.model_id}")
    snapshot_download(repo_id=s.model_id, local_files_only=False, token=s.hf_token)
    print(f"Downloading controlnet: {s.controlnet_id}")
    snapshot_download(repo_id=s.controlnet_id, local_files_only=False, token=s.hf_token)
    print("Done.")


if __name__ == "__main__":
    main()
