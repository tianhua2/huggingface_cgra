from typing import Optional

from huggingface_hub import Discussion, HfApi

from .utils import cached_file, logging


logger = logging.get_logger(__name__)


def previous_pr(api: HfApi, model_id: str, pr_title: str) -> Optional["Discussion"]:
    try:
        main_commit = api.list_repo_commits(model_id)[0].commit_id
        discussions = api.get_repo_discussions(repo_id=model_id)
    except Exception as e:
        logger.info(f"Could not retrieve repository discussions: {repr(e)}")
        return None
    for discussion in discussions:
        if discussion.status == "open" and discussion.is_pull_request and discussion.title == pr_title:
            commits = api.list_repo_commits(model_id, revision=discussion.git_reference)

            if main_commit == commits[1].commit_id:
                return discussion
    return None


def spawn_conversion(token: str, private: bool, model_id: str):
    logger.info("Attempting to convert .bin model on the fly to safetensors.")

    try:
        import json
        import uuid

        import requests
    except (ImportError, ModuleNotFoundError) as e:
        raise ValueError("Could not perform on-the-fly conversion as `requests` isn't installed.") from e

    sse_url = "https://safetensors-convert.hf.space/queue/join"
    sse_data_url = "https://safetensors-convert.hf.space/queue/data"
    hash_data = {"fn_index": 1, "session_hash": str(uuid.uuid4())}

    def start(_sse_connection, payload):
        for line in _sse_connection.iter_lines():
            line = line.decode()
            if line.startswith("data:"):
                resp = json.loads(line[5:])
                print(resp)

                if resp["msg"] == "queue_full":
                    raise ValueError("Queue is full! Please try again.")
                elif resp["msg"] == "send_data":
                    event_id = resp["event_id"]
                    response = requests.post(
                        sse_data_url,
                        stream=True,
                        params=hash_data,
                        json={"event_id": event_id, **payload, **hash_data},
                    )
                    response.raise_for_status()
                elif resp["msg"] == "process_completed":
                    return

    with requests.get(sse_url, stream=True, params=hash_data) as sse_connection:
        data = {"data": [model_id, private, token]}
        try:
            start(sse_connection, data)
        except Exception as e:
            logger.info(f"Error during conversion: {repr(e)}")


def get_sha(api: HfApi, model_id: str, **kwargs):
    private = api.model_info(model_id).private

    logger.info("Attempting to create safetensors variant")
    pr_title = "Adding `safetensors` variant of this model"

    pr = previous_pr(api, model_id, pr_title)

    if pr is None:
        spawn_conversion(kwargs.get("token"), private, model_id)
        pr = previous_pr(api, model_id, pr_title)
    else:
        logger.info("Safetensors PR exists")

    sha = f"refs/pr/{pr.num}"

    return sha


def auto_conversion(pretrained_model_name_or_path: str, **cached_file_kwargs):
    api = HfApi(token=cached_file_kwargs.get("token"))
    sha = get_sha(api, pretrained_model_name_or_path, **cached_file_kwargs)

    if sha is None:
        return None, None
    cached_file_kwargs["revision"] = sha
    del cached_file_kwargs["_commit_hash"]

    # This is an additional HEAD call that could be removed if we could infer sharded/non-sharded from the PR
    # description.
    sharded = api.file_exists(pretrained_model_name_or_path, "model.safetensors.index.json", revision=sha)
    filename = "model.safetensors.index.json" if sharded else "model.safetensors"

    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
    return resolved_archive_file, sha, sharded
