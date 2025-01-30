import csv
import subprocess
import tempfile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import git
from tqdm import tqdm


@dataclass
class SccData:
    loc: int
    lloc: int
    uloc: int
    comments: int
    blanks: int
    complexity: int
    bytes: int
    language: str


def _load_bytes(repo: git.Repo, hexsha: str) -> bytes:
    return repo.odb.stream(bytes.fromhex(hexsha)).read()  # type: ignore


def _load_blob_name(repo: git.Repo) -> dict[str, str]:
    blob_to_name: dict[str, str] = dict()

    def visit(history: set[str], tree: git.Tree) -> None:
        if tree.hexsha in history:
            return
        history.add(tree.hexsha)
        for blob in tree.blobs:
            blob_to_name[blob.hexsha] = blob.name
        for other in tree.trees:
            visit(history, other)

    history: set[str] = set()
    for tree in tqdm(list(repo.iter_trees(all=True))):
        visit(history, tree)

    return blob_to_name


def _prepare_scc(tmp_dir: str, repo: git.Repo, blob_to_name: dict[str, str]) -> None:
    for blob, name in tqdm(blob_to_name.items()):
        content = _load_bytes(repo, blob)
        path = Path(tmp_dir, blob, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)


def _run_scc(tmp_dir: str) -> dict[str, SccData]:
    # Run SCC
    args = ["scc", "--by-file", "--format=csv"]
    res = subprocess.run(args, cwd=tmp_dir, capture_output=True, text=True)

    # Collect output into dict
    scc_data: dict[str, SccData] = dict()
    for row in csv.DictReader(StringIO(res.stdout)):
        hexsha = row["Provider"].split("/")[0]
        scc_data[hexsha] = SccData(
            loc=int(row["Lines"]),
            lloc=int(row["Code"]),
            uloc=int(row["ULOC"]),
            comments=int(row["Comments"]),
            blanks=int(row["Blanks"]),
            complexity=int(row["Complexity"]),
            bytes=int(row["Bytes"]),
            language=row["Language"],
        )
    return scc_data


def collect_scc(repo_path: str) -> dict[str, SccData]:
    repo = git.Repo(repo_path, odbt=git.GitCmdObjectDB)
    blob_to_name = _load_blob_name(repo)
    with tempfile.TemporaryDirectory() as tmp_dir:
        _prepare_scc(tmp_dir, repo, blob_to_name)
        return _run_scc(tmp_dir)
