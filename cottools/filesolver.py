import abc
import itertools as it
import pickle
import re
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from io import StringIO, TextIOBase
from typing import Any, Iterable

import networkx as nx
from bidict import bidict

_NULL_COMMIT_ID = "0" * 40


class _NameTableListener(abc.ABC):
    @abc.abstractmethod
    def on_add(self, id: int, name: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_delete(self, id: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_modify(self, id: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_rename(self, old_id: int, new_id: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_merge(self, old_id: int, new_id: int) -> None:
        raise NotImplementedError


class _IdProvider:
    def __init__(self) -> None:
        self._id = 0

    def next_id(self) -> int:
        id = self._id
        self._id += 1
        return id


class _NameTable:
    _table: bidict[str, int]
    _id_provider: _IdProvider
    _listeners: list[_NameTableListener]

    def __init__(self, parent: "_NameTable | None", id_provider: _IdProvider) -> None:
        if parent is None:
            self._table = bidict()
            self._id_provider = id_provider
            self._listeners = []
        else:
            self._table = bidict(parent._table)
            self._id_provider = id_provider
            self._listeners = parent._listeners

    def add_listener(self, listener: _NameTableListener) -> None:
        self._listeners.append(listener)

    def create_child(self) -> "_NameTable":
        table = _NameTable(self, self._id_provider)
        table._listeners = self._listeners
        return table

    def __getitem__(self, name: str) -> int:
        return self._table[name]

    def get_id(self, name: str) -> int | None:
        return self._table.get(name)

    def get_name(self, id: int) -> str | None:
        return self._table.inverse.get(id)

    def ids(self) -> Iterable[int]:
        return self._table.values()

    def table(self) -> bidict[str, int]:
        return self._table

    def add(self, name: str) -> int:
        if name in self._table:
            raise RuntimeError(f"Attempted to add a name that already exists: {name}")
        id = self._id_provider.next_id()
        self._table[name] = id
        for listener in self._listeners:
            listener.on_add(id, name)
        return id

    def modify(self, name: str) -> None:
        if name not in self._table:
            raise RuntimeError(
                f"Attempted to modify a name that doesn't exists: {name}"
            )
        for listener in self._listeners:
            listener.on_modify(self._table[name])

    def delete(self, name: str) -> int:
        if name not in self._table:
            raise RuntimeError(f"Attempted to delete a name that doesn't exist: {name}")
        id = self._table[name]
        del self._table[name]
        for listener in self._listeners:
            listener.on_delete(id)
        return id

    def rename(self, old_name: str, new_name: str) -> None:
        old_id = self.delete(old_name)
        new_id = self.add(new_name)
        for listener in self._listeners:
            listener.on_rename(old_id, new_id)

    def merge_into(self, new_table: "_NameTable") -> None:
        if self._table.keys() != new_table._table.keys():
            raise RuntimeError("Expected identical key sets")
        for key in sorted(self._table):
            old_id, new_id = self._table[key], new_table._table[key]
            if old_id == new_id:
                continue
            for listener in self._listeners:
                listener.on_merge(old_id, new_id)


class _FileEdgeRecorder(_NameTableListener):
    def __init__(self) -> None:
        self._id_to_name: dict[int, str] = dict()
        self._name_to_ids: dict[str, list[int]] = defaultdict(list)
        self._renames: set[tuple[int, int]] = set()
        self._merges: set[tuple[int, int]] = set()

    def on_add(self, id: int, name: str) -> None:
        self._id_to_name[id] = name
        self._name_to_ids[name].append(id)

    def on_modify(self, id: int) -> None:
        return

    def on_delete(self, id: int) -> None:
        return

    def on_rename(self, old_id: int, new_id: int) -> None:
        self._renames.add((min(old_id, new_id), max(old_id, new_id)))

    def on_merge(self, old_id: int, new_id: int) -> None:
        self._merges.add((min(old_id, new_id), max(old_id, new_id)))

    def names(self) -> dict[int, str]:
        return self._id_to_name

    def get_merge_edges(self) -> set[tuple[int, int]]:
        return self._merges

    def get_rename_edges(self) -> set[tuple[int, int]]:
        return self._renames

    def get_reuse_edges(self) -> set[tuple[int, int]]:
        edges: set[tuple[int, int]] = set()
        for ids in self._name_to_ids.values():
            edges.update(it.combinations(sorted(ids), r=2))
        return edges


class _DiffListener(abc.ABC):
    @abc.abstractmethod
    def on_enter_diff(self, commit: str, parent: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_add(self, name: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_delete(self, name: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_modify(self, name: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_rename(self, old_name: str, new_name: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_exit_diff(self) -> None:
        raise NotImplementedError


def _to_adj(edges: Iterable[tuple[int, int]]) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _to_trans_closure(edges: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    adj = _to_adj(edges)

    def visit(history: set[int], node: int) -> None:
        if node in history:
            return
        history.add(node)
        for other in adj[node]:
            visit(history, other)

    tc_edges: set[tuple[int, int]] = set()
    for node in adj:
        history: set[int] = set()
        visit(history, node)
        history.remove(node)
        tc_edges.update((min(u, v), max(u, v)) for u, v in it.product([node], history))
    return tc_edges


def _find_exclusive_cliques(edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    # Find cliques
    G = nx.Graph((u, v) for u, v in edges if u != v)
    cliques: list[list[int]] = [sorted(c) for c in nx.find_cliques(G)]  # type: ignore

    # Greedily filter to exclusive cliques using min ID as priority
    exclusive_cliques: list[list[int]] = []
    deleted_nodes: set[int] = set()
    for clique in sorted(cliques):
        if len(set(clique) & deleted_nodes) != 0:
            continue
        exclusive_cliques.append(clique)
        deleted_nodes.update(clique)
    return exclusive_cliques


def _print_edge_debug(edges: set[tuple[int, int]], cliques: list[list[int]]) -> None:
    used_edges = set(it.chain(*(it.combinations(c, r=2) for c in cliques)))
    unused_edges = edges - used_edges
    print(f"Total edges:  {len(edges)}")
    print(f"Used edges:   {len(used_edges)}")
    print(f"Unused edges: {len(unused_edges)}")


class _FileLookup:
    def __init__(self, tables: dict[str, bidict[str, int]]) -> None:
        self._tables = tables
        self._id_to_commits: dict[int, list[str]] = defaultdict(list)
        self._name_to_commits: dict[str, list[str]] = defaultdict(list)
        for commit, table in self._tables.items():
            for id in table.values():
                self._id_to_commits[id].append(commit)
            for name in table.keys():
                self._name_to_commits[name].append(commit)

    def file_table(self, commit: str) -> bidict[str, int]:
        return self._tables[commit]

    def file_id(self, commit: str, name: str) -> int:
        return self._tables[commit][name]

    def file_name(self, commit: str, id: int) -> str:
        return self._tables[commit].inv[id]

    def commits_by_id(self, id: int) -> list[str]:
        return self._id_to_commits[id]

    def commits_by_name(self, name: str) -> list[str]:
        return self._name_to_commits[name]


class _FileSolver(_DiffListener):
    def __init__(self) -> None:
        self._tables: dict[str, _NameTable] = dict()
        self._commits: dict[int, set[str]] = defaultdict(set)
        self._edge_recorder = _FileEdgeRecorder()

        self._curr_commit = _NULL_COMMIT_ID
        self._curr_table = _NameTable(None, _IdProvider())
        self._curr_table.add_listener(self._edge_recorder)
        self._tables[_NULL_COMMIT_ID] = self._curr_table

    def on_enter_diff(self, commit: str, parent: str) -> None:
        self._curr_commit = commit
        self._curr_table = self._tables[parent].create_child()

    def on_add(self, name: str) -> None:
        self._curr_table.add(name)

    def on_delete(self, name: str) -> None:
        self._curr_table.delete(name)

    def on_modify(self, name: str) -> None:
        self._curr_table.modify(name)

    def on_rename(self, old_name: str, new_name: str) -> None:
        self._curr_table.rename(old_name, new_name)

    def on_exit_diff(self) -> None:
        for id in self._curr_table.ids():
            self._commits[id].add(self._curr_commit)
        if self._curr_commit not in self._tables:
            self._tables[self._curr_commit] = self._curr_table
        else:
            self._curr_table.merge_into(self._tables[self._curr_commit])

    def solve_files(self) -> _FileLookup:
        walk_to_name = self._edge_recorder.names()
        walk_to_commits = self._commits
        merge_edges = self._edge_recorder.get_merge_edges()
        rename_edges = self._edge_recorder.get_rename_edges()
        reuse_edges = self._edge_recorder.get_reuse_edges()
        walk_to_file: dict[int, int] = dict()

        # Step 1: Solve using merge edges
        merge_adj = _to_adj(merge_edges)

        def visit_merge(walk: int, file: int):
            if walk in walk_to_file:
                return
            walk_to_file[walk] = file
            for other in merge_adj[walk]:
                visit_merge(other, file)

        for walk in walk_to_name:
            visit_merge(walk, walk)

        # Step 2: Solve using rename edges
        file_to_walks: dict[int, set[int]] = defaultdict(set)
        file_to_commits: dict[int, set[str]] = defaultdict(set)
        for walk, file in walk_to_file.items():
            file_to_walks[walk].add(file)
            file_to_commits[file].update(walk_to_commits[walk])

        rename_edges = _to_trans_closure(rename_edges)
        rename_edges = {(walk_to_file[u], walk_to_file[v]) for u, v in rename_edges}
        rename_edges = {
            (min(u, v), max(u, v))
            for u, v in rename_edges
            if len(file_to_commits[u] & file_to_commits[v]) == 0
        }
        cliques = _find_exclusive_cliques(rename_edges)
        _print_edge_debug(rename_edges, cliques)

        for clique in cliques:
            for file in clique:
                for walk in file_to_walks[file]:
                    walk_to_file[walk] = min(clique)

        # Step 3: Solve using reuse edges
        file_to_walks: dict[int, set[int]] = defaultdict(set)
        file_to_commits: dict[int, set[str]] = defaultdict(set)
        for walk, file in walk_to_file.items():
            file_to_walks[walk].add(file)
            file_to_commits[file].update(walk_to_commits[walk])

        reuse_edges = {(walk_to_file[u], walk_to_file[v]) for u, v in reuse_edges}
        reuse_edges = {
            (min(u, v), max(u, v))
            for u, v in reuse_edges
            if len(file_to_commits[u] & file_to_commits[v]) == 0
        }
        cliques = _find_exclusive_cliques(reuse_edges)
        _print_edge_debug(reuse_edges, cliques)

        for clique in cliques:
            for file in clique:
                for walk in file_to_walks[file]:
                    walk_to_file[walk] = min(clique)

        print("Done")

        # Step 4: Build final tables. This step is actually 3x more time-consuming
        # than everything else.
        tables: dict[str, bidict[str, int]] = dict()
        for commit, name_table in self._tables.items():
            table: dict[str, int] = dict()
            for name, walk in name_table.table().items():
                table[name] = walk_to_file[walk]
            tables[commit] = bidict(table)
        return _FileLookup(tables)


class _ChangeRecorder(_DiffListener):
    def __init__(self) -> None:
        self._curr_commit: str = _NULL_COMMIT_ID
        self._curr_names: set[str] = set()
        self._commit_to_names: dict[str, set[str]] = dict()

    def on_enter_diff(self, commit: str, parent: str) -> None:
        self._curr_commit = commit
        self._curr_names = set()

    def on_add(self, name: str) -> None:
        self._curr_names.add(name)

    def on_delete(self, name: str) -> None:
        pass

    def on_modify(self, name: str) -> None:
        self._curr_names.add(name)

    def on_rename(self, old_name: str, new_name: str) -> None:
        self._curr_names.add(new_name)

    def on_exit_diff(self) -> None:
        if self._curr_commit not in self._commit_to_names:
            self._commit_to_names[self._curr_commit] = self._curr_names
        else:
            self._commit_to_names[self._curr_commit] &= self._curr_names

    def to_changes_by_name(self) -> dict[str, list[str]]:
        name_to_commits: dict[str, list[str]] = defaultdict(list)
        for commit, names in self._commit_to_names.items():
            for name in names:
                name_to_commits[name].append(commit)
        return name_to_commits

    def to_changes_by_id(self, lookup: _FileLookup) -> dict[int, list[str]]:
        id_to_commits: dict[int, list[str]] = defaultdict(list)
        for commit, names in self._commit_to_names.items():
            file_table = lookup.file_table(commit)
            for name in names:
                id_to_commits[file_table[name]].append(commit)
        return id_to_commits


class _DiffPrinter(_DiffListener):
    def __init__(self) -> None:
        self._count = 0

    def on_enter_diff(self, commit: str, parent: str) -> None:
        self._count = 0
        print(f"Entering diff of {commit} from {parent}")

    def on_add(self, name: str) -> None:
        self._count += 1
        print(f"A\t{name}")

    def on_delete(self, name: str) -> None:
        self._count += 1
        print(f"D\t{name}")

    def on_modify(self, name: str) -> None:
        self._count += 1
        print(f"M\t{name}")

    def on_rename(self, old_name: str, new_name: str) -> None:
        self._count += 1
        print(f"R\t{old_name}\t{new_name}")

    def on_exit_diff(self) -> None:
        if self._count != 0:
            print("Exiting diff")
        else:
            print("Exiting EMPTY diff")
        print()


class _LogListener(abc.ABC):
    @abc.abstractmethod
    def on_enter_commit(
        self, commit: str, parents: list[str], from_parent: str
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_field(self, key: str, value: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_message(self, message: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_change(self, status: str, name_a: str, name_b: str | None) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_exit_commit(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_done(self) -> None:
        raise NotImplementedError


class _DiffNotifier(_LogListener):
    def __init__(self) -> None:
        self._listeners: list[_DiffListener] = []
        self._curr_commit = _NULL_COMMIT_ID
        self._curr_parents: list[str] = list()
        self._prev_index: int | None = None

    def add_listener(self, listener: _DiffListener) -> None:
        self._listeners.append(listener)

    def on_enter_commit(
        self, commit: str, parents: list[str], from_parent: str
    ) -> None:
        if from_parent not in parents:
            raise RuntimeError('The "from" parent is not listed as a parent')
        if len(parents) == 0:
            raise RuntimeError("Expected at least one parent")
        if commit != self._curr_commit:
            self._notify_empty_diffs()
            self._curr_commit = commit
            self._curr_parents = list(parents)
            self._prev_index = None
        curr_index = self._curr_parents.index(from_parent)
        self._notify_empty_diffs(curr_index)
        self._prev_index = curr_index + 1
        for listener in self._listeners:
            listener.on_enter_diff(commit, from_parent)

    def on_field(self, key: str, value: str) -> None:
        pass

    def on_message(self, message: str) -> None:
        pass

    def on_change(self, status: str, name_a: str, name_b: str | None) -> None:
        if status.startswith("A"):
            for listener in self._listeners:
                listener.on_add(name_a)
        elif status.startswith("D"):
            for listener in self._listeners:
                listener.on_delete(name_a)
        elif status.startswith("M"):
            for listener in self._listeners:
                listener.on_modify(name_a)
        elif status.startswith("R"):
            if name_b is None:
                raise RuntimeError("Expected second name for rename")
            for listener in self._listeners:
                listener.on_rename(name_a, name_b)
        else:
            raise RuntimeError(f"Unexpected status: {status}")

    def on_exit_commit(self) -> None:
        for listener in self._listeners:
            listener.on_exit_diff()

    def on_done(self) -> None:
        self._notify_empty_diffs()

    def _notify_empty_diffs(self, stop: int | None = None) -> None:
        for parent in self._curr_parents[self._prev_index : stop]:
            for listener in self._listeners:
                listener.on_enter_diff(self._curr_commit, parent)
            for listener in self._listeners:
                listener.on_exit_diff()


@dataclass
class CommitData:
    id: str
    parents: list[str]
    author_user: str
    author_email: str
    author_date: str
    commit_user: str
    commit_email: str
    commit_date: str
    message: list[str]


class _CommitDataRecorder(_LogListener):
    def __init__(self) -> None:
        self._curr_commit: dict[str, Any] | None = None
        self._commits: dict[str, dict[str, Any]] = dict()

    def on_enter_commit(
        self, commit: str, parents: list[str], from_parent: str
    ) -> None:
        if commit in self._commits:
            self._curr_commit = None
            return
        self._curr_commit = dict()
        self._commits[commit] = self._curr_commit
        self._curr_commit["id"] = commit
        self._curr_commit["parents"] = list(parents)

    def on_field(self, key: str, value: str) -> None:
        if self._curr_commit is None:
            return
        key = key.lower()
        if key == "author" or key == "commit":
            match = re.match("^(.*) <(.*)>$", value)
            self._curr_commit[f"{key}user"] = match.group(1)  # type: ignore
            self._curr_commit[f"{key}email"] = match.group(2)  # type: ignore
        elif key == "authordate" or key == "commitdate":
            self._curr_commit[key] = datetime.fromisoformat(value)
        else:
            self._curr_commit[key] = value

    def on_message(self, message: str) -> None:
        if self._curr_commit is None:
            return
        if "message" in self._curr_commit:
            self._curr_commit["message"].append(message)
        else:
            self._curr_commit["message"] = [message]

    def on_change(self, status: str, name_a: str, name_b: str | None) -> None:
        pass

    def on_exit_commit(self) -> None:
        pass

    def on_done(self) -> None:
        pass

    def to_dict(self) -> dict[str, CommitData]:
        return {
            key: CommitData(
                id=value["id"],
                parents=value["parents"],
                author_user=value["authoruser"],
                author_email=value["authoremail"],
                author_date=value["authordate"],
                commit_user=value["commituser"],
                commit_email=value["commitemail"],
                commit_date=value["commitdate"],
                message=value["message"],
            )
            for key, value in self._commits.items()
        }


class _LogParser:
    _reader: TextIOBase
    _listeners: list[_LogListener]
    _lineno: int
    _line: str

    def __init__(self, reader: TextIOBase):
        self._reader = reader
        self._listeners = []
        self._lineno = 0
        self._advance()

    def add_listener(self, listener: _LogListener) -> None:
        self._listeners.append(listener)

    def parse(self) -> None:
        while self._match_commit_line():
            self._advance()
            self._parse_field_lines()
            self._parse_message_lines()
            self._parse_change_lines()
            for listener in self._listeners:
                listener.on_exit_commit()
        if not self._is_eof():
            raise RuntimeError(f"Failed to parse line {self._lineno}")
        for listener in self._listeners:
            listener.on_done()

    def _parse_field_lines(self) -> None:
        while self._match_field_line():
            self._advance()

    def _parse_message_lines(self) -> None:
        while self._match_message_line():
            self._advance()

    def _parse_change_lines(self) -> None:
        while self._match_change_line():
            self._advance()

    def _match_commit_line(self) -> bool:
        pattern = r"^commit ((?:[0-9a-f]{40} ?)+)(?:\(from ([0-9a-f]{40})\))?"
        if (match := re.match(pattern, self._line)) is None:
            return False
        hashes: list[str] = match.group(1).strip().split()
        commit, parents = hashes[0], hashes[1:]
        if len(parents) == 0:
            parents.append(_NULL_COMMIT_ID)
        from_parent: str | None = match.group(2)
        if from_parent is None:
            if len(parents) < 2:
                from_parent = parents[0]
            else:
                raise RuntimeError('No "from" parent specified on merge commit')
        for listener in self._listeners:
            listener.on_enter_commit(commit, parents, from_parent)
        return True

    def _match_field_line(self) -> bool:
        pattern = r"^([a-zA-z]+): +(.+)"
        if (match := re.match(pattern, self._line)) is None:
            return False
        for listener in self._listeners:
            listener.on_field(match.group(1), match.group(2))
        return True

    def _match_message_line(self) -> bool:
        if not self._line.startswith("    "):
            return False
        for listener in self._listeners:
            listener.on_message(self._line.removeprefix("    "))
        return True

    def _match_change_line(self) -> bool:
        pattern = r"^([A-Z]\d{0,3})\t([^\t\v]+)\t?([^\t\v]+)?"
        if (match := re.match(pattern, self._line)) is None:
            return False
        for listener in self._listeners:
            listener.on_change(match.group(1), match.group(2), match.group(3))
        return True

    def _advance(self) -> None:
        while True:
            self._line = self._reader.readline()
            if self._line == "":
                break
            self._lineno += 1
            self._line = self._line.rstrip()
            if self._line != "":
                break

    def _is_eof(self) -> bool:
        return self._line == ""


def find_topo_order(
    parents: dict[str, list[str]], *, reverse_branches: bool = False
) -> list[str]:
    r"""Returns a topological order of nodes in a DAG. Intended for commits.

    Must have only a single leaf node. Each node must be in the keyset of
    parents even if it has no parents.

    Parents are always shown before children. For example, in a commit history
    like this:

    1---2----4----7---
     \           /
      3----5----6

    the resulting order will be: 1, 2, 4, 3, 5, 6, 7.

    In this example, 7 is a merge commit because it has more than one parent.
    The order of the parents of a merge commit is significant. Typically, the
    first parent is the "main" branch while the other(s) are "feature" branches.

    By default, the contents of the first branch ("main") are listed first. In
    the above example, this assumes 7 has its parents ordered as: 4, 6. But with
    reverse_branches=True, the last branch will be listed first. In this case,
    the order would be: 1, 3, 5, 6, 2, 4, 7.
    """
    non_leafs: set[str] = set(p for ps in parents.values() for p in ps)
    leafs: set[str] = parents.keys() - non_leafs
    if len(leafs) != 1:
        raise ValueError(f"Expected a single leaf, found {len(leafs)}")

    if reverse_branches:

        def iter_parents(node: str) -> Iterable[str]:
            return parents[node]
    else:

        def iter_parents(node: str) -> Iterable[str]:
            return reversed(parents[node])

    topo: list[str] = []
    perm: set[str] = set()
    temp: set[str] = set()
    stack: deque[str] = deque([next(iter(leafs))])

    while stack:
        node = stack[-1]
        if node in perm:
            stack.pop()
            continue
        if node in temp:
            temp.remove(node)
            perm.add(node)
            topo.append(node)
            stack.pop()
            continue
        temp.add(node)
        for parent in iter_parents(node):
            if parent in perm:
                continue
            if parent in temp:
                raise ValueError("Graph must be acyclic")
            stack.append(parent)

    return topo


def find_longest_path(
    parents: dict[str, list[str]],
    subset: set[str],
    *,
    topo_order: list[str] | None = None,
) -> list[str]:
    if topo_order is None:
        topo_order = find_topo_order(parents)

    predecessors: dict[str, str] = dict()
    values: dict[str, int] = dict()

    for node in topo_order:
        node_parents = parents[node]

        if len(node_parents) == 0:
            values[node] = 0
            continue

        parent_values = [values[p] for p in node_parents]
        max_value = max(parent_values)
        predecessors[node] = node_parents[parent_values.index(max_value)]
        values[node] = max_value + 1 if node in subset else 0

    path: list[str] = [topo_order[-1]]
    while (node := path[-1]) in predecessors:
        path.append(predecessors[node])

    return [n for n in reversed(path) if n in subset]


def _extract_log(repo_path: str, ref: str) -> str:
    args = [
        "git",
        "log",
        # Commit Ordering
        "--topo-order",
        "--reverse",
        # Commit Formatting
        "--format=fuller",
        "--date=iso-strict",
        "--parents",
        # Diff Formatting
        "--diff-merges=separate",
        "--diff-algorithm=histogram",
        "--name-status",
        "--find-renames",
        "-l0",
        ref,
    ]
    res = subprocess.run(args, cwd=repo_path, capture_output=True, text=True)
    return res.stdout


class NameRepo:
    def __init__(
        self,
        files: _FileLookup,
        commits: dict[str, CommitData],
        changes_by_id: dict[int, list[str]],
        changes_by_name: dict[str, list[str]],
    ) -> None:
        self._files = files
        self._commits = commits
        self._changes_by_id = changes_by_id
        self._changes_by_name = changes_by_name
        self._parents: dict[str, list[str]] = defaultdict(list)
        for commit in self._commits.values():
            self._parents[commit.id] = list(commit.parents)
        self._topo_order: list[str] = find_topo_order(self._parents)

    @staticmethod
    def parse_log(repo_path: str, ref: str) -> "NameRepo":
        name_recorder = _FileSolver()
        change_recorder = _ChangeRecorder()
        # diff_printer = DiffPrinter()

        diff_notifier = _DiffNotifier()
        diff_notifier.add_listener(name_recorder)
        diff_notifier.add_listener(change_recorder)
        # diff_notifier.add_listener(diff_printer)

        commit_data_recorder = _CommitDataRecorder()

        reader = StringIO(_extract_log(repo_path, ref))
        parser = _LogParser(reader)
        parser.add_listener(diff_notifier)
        parser.add_listener(commit_data_recorder)
        parser.parse()

        files = name_recorder.solve_files()
        commits = commit_data_recorder.to_dict()
        changes_by_id = change_recorder.to_changes_by_id(files)
        changes_by_name = change_recorder.to_changes_by_name()
        return NameRepo(files, commits, changes_by_id, changes_by_name)

    @staticmethod
    def load(pkl_path: str) -> "NameRepo":
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    def dump(self, pkl_path: str) -> None:
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)

    def commits(self) -> list[str]:
        return list(self._commits)

    def latest_commit(self) -> str:
        return list(self._commits)[-1]

    def commit_data(self, commit: str) -> CommitData:
        return self._commits[commit]

    def file_ids(self, commit: str) -> list[int]:
        return sorted(self._files.file_table(commit).values())

    def file_names(self, commit: str) -> list[str]:
        return sorted(self._files.file_table(commit).keys())

    def file_id_by_name(self, commit: str, file_name: str) -> int:
        return self._files.file_id(commit, file_name)

    def file_name_by_id(self, commit: str, file_id: int) -> str:
        return self._files.file_name(commit, file_id)

    def file_table(self, commit: str) -> bidict[str, int]:
        return self._files.file_table(commit)

    def commits_by_id(self, file_id: int) -> list[str]:
        return self._files.commits_by_id(file_id)

    def commits_by_name(self, file_name: str) -> list[str]:
        return self._files.commits_by_name(file_name)

    def changes_by_id(self, file_id: int) -> list[str]:
        return self._changes_by_id[file_id]
    #Commits that change by name
    def changes_by_name(self, file_name: str) -> list[str]:
        return self._changes_by_name[file_name]
    
    def cont_changes_by_id(self, file_id: int) -> list[str]:
        commits = set(self.changes_by_id(file_id))
        return find_longest_path(self._parents, commits, topo_order=self._topo_order)

    def cont_changes_by_name(self, file_name: str) -> list[str]:
        commits = set(self.changes_by_name(file_name))
        return find_longest_path(self._parents, commits, topo_order=self._topo_order)
