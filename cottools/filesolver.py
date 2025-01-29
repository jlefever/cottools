import abc
import itertools as it
import re
from dataclasses import dataclass
from collections import Counter, defaultdict
from io import TextIOBase
from typing import Iterable, Any

from datetime import datetime

import networkx as nx
from bidict import bidict

NULL_COMMIT_ID = "0" * 40


class NameTableListener(abc.ABC):
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


class IdProvider:
    def __init__(self) -> None:
        self._id = 0

    def next_id(self) -> int:
        id = self._id
        self._id += 1
        return id


class NameTable:
    _table: bidict[str, int]
    _id_provider: IdProvider
    _listeners: list[NameTableListener]

    def __init__(self, parent: "NameTable | None", id_provider: IdProvider) -> None:
        if parent is None:
            self._table = bidict()
            self._id_provider = id_provider
            self._listeners = []
        else:
            self._table = bidict(parent._table)
            self._id_provider = id_provider
            self._listeners = parent._listeners

    def add_listener(self, listener: NameTableListener) -> None:
        self._listeners.append(listener)

    def create_child(self) -> "NameTable":
        table = NameTable(self, self._id_provider)
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
            # raise RuntimeError(f"Attempted to modify a name that doesn't exists: {name}")
            self.add(name)
        else:
            for listener in self._listeners:
                listener.on_modify(self._table[name])

    def delete(self, name: str) -> int:
        if name not in self._table:
            return -1
            # raise RuntimeError(f"Attempted to delete a name that doesn't exist: {name}")
        else:
            id = self._table[name]
            del self._table[name]
            for listener in self._listeners:
                listener.on_delete(id)
            return id

    def rename(self, old_name: str, new_name: str) -> None:
        old_id = self.delete(old_name)
        if old_id < 0:
            self.add(new_name)
        else:
            new_id = self.add(new_name)
            for listener in self._listeners:
                listener.on_rename(old_id, new_id)

    def merge_into(self, new_table: "NameTable") -> None:
        if self._table.keys() != new_table._table.keys():
            raise RuntimeError("Expected identical key sets")
        for key in sorted(self._table):
            old_id, new_id = self._table[key], new_table._table[key]
            if old_id == new_id:
                continue
            for listener in self._listeners:
                listener.on_merge(old_id, new_id)


class NameSolver(NameTableListener):
    def __init__(self) -> None:
        self._names: dict[int, str] = dict()
        self._walk_ids: dict[str, list[int]] = defaultdict(list)
        self._add_counts: Counter[str] = Counter()
        self._renames: dict[int, list[int]] = defaultdict(list)
        self._merges: dict[int, list[int]] = defaultdict(list)

    def on_add(self, id: int, name: str) -> None:
        self._names[id] = name
        self._walk_ids[name].append(id)
        self._add_counts[name] += 1

    def on_modify(self, id: int) -> None:
        return

    def on_delete(self, id: int) -> None:
        return

    def on_rename(self, old_id: int, new_id: int) -> None:
        self._renames[old_id].append(new_id)
        self._renames[new_id].append(old_id)

    def on_merge(self, old_id: int, new_id: int) -> None:
        self._merges[old_id].append(new_id)
        self._merges[new_id].append(old_id)

    def names(self) -> dict[int, str]:
        return self._names

    def solve_names(self, *, renames: bool) -> dict[int, int]:
        files: dict[int, int] = dict()

        def visit(walk_id: int, file_id: int):
            if walk_id in files:
                return
            files[walk_id] = file_id
            for other in self._merges[walk_id]:
                visit(other, file_id)
            if not renames:
                return
            for other in self._renames[walk_id]:
                visit(other, file_id)

        for file_id, walk_id in enumerate(self._names.keys()):
            visit(walk_id, file_id)

        # Debug: Print name reuse
        for name, count in self._add_counts.items():
            if count < 2:
                continue
            n_files = len(set(files[w] for w in self._walk_ids[name]))
            if n_files < 2:
                continue
            print(f"{name} added {count} times: {n_files} files")

        return files


class DiffListener(abc.ABC):
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


def find_exclusive_cliques(edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    # Find cliques
    G = nx.Graph(edges)
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


class FileLookup:
    def __init__(self, tables: dict[str, bidict[str, int]]) -> None:
        self._tables = tables

    def get_file_id(self, commit: str, name: str) -> int | None:
        try:
            return self._tables[commit][name]
        except KeyError:
            return None
    
    def get_file_name(self, commit: str, id: int) -> str | None:
        try:
            return self._tables[commit].inv[id]
        except KeyError:
            return None


class NameRecorder(DiffListener):
    def __init__(self) -> None:
        self._tables: dict[str, NameTable] = dict()
        self._commits: dict[int, list[str]] = defaultdict(list)
        self._solver = NameSolver()

        self._curr_commit = NULL_COMMIT_ID
        self._curr_table = NameTable(None, IdProvider())
        self._curr_table.add_listener(self._solver)
        self._tables[NULL_COMMIT_ID] = self._curr_table

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
            self._commits[id].append(self._curr_commit)
        if self._curr_commit not in self._tables:
            self._tables[self._curr_commit] = self._curr_table
        else:
            self._curr_table.merge_into(self._tables[self._curr_commit])

    def solve_files(self, *, renames: bool) -> FileLookup:
        walk_to_file = self._solver.solve_names(renames=renames)
        walk_to_name = self._solver.names()
        walk_to_commits = self._commits

        file_to_walks: dict[int, list[int]] = defaultdict(list)
        file_to_names: dict[int, list[str]] = defaultdict(list) # why not a set ?
        file_to_commits: dict[int, set[str]] = defaultdict(set)
        for walk, file in walk_to_file.items():
            file_to_walks[file].append(walk)
            file_to_names[file].append(walk_to_name[walk])
            file_to_commits[file].update(walk_to_commits[walk])

        name_to_files: dict[str, list[int]] = defaultdict(list)
        for walk, name in walk_to_name.items():
            name_to_files[name].append(walk_to_file[walk])

        edges: set[tuple[int, int]] = set()
        for file_a, commits_a in file_to_commits.items():
            for name in file_to_names[file_a]:
                for file_b in name_to_files[name]:
                    if file_a == file_b:
                        continue
                    if len(commits_a & file_to_commits[file_b]) != 0:
                        continue
                    edges.add((min(file_a, file_b), max(file_a, file_b)))

        cliques = find_exclusive_cliques(edges)

        # Debug: Print unused edges
        used_edges = set(it.chain(*(it.combinations(c, r=2) for c in cliques)))
        unused_edges = edges - used_edges
        print(f"Total edges:  {len(edges)}")
        print(f"Used edges:   {len(used_edges)}")
        print(f"Unused edges: {len(unused_edges)}")

        for clique in cliques:
            for file in clique:
                for walk in file_to_walks[file]:
                    walk_to_file[walk] = min(clique)

        tables: dict[str, bidict[str, int]] = dict()
        for commit, name_table in self._tables.items():
            table: bidict[str, int] = bidict(name_table.table())
            for name, walk in name_table.table().items():
                table[name] = walk_to_file[walk]
            tables[commit] = table

        return FileLookup(tables)


class ChangeRecorder(DiffListener):
    def __init__(self) -> None:
        self._curr_commit: str = NULL_COMMIT_ID
        self._curr_names: set[str] = set()
        self._names: dict[str, set[str]] = dict()

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
        if self._curr_commit not in self._names:
            self._names[self._curr_commit] = self._curr_names
        else:
            self._names[self._curr_commit] &= self._curr_names


class DiffPrinter(DiffListener):
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


class LogListener(abc.ABC):
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


class DiffNotifier(LogListener):
    def __init__(self) -> None:
        self._listeners: list[DiffListener] = []
        self._curr_commit = NULL_COMMIT_ID
        self._curr_parents: list[str] = list()
        self._prev_index: int | None = None

    def add_listener(self, listener: DiffListener) -> None:
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


class CommitDataRecorder(LogListener):
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = dict()
        self._commit: dict[str, Any] = dict()

    def on_enter_commit(
        self, commit: str, parents: list[str], from_parent: str
    ) -> None:
        self._commit = dict()
        self._data[commit] = self._commit
        self._commit["id"] = commit
        self._commit["parents"] = list(parents)

    def on_field(self, key: str, value: str) -> None:
        key = key.lower()
        if key == "author" or key == "commit":
            match = re.match("^(.+) <(.+)>$", value)
            self._commit[f"{key}user"] = match.group(1) # type: ignore
            self._commit[f"{key}email"] = match.group(2) # type: ignore
        elif key == "authordate" or key == "commitdate":
            self._commit[key] = datetime.fromisoformat(value)
        else:
            self._commit[key] = value

    def on_message(self, message: str) -> None:
        if "message" in self._commit:
            self._commit["message"].append(message)
        else:
            self._commit["message"] = [message]

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
            for key, value in self._data.items()
        }


class LogParser:
    _reader: TextIOBase
    _listeners: list[LogListener]
    _lineno: int
    _line: str

    def __init__(self, reader: TextIOBase):
        self._reader = reader
        self._listeners = []
        self._lineno = 0
        self._advance()

    def add_listener(self, listener: LogListener) -> None:
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
            parents.append(NULL_COMMIT_ID)
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
