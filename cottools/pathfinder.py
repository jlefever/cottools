import abc
import enum
import itertools as it
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable
from uuid import UUID, uuid4

import networkx as nx
from bidict import bidict

NULL_COMMIT_ID = "0" * 40


@dataclass
class CommitLine:
    id: str
    parent_ids: list[str]
    diff_parent_index: int

    def diff_parent_id(self) -> str:
        return self.parent_ids[self.diff_parent_index]

    @staticmethod
    def root(id: str) -> "CommitLine":
        return CommitLine(id, [NULL_COMMIT_ID], 0)

    @staticmethod
    def child(id: str, parent_ids: list[str], diff_parent_id: str) -> "CommitLine":
        if len(parent_ids) == 0:
            raise ValueError("Expected at least one parent")
        return CommitLine(id, parent_ids, parent_ids.index(diff_parent_id))


class Status(enum.Enum):
    ADD = "A"
    DELETE = "D"
    MODIFY = "M"
    RENAME = "R"

    @classmethod
    def from_string(cls, text: str) -> "Status":
        for status in cls:
            if text.startswith(status.value):
                return status
        raise ValueError(f"Expected A, D, M, or R (found {text})")


@dataclass
class NameLine:
    status: Status
    score: int | None
    old_name: str
    new_name: str


class NameTableListener(abc.ABC):
    @abc.abstractmethod
    def on_add(self, id: UUID, name: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_delete(self, id: UUID) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_rename(self, old_id: UUID, new_id: UUID) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_merge(self, old_id: UUID, new_id: UUID) -> None:
        raise NotImplementedError


class NameTable:
    _table: bidict[str, UUID]
    _listeners: list[NameTableListener]

    def __init__(self, parent: "NameTable | None" = None) -> None:
        if parent is None:
            self._table = bidict()
            self._listeners = []
        else:
            self._table = bidict(parent._table)
            self._listeners = parent._listeners

    def __getitem__(self, name: str) -> UUID:
        return self._table[name]

    def add_listener(self, listener: NameTableListener) -> None:
        self._listeners.append(listener)

    def get_id(self, name: str) -> UUID | None:
        return self._table.get(name)

    def get_name(self, id: UUID) -> str | None:
        return self._table.inverse.get(id)

    def ids(self) -> Iterable[UUID]:
        return self._table.values()

    def add(self, name: str) -> UUID:
        if name in self._table:
            raise RuntimeError(f"Attempted to add a name that already exists: {name}")
        id = uuid4()
        self._table[name] = id
        for listener in self._listeners:
            listener.on_add(id, name)
        return id

    def delete(self, name: str) -> UUID:
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

    def merge_into(self, new_table: "NameTable") -> None:
        if self._table.keys() != new_table._table.keys():
            raise RuntimeError("Expected identical key sets")
        for key in sorted(self._table):
            old_id, new_id = self._table[key], new_table._table[key]
            if old_id == new_id:
                continue
            for listener in self._listeners:
                listener.on_merge(old_id, new_id)

    def update(self, lines: list[NameLine]) -> None:
        for line in lines:
            match line.status:
                case Status.ADD:
                    self.add(line.new_name)
                case Status.DELETE:
                    self.delete(line.old_name)
                case Status.RENAME:
                    self.rename(line.old_name, line.new_name)
                case _:
                    pass


class RenameSolver(NameTableListener):
    def __init__(self) -> None:
        self._names: dict[UUID, str] = dict()
        self._walk_ids: dict[str, list[UUID]] = defaultdict(list)
        self._add_counts: Counter[str] = Counter()
        self._equivalencies: dict[UUID, list[UUID]] = defaultdict(list)

    def on_add(self, id: UUID, name: str) -> None:
        self._names[id] = name
        self._walk_ids[name].append(id)
        self._add_counts[name] += 1

    def on_delete(self, id: UUID) -> None:
        pass

    def on_rename(self, old_id: UUID, new_id: UUID) -> None:
        self._equivalencies[old_id].append(new_id)
        self._equivalencies[new_id].append(old_id)

    def on_merge(self, old_id: UUID, new_id: UUID) -> None:
        self._equivalencies[old_id].append(new_id)
        self._equivalencies[new_id].append(old_id)

    def names(self) -> dict[UUID, str]:
        return self._names

    def solve_renames(self) -> dict[UUID, UUID]:
        files: dict[UUID, UUID] = dict()

        def visit(walk_id: UUID, file_id: UUID):
            if walk_id in files:
                return
            files[walk_id] = file_id
            for other in self._equivalencies[walk_id]:
                visit(other, file_id)

        for walk_id in self._names.keys():
            visit(walk_id, uuid4())

        # Debug: Print name reuse
        for name, count in self._add_counts.items():
            if count < 2:
                continue
            n_files = len(set(files[w] for w in self._walk_ids[name]))
            if n_files < 2:
                continue
            print(f'Name "{name}" added {count} times: {n_files} files')

        return files


@dataclass
class Edge[T]:
    src: str
    tgt: str
    data: T


class Walker[T](abc.ABC):
    @abc.abstractmethod
    def visit_node(self, node: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_edge(self, edge: Edge[T]) -> None:
        raise NotImplementedError


class Graph[T]:
    _nodes: list[str]
    _edges: list[Edge[T]]
    _roots: list[str]
    _leafs: list[str]
    _incoming: dict[str, list[Edge[T]]]
    _outgoing: dict[str, list[Edge[T]]]

    def __init__(self, nodes: list[str], edges: list[Edge[T]]):
        self._nodes = nodes
        self._edges = edges
        self._roots = []
        self._leafs = []
        self._outgoing = defaultdict(list)
        self._incoming = defaultdict(list)
        for edge in edges:
            self._incoming[edge.tgt].append(edge)
            self._outgoing[edge.src].append(edge)
        for node in nodes:
            if len(self._incoming[node]) == 0:
                self._roots.append(node)
            if len(self._outgoing[node]) == 0:
                self._leafs.append(node)

    @property
    def nodes(self) -> list[str]:
        return self._nodes

    @property
    def edges(self) -> list[Edge[T]]:
        return self._edges

    @property
    def roots(self) -> list[str]:
        return self._roots

    @property
    def leafs(self) -> list[str]:
        return self._leafs

    def incoming(self, node: str) -> list[Edge[T]]:
        return self._incoming[node]

    def outgoing(self, node: str) -> list[Edge[T]]:
        return self._outgoing[node]

    def walk(self, walker: Walker[T]) -> None:
        visited: set[str] = set()
        for root in self._roots:
            walker.visit_node(root)
            visited.add(root)
            stack: list[Edge[T]] = list(reversed(self.outgoing(root)))
            while len(stack) != 0:
                edge = stack.pop()
                walker.visit_edge(edge)
                if edge.tgt in visited:
                    continue
                walker.visit_node(edge.tgt)
                visited.add(edge.tgt)
                stack.extend(reversed(self.outgoing(edge.tgt)))


class FileLookup:
    def __init__(self, tables: dict[str, NameTable], files: dict[UUID, UUID]):
        self._tables = tables
        self._files = files

    def get_file_id(self, commit: str, name: str) -> UUID | None:
        try:
            return self._files[self._tables[commit][name]]
        except KeyError:
            return None


def find_exclusive_cliques(edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    # Find cliques
    G = nx.Graph(edges)
    cliques: list[list[int]] = [sorted(c) for c in nx.find_cliques(G)] # type: ignore

    # Greedily filter to exclusive cliques using max size then min ID as priority
    exclusive_cliques: list[list[int]] = []
    deleted_nodes: set[int] = set()
    for clique in sorted(cliques, key=lambda c: [-1 * len(c)] + c):
        if len(set(clique) & deleted_nodes) != 0:
            continue
        exclusive_cliques.append(clique)
        deleted_nodes.update(clique)
    return exclusive_cliques


class CommitWalker(Walker[list[NameLine]]):
    def __init__(self) -> None:
        self._tables: dict[str, NameTable] = dict()
        self._commits: dict[UUID, list[str]] = defaultdict(list)
        self._solver = RenameSolver()

    def visit_edge(self, edge: Edge[list[NameLine]]) -> None:
        table = NameTable(self._tables[edge.src])
        table.update(edge.data)
        if edge.tgt in self._tables:
            table.merge_into(self._tables[edge.tgt])
        else:
            self._tables[edge.tgt] = table

    def visit_node(self, node: str) -> None:
        if node not in self._tables:
            table = NameTable()
            table.add_listener(self._solver)
            self._tables[node] = table
        for id in self._tables[node].ids():
            self._commits[id].append(node)

    def solve_files(self) -> FileLookup:
        walk_to_file = self._solver.solve_renames()
        walk_to_name = self._solver.names()
        walk_to_commits = self._commits

        file_to_walks: dict[UUID, list[UUID]] = defaultdict(list)
        file_to_names: dict[UUID, list[str]] = defaultdict(list)
        file_to_commits: dict[UUID, set[str]] = defaultdict(set)
        for walk, file in walk_to_file.items():
            file_to_walks[file].append(walk)
            file_to_names[file].append(walk_to_name[walk])
            file_to_commits[file].update(walk_to_commits[walk])

        name_to_files: dict[str, list[UUID]] = defaultdict(list)
        for walk, name in walk_to_name.items():
            name_to_files[name].append(walk_to_file[walk])

        # TODO: Use git topo-order
        file_to_ix: bidict[UUID, int] = bidict()
        for ix, file in enumerate(file_to_commits):
            file_to_ix[file] = ix

        edges: set[tuple[int, int]] = set()
        for file_a, commits_a in file_to_commits.items():
            for name in file_to_names[file_a]:
                for file_b in name_to_files[name]:
                    if file_a == file_b:
                        continue
                    if len(commits_a & file_to_commits[file_b]) != 0:
                        continue
                    a, b = file_to_ix[file_a], file_to_ix[file_b]
                    edges.add((min(a, b), max(a, b)))

        cliques = find_exclusive_cliques(edges)

        # Debug: Print unused edges
        used_edges = set(it.chain(*(it.combinations(c, r=2) for c in cliques)))
        unused_edges = edges - used_edges
        print(f"Total edges:  {len(edges)}")
        print(f"Used edges:   {len(used_edges)}")
        print(f"Unused edges: {len(unused_edges)}")

        walk_to_clique = dict(walk_to_file)
        for clique in cliques:
            clique_id = uuid4()
            for ix in clique:
                for walk in file_to_walks[file_to_ix.inv[ix]]:
                    walk_to_clique[walk] = clique_id

        return FileLookup(self._tables, walk_to_clique)


CommitGraph = Graph[list[NameLine]]


def build_commit_graph(lines: list[CommitLine | NameLine]) -> CommitGraph:
    nodes: list[str] = [NULL_COMMIT_ID]
    edges: dict[tuple[str, str], list[NameLine]] = dict()
    curr_edge: tuple[str, str] | None = None
    for line in lines:
        if isinstance(line, NameLine):
            edges[curr_edge].append(line)  # type: ignore
            continue
        curr_edge = (line.diff_parent_id(), line.id)
        if nodes[-1] == line.id:
            continue
        nodes.append(line.id)
        for parent_id in line.parent_ids:
            edges[(parent_id, line.id)] = []
    return Graph(nodes, [Edge(s, t, v) for (s, t), v in edges.items()])


def parse_log(log: str) -> list[CommitLine | NameLine]:
    lines: list[CommitLine | NameLine] = []
    for i, line in enumerate(log.splitlines()):
        try:
            lines.append(parse_line(line))
        except StopParsing:
            print(f"Warning: failed to parse line {i + 1}: {line}")
            continue
    return lines


def parse_line(line: str) -> CommitLine | NameLine:
    try:
        return parse_name_line(line)
    except StopParsing:
        return parse_commit_line(line)


def parse_name_line(line: str) -> NameLine:
    tokens = line.split("\t")
    if len(tokens) == 3:
        old_name = tokens[1]
        new_name = tokens[2]
    elif len(tokens) == 2:
        name = tokens[1]
        old_name, new_name = name, name
    else:
        raise StopParsing
    status, score = parse_status(tokens[0])
    if status == Status.ADD:
        old_name = ""
    elif status == Status.DELETE:
        new_name = ""
    return NameLine(status, score, old_name, new_name)


def parse_status(token: str) -> tuple[Status, int | None]:
    if token[0] not in "ADMR":
        raise StopParsing
    status = Status.from_string(token)
    score = None if len(token[1:]) == 0 else int(token[1:])
    return status, score


def parse_commit_line(line: str) -> CommitLine:
    tokens = line.split()
    id = parse_hash(tokens[0])
    parents: list[str] = []
    try:
        for token in tokens[1:]:
            parents.append(parse_hash(token))
    except StopParsing:
        pass
    if len(parents) == 0:
        return CommitLine.root(id)
    diff_parent_id = parents[0]
    if len(parents) > 1:
        if tokens[len(parents) + 1] == "(from":
            diff_parent_id = tokens[len(parents) + 2].removesuffix(")")
        else:
            raise RuntimeError("diff parent not specified")
    return CommitLine.child(id, parents, diff_parent_id)


def parse_hash(token: str) -> str:
    if len(token) != 40:
        raise StopParsing
    try:
        int(token, base=16)
        return token
    except ValueError:
        raise StopParsing


class StopParsing(Exception):
    """Custom exception to indicate that parsing should stop."""

    pass


def extract_log(repo_path: str) -> str:
    args = [
        "git",
        "log",
        
        # Commit Ordering
        "--date-order",
        "--reverse",

        # Commit Formatting
        "--format=oneline",
        "--parents",

        # Diff Formatting
        "--diff-merges=separate",
        "--diff-algorithm=histogram",
        "--name-status",
        "--find-renames",
        "-l0",
        "--diff-filter=ADMR",
    ]
    res = subprocess.run(args, cwd=repo_path, capture_output=True, text=True)
    return res.stdout
