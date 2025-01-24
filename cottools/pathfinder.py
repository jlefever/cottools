import abc
import enum
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from uuid import UUID, uuid4

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
    def on_add(self, walk_id: UUID, name: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_delete(self, walk_id: UUID) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_rename(self, old_walk_id: UUID, new_walk_id: UUID) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_merge(self, old_walk_id: UUID, new_walk_id: UUID) -> None:
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

    def get_walk_id(self, name: str) -> UUID | None:
        return self._table.get(name)

    def get_name(self, walk_id: UUID) -> str | None:
        return self._table.inverse.get(walk_id)

    def add(self, name: str) -> UUID:
        if name in self._table:
            raise RuntimeError(f"Attempted to add a name that already exists: {name}")
        walk_id = uuid4()
        self._table[name] = walk_id
        for listener in self._listeners:
            listener.on_add(walk_id, name)
        return walk_id

    def delete(self, name: str) -> UUID:
        if name not in self._table:
            raise RuntimeError(f"Attempted to delete a name that doesn't exist: {name}")
        walk_id = self._table[name]
        del self._table[name]
        for listener in self._listeners:
            listener.on_delete(walk_id)
        return walk_id

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


class FileSolver(NameTableListener):
    def __init__(self) -> None:
        self._names: dict[UUID, str] = dict()
        self._walk_ids: dict[str, list[UUID]] = defaultdict(list)
        self._add_counts: Counter[str] = Counter()
        self._equivalencies: dict[UUID, list[UUID]] = defaultdict(list)

    def on_add(self, walk_id: UUID, name: str) -> None:
        self._names[walk_id] = name
        self._walk_ids[name].append(walk_id)
        self._add_counts[name] += 1

    def on_delete(self, walk_id: UUID) -> None:
        pass

    def on_rename(self, old_walk_id: UUID, new_walk_id: UUID) -> None:
        self._equivalencies[old_walk_id].append(new_walk_id)
        self._equivalencies[new_walk_id].append(old_walk_id)

    def on_merge(self, old_walk_id: UUID, new_walk_id: UUID) -> None:
        self._equivalencies[old_walk_id].append(new_walk_id)
        self._equivalencies[new_walk_id].append(old_walk_id)

    def solve_files(self) -> dict[UUID, UUID]:
        files: dict[UUID, UUID] = dict()

        def visit(walk_id: UUID, file_id: UUID):
            if walk_id in files:
                return
            files[walk_id] = file_id
            for other in self._equivalencies[walk_id]:
                visit(other, file_id)

        for walk_id in self._names.keys():
            visit(walk_id, uuid4())

        # Debugging name reuse
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


class CommitWalker(Walker[list[NameLine]]):
    def __init__(self) -> None:
        self._tables: dict[str, NameTable] = dict()
        self._solver = FileSolver()

    def visit_edge(self, edge: Edge[list[NameLine]]) -> None:
        # print(f"Visiting EDGE: {edge.tgt} <- {edge.src}")
        table = NameTable(self._tables[edge.src])
        table.update(edge.data)
        if edge.tgt in self._tables:
            table.merge_into(self._tables[edge.tgt])
        else:
            self._tables[edge.tgt] = table

    def visit_node(self, node: str) -> None:
        # print(f"Visiting NODE: {node}")
        if node not in self._tables:
            table = NameTable()
            table.add_listener(self._solver)
            self._tables[node] = table

    def to_file_lookup(self) -> FileLookup:
        files = self._solver.solve_files()
        return FileLookup(self._tables, files)


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
            print(f"Warning: failed to parse line {i}: {line}")
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
        old_name = parse_name(tokens[1])
        new_name = parse_name(tokens[2])
    elif len(tokens) == 2:
        name = parse_name(tokens[1])
        old_name, new_name = name, name
    else:
        raise StopParsing
    status, score = parse_status(tokens[0])
    if status == Status.ADD:
        old_name = ""
    elif status == Status.DELETE:
        new_name = ""
    return NameLine(status, score, old_name, new_name)


def parse_name(token: str) -> str:
    if token.startswith('"'):
        # Skip paths with "unusual" characters
        # https://git-scm.com/docs/git-config#Documentation/git-config.txt-corequotePath
        # print("Warning: Skipping path with unusual character...")
        raise StopParsing
    return token


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
    try:
        diff_parent_id = tokens[tokens.index("(from") + 1].removesuffix(")")
    except ValueError:
        diff_parent_id = parents[0]
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
