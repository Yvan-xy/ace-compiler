"""Immutable public contracts for the Python AIR pass framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Generic, Optional, Tuple, TypeVar, Union

from .diagnostics import Diagnostic

T = TypeVar("T")


class FrozenMap(Mapping[str, Any]):
    """A small, deterministic immutable mapping used in result records."""

    __slots__ = ("_items", "_mapping")

    def __init__(self, values: Optional[Mapping[str, Any]] = None):
        items = tuple(
            sorted(
                (
                    (str(key), _freeze_host_value(value))
                    for key, value in (values or {}).items()
                ),
                key=lambda item: item[0],
            )
        )
        if len({key for key, _ in items}) != len(items):
            raise ValueError("duplicate metric key")
        self._items = items
        self._mapping = MappingProxyType(dict(items))

    def __getitem__(self, key: str) -> Any:
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __repr__(self) -> str:
        return f"FrozenMap({dict(self._items)!r})"

    def __hash__(self) -> int:
        return hash(self._items)


def _freeze_host_value(value):
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, tuple):
        return tuple(_freeze_host_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_host_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze_host_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_host_value(item) for item in value)
    if value is None or isinstance(
        value, (bool, int, float, complex, str, bytes, Enum)
    ):
        return value
    if is_dataclass(value):
        params = getattr(type(value), "__dataclass_params__", None)
        if params and params.frozen and all(
            _is_deeply_immutable(getattr(value, field.name))
            for field in fields(value)
        ):
            return value
    raise TypeError(
        f"result mappings require immutable host values, got {type(value).__name__}"
    )


def _is_deeply_immutable(value) -> bool:
    if value is None or isinstance(
        value, (bool, int, float, complex, str, bytes, Enum)
    ):
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(_is_deeply_immutable(item) for item in value)
    if isinstance(value, FrozenMap):
        return all(_is_deeply_immutable(item) for item in value.values())
    if is_dataclass(value):
        params = getattr(type(value), "__dataclass_params__", None)
        return bool(params and params.frozen) and all(
            _is_deeply_immutable(getattr(value, field.name))
            for field in fields(value)
        )
    return False


def freeze_metrics(values: Optional[Mapping[str, Any]] = None) -> FrozenMap:
    return values if isinstance(values, FrozenMap) else FrozenMap(values)


@dataclass(frozen=True, order=True)
class IRUnitRef:
    kind: str
    module_id: object
    native_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in ("module", "function"):
            raise ValueError(f"unsupported IR unit kind: {self.kind}")
        if (self.kind == "module") != (self.native_id is None):
            raise ValueError(
                "module units have no native_id; function units require one"
            )


class UnitSelector(str, Enum):
    CURRENT_MODULE = "current-module"
    CURRENT_PASS_UNIT = "current-pass-unit"


CURRENT_MODULE = UnitSelector.CURRENT_MODULE
CURRENT_PASS_UNIT = UnitSelector.CURRENT_PASS_UNIT


def _configured_identity(instance: object) -> Tuple[Any, ...]:
    if is_dataclass(instance):
        return tuple(
            (field.name, _identity_value(getattr(instance, field.name)))
            for field in fields(instance)
        )
    values = getattr(instance, "__dict__", {})
    return tuple(
        sorted(
            (key, _identity_value(value))
            for key, value in values.items()
            if not key.startswith("_")
        )
    )


def _identity_value(value):
    air_view_type = globals().get("AIRView")
    module_view_type = globals().get("ModuleView")
    if (
        air_view_type is not None
        and isinstance(value, air_view_type)
        or module_view_type is not None
        and isinstance(value, module_view_type)
        or type(value).__module__.startswith("ace_bindings.")
    ):
        raise TypeError("pass configuration cannot retain live AIR objects")
    if isinstance(value, Mapping):
        return tuple(
            sorted((key, _identity_value(item)) for key, item in value.items())
        )
    if isinstance(value, (tuple, list)):
        return tuple(_identity_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_identity_value(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("pass configuration must be immutable and hashable") from exc
    return value


def pass_identity(instance: object) -> Tuple[Any, ...]:
    explicit = getattr(instance, "identity", None)
    if explicit is not None:
        return (
            instance.__class__.__module__,
            instance.__class__.__qualname__,
            _identity_value(explicit),
        )
    return (
        instance.__class__.__module__,
        instance.__class__.__qualname__,
        _configured_identity(instance),
    )


def pass_instance_id(instance: object) -> str:
    explicit = getattr(instance, "pass_id", None)
    if explicit:
        return str(explicit)
    return f"{instance.__class__.__module__}.{instance.__class__.__qualname__}"


class AnalysisPass(ABC, Generic[T]):
    """Immutable analysis configuration; implementations return host data."""

    cross_revision_safe = False

    def dependencies(self, unit: IRUnitRef) -> Tuple["AnalysisRequest", ...]:
        return ()

    @abstractmethod
    def run(self, read_only_unit, context) -> "AnalysisResult[T]":
        raise NotImplementedError


class TransformationPass(ABC):
    """Immutable transformation configuration run in one AIR transaction."""

    def required_analyses(self, current_unit: IRUnitRef) -> Tuple["AnalysisRequest", ...]:
        return ()

    @abstractmethod
    def run(self, edit_unit, context) -> "TransformResult":
        raise NotImplementedError


@dataclass(frozen=True)
class AnalysisRequest:
    analysis: AnalysisPass
    unit: Union[UnitSelector, IRUnitRef] = CURRENT_PASS_UNIT

    def __post_init__(self) -> None:
        if not isinstance(self.analysis, AnalysisPass):
            raise TypeError("analysis request requires an AnalysisPass")
        if not isinstance(self.unit, (UnitSelector, IRUnitRef)):
            raise TypeError("analysis request unit must be a selector or IRUnitRef")


@dataclass(frozen=True)
class AnalysisResult(Generic[T]):
    success: bool
    value: Optional[T] = None
    diagnostics: Tuple[Diagnostic, ...] = ()
    metrics: FrozenMap = FrozenMap()

    def __post_init__(self) -> None:
        if not self.success and self.value is not None:
            raise ValueError("failed analysis results cannot carry a value")
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "metrics", freeze_metrics(self.metrics))


@dataclass(frozen=True)
class PreservedAnalyses:
    identities: frozenset[Tuple[Any, ...]] = frozenset()

    @classmethod
    def none(cls) -> "PreservedAnalyses":
        return cls()

    @classmethod
    def of(cls, *analyses: AnalysisPass) -> "PreservedAnalyses":
        return cls(frozenset(pass_identity(analysis) for analysis in analyses))

    def preserves(self, analysis: AnalysisPass) -> bool:
        return pass_identity(analysis) in self.identities


@dataclass(frozen=True)
class TransformResult:
    success: bool
    diagnostics: Tuple[Diagnostic, ...] = ()
    metrics: FrozenMap = FrozenMap()
    preserved_analyses: PreservedAnalyses = PreservedAnalyses()

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "metrics", freeze_metrics(self.metrics))
        if not isinstance(self.preserved_analyses, PreservedAnalyses):
            raise TypeError("preserved_analyses must be PreservedAnalyses")


@dataclass(frozen=True)
class AnalysisExecutionResult:
    pass_id: str
    unit: IRUnitRef
    success: bool
    cached: bool
    diagnostics: Tuple[Diagnostic, ...] = ()
    metrics: FrozenMap = FrozenMap()
    analysis_result: Optional[AnalysisResult] = None
    before_revision: int = 0
    after_revision: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "metrics", freeze_metrics(self.metrics))
        if self.analysis_result is not None and not isinstance(
            self.analysis_result, AnalysisResult
        ):
            raise TypeError("analysis_result must be AnalysisResult")
        if self.before_revision != self.after_revision:
            raise ValueError("analysis execution cannot change the AIR revision")

    @property
    def changed(self) -> bool:
        return False


@dataclass(frozen=True)
class PassExecutionResult:
    pass_id: str
    unit: IRUnitRef
    success: bool
    changed: bool
    before_revision: int
    after_revision: int
    diagnostics: Tuple[Diagnostic, ...] = ()
    metrics: FrozenMap = FrozenMap()

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "metrics", freeze_metrics(self.metrics))


RequestedExecutionResult = Union[AnalysisExecutionResult, PassExecutionResult]


@dataclass(frozen=True)
class PassManagerResult:
    success: bool
    changed: bool
    executions: Tuple[RequestedExecutionResult, ...] = ()
    dependency_trace: Tuple[AnalysisExecutionResult, ...] = ()
    unexecuted_step_ids: Tuple[str, ...] = ()
    diagnostics: Tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "executions", tuple(self.executions))
        object.__setattr__(self, "dependency_trace", tuple(self.dependency_trace))
        object.__setattr__(self, "unexecuted_step_ids", tuple(self.unexecuted_step_ids))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def ordered_metrics(self) -> Tuple[Tuple[str, FrozenMap], ...]:
        return tuple((record.pass_id, record.metrics) for record in self.executions)


PassStep = Union[AnalysisRequest, TransformationPass]


_AIR_ID_KINDS = frozenset(
    {
        "module",
        "function",
        "entry",
        "formal",
        "local",
        "symbol",
        "preg",
        "iv",
        "block",
        "statement",
        "node",
        "type",
        "constant",
    }
)


@dataclass(frozen=True)
class AIRObjectId:
    """Stable host descriptor for an AIR object in one manager invocation."""

    kind: str
    module_id: object
    native_id: int
    function_id: Optional[int] = None
    revision: int = 0
    generation: int = 0

    def __post_init__(self) -> None:
        if self.kind not in _AIR_ID_KINDS:
            raise ValueError(f"unsupported AIR object ID kind: {self.kind}")
        if self.kind in {
            "formal",
            "local",
            "preg",
            "block",
            "statement",
            "node",
        } and self.function_id is None:
            raise ValueError(f"{self.kind} IDs require an owning function")
        if self.kind == "module" and self.function_id is not None:
            raise ValueError("module IDs cannot have an owning function")

    @property
    def backend_key(self) -> Tuple[str, Optional[int], int]:
        return self.kind, self.function_id, self.native_id


@dataclass(frozen=True)
class SourcePosition:
    file_id: int = 0
    line: int = 0
    column: int = 0
    count: int = 0
    statement_begin: bool = False
    basic_block_begin: bool = False


@dataclass(frozen=True)
class AIRAttribute:
    key: str
    element_type: str
    count: int
    payload: bytes


def _backend_value(target, name):
    value = getattr(target, name)
    return value() if callable(value) else value


def _normalize_key(key) -> Tuple[str, Optional[int], int]:
    kind, owner, native_id = tuple(key)
    return str(kind), None if owner is None or int(owner) < 0 else int(owner), int(native_id)


class _ViewState:
    __slots__ = (
        "glob_scope",
        "module_id",
        "transaction",
        "expected_generation",
        "snapshot",
        "records",
    )

    def __init__(self, glob_scope, module_id, transaction=None):
        self.glob_scope = glob_scope
        self.module_id = module_id
        self.transaction = transaction
        owner = transaction or glob_scope
        self.expected_generation = int(
            _backend_value(owner, "air_pass_generation")
        )
        self.snapshot = {}
        self.records = {}
        self.refresh()

    def check(self) -> None:
        owner = self.transaction or self.glob_scope
        if self.transaction is not None and not bool(
            _backend_value(self.transaction, "active")
        ):
            raise RuntimeError("stale AIR transaction view")
        if int(_backend_value(owner, "air_pass_generation")) != self.expected_generation:
            raise RuntimeError("stale AIR view generation")

    def refresh(self) -> None:
        self.check()
        owner = self.transaction or self.glob_scope
        snapshot = _backend_value(owner, "air_pass_snapshot")
        if not isinstance(snapshot, Mapping):
            raise TypeError("AIR pass snapshot must be a mapping")
        self.snapshot = dict(snapshot)
        self.records = {
            _normalize_key(record["key"]): dict(record)
            for record in self.snapshot.get("objects", ())
        }

    def record(self, stable_id: AIRObjectId):
        self.check()
        try:
            return self.records[stable_id.backend_key]
        except KeyError as exc:
            raise RuntimeError(f"AIR object is no longer present: {stable_id}") from exc

    def stable_id(self, key) -> AIRObjectId:
        kind, owner, native_id = _normalize_key(key)
        return AIRObjectId(
            kind,
            self.module_id,
            native_id,
            owner,
            int(self.snapshot.get("revision", 0)),
            self.expected_generation,
        )

    def view(self, key):
        stable_id = self.stable_id(key)
        view_type = _MUTABLE_VIEW_TYPES[stable_id.kind] if self.transaction else _VIEW_TYPES[stable_id.kind]
        return view_type(self, stable_id)


class AIRView:
    __slots__ = ("_state", "_id")

    def __init__(self, state: _ViewState, stable_id: AIRObjectId):
        self._state = state
        self._id = stable_id

    @property
    def id(self) -> AIRObjectId:
        self._state.record(self._id)
        return self._id

    @property
    def _record(self):
        return self._state.record(self._id)

    @property
    def native_id(self) -> int:
        self._state.record(self._id)
        return self._id.native_id

    @property
    def source_position(self) -> SourcePosition:
        raw = self._record.get("source_position", {})
        return SourcePosition(
            int(raw.get("file_id", 0)),
            int(raw.get("line", 0)),
            int(raw.get("column", 0)),
            int(raw.get("count", 0)),
            bool(raw.get("statement_begin", False)),
            bool(raw.get("basic_block_begin", False)),
        )

    @property
    def attributes(self) -> Tuple[AIRAttribute, ...]:
        return tuple(
            AIRAttribute(
                str(item["key"]),
                str(item["element_type"]),
                int(item["count"]),
                bytes(item["payload"]),
            )
            for item in self._record.get("attributes", ())
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, AIRView) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class FunctionView(AIRView):
    @property
    def unit_ref(self) -> IRUnitRef:
        stable_id = self.id
        return IRUnitRef("function", stable_id.module_id, stable_id.native_id)

    @property
    def name(self) -> str:
        return str(self._record.get("name", ""))

    @property
    def entries(self):
        return tuple(self._state.view(key) for key in self._record.get("entries", ()))

    @property
    def formals(self):
        return tuple(self._state.view(key) for key in self._record.get("formals", ()))

    @property
    def locals(self):
        return tuple(self._state.view(key) for key in self._record.get("locals", ()))

    @property
    def symbols(self):
        return tuple(self._state.view(key) for key in self._record.get("symbols", ()))

    @property
    def pregs(self):
        return tuple(self._state.view(key) for key in self._record.get("pregs", ()))

    @property
    def blocks(self):
        return tuple(self._state.view(key) for key in self._record.get("blocks", ()))

    @property
    def entry_block(self):
        key = self._record.get("entry_block")
        return None if key is None else self._state.view(key)

    @property
    def entry_statement(self):
        key = self._record.get("entry_statement")
        return None if key is None else self._state.view(key)

    @property
    def ivs(self):
        return tuple(self._state.view(key) for key in self._record.get("ivs", ()))

    def lookup(self, stable_id: AIRObjectId):
        if stable_id.module_id != self._id.module_id:
            raise ValueError("AIR ID belongs to another manager invocation")
        if stable_id.revision != self._id.revision or stable_id.generation != self._id.generation:
            raise RuntimeError("AIR ID belongs to a stale AIR revision or generation")
        if stable_id.function_id not in (None, self.native_id):
            raise ValueError("AIR ID belongs to another function")
        view = self._state.view(stable_id.backend_key)
        view._record
        return view


class EntryView(AIRView):
    @property
    def name(self) -> str:
        return str(self._record.get("name", ""))

    @property
    def owning_function(self):
        return self._state.view(self._record["owning_function"])

    @property
    def program_entry(self) -> bool:
        return bool(self._record.get("program_entry", False))

    @property
    def exported(self) -> bool:
        return bool(self._record.get("exported", False))


class DatumView(AIRView):
    @property
    def name(self) -> str:
        return str(self._record.get("name", ""))

    @property
    def type(self):
        key = self._record.get("type")
        return None if key is None else self._state.view(key)

    @property
    def address_taken(self) -> bool:
        return bool(self._record.get("address_taken", False))


class PregView(AIRView):
    @property
    def type(self):
        return self._state.view(self._record["type"])

    @property
    def home_symbol(self):
        key = self._record.get("home_symbol")
        return None if key is None else self._state.view(key)


class BlockView(AIRView):
    @property
    def statements(self):
        return tuple(self._state.view(key) for key in self._record.get("statements", ()))

    @property
    def parent_statement(self):
        key = self._record.get("parent_statement")
        return None if key is None else self._state.view(key)


class StatementView(AIRView):
    @property
    def node(self):
        return self._state.view(self._record["node"])

    @property
    def parent_block(self):
        key = self._record.get("parent_block")
        return None if key is None else self._state.view(key)


class NodeView(AIRView):
    @property
    def opcode(self) -> str:
        return str(self._record.get("opcode", ""))

    @property
    def result_type(self):
        key = self._record.get("result_type")
        return None if key is None else self._state.view(key)

    @property
    def children(self):
        return tuple(self._state.view(key) for key in self._record.get("children", ()))

    @property
    def child_blocks(self):
        return tuple(child for child in self.children if isinstance(child, BlockView))

    @property
    def parent_statement(self):
        key = self._record.get("parent_statement")
        return None if key is None else self._state.view(key)

    @property
    def call_target(self):
        key = self._record.get("call_target")
        return None if key is None else self._state.view(key)

    @property
    def arguments(self):
        return tuple(self._state.view(key) for key in self._record.get("arguments", ()))

    @property
    def result_preg(self):
        key = self._record.get("result_preg")
        return None if key is None else self._state.view(key)

    @property
    def preg(self):
        key = self._record.get("preg")
        return None if key is None else self._state.view(key)

    @property
    def return_value(self):
        key = self._record.get("return_value")
        return None if key is None else self._state.view(key)

    @property
    def symbol(self):
        key = self._record.get("symbol")
        return None if key is None else self._state.view(key)

    @property
    def iv(self):
        key = self._record.get("iv")
        return None if key is None else self._state.view(key)

    @property
    def constant(self):
        key = self._record.get("constant")
        return None if key is None else self._state.view(key)

    @property
    def indirect_call(self) -> bool:
        return bool(self._record.get("indirect_call", False))


class TypeView(AIRView):
    @property
    def name(self) -> str:
        return str(self._record.get("name", ""))

    @property
    def kind(self) -> str:
        return str(self._record.get("type_kind", ""))


class ConstantView(AIRView):
    @property
    def kind(self) -> str:
        return str(self._record.get("constant_kind", ""))

    @property
    def type(self):
        key = self._record.get("type")
        return None if key is None else self._state.view(key)

    @property
    def referenced_entry(self):
        key = self._record.get("referenced_entry")
        return None if key is None else self._state.view(key)


class ModuleView:
    """Read-only deterministic view of one committed AIR module revision."""

    __slots__ = ("_state", "_unit_ref")

    def __init__(self, glob_scope, module_id, *, transaction=None):
        if transaction is not None and type(self) is ModuleView:
            raise TypeError("read-only ModuleView cannot wrap a transaction")
        self._state = _ViewState(glob_scope, module_id, transaction)
        self._unit_ref = IRUnitRef("module", module_id)

    @property
    def unit_ref(self) -> IRUnitRef:
        self._state.check()
        return self._unit_ref

    @property
    def id(self) -> AIRObjectId:
        self._state.check()
        return self._state.stable_id(self._state.snapshot["module"])

    @property
    def revision(self) -> int:
        self._state.check()
        return int(self._state.snapshot.get("revision", 0))

    @property
    def functions(self):
        self._state.check()
        return tuple(
            self._state.view(key)
            for key in self._state.snapshot.get("functions", ())
        )

    @property
    def entries(self):
        self._state.check()
        return tuple(
            self._state.view(key) for key in self._state.snapshot.get("entries", ())
        )

    @property
    def types(self):
        self._state.check()
        return tuple(
            self._state.view(key) for key in self._state.snapshot.get("types", ())
        )

    @property
    def constants(self):
        self._state.check()
        return tuple(
            self._state.view(key)
            for key in self._state.snapshot.get("constants", ())
        )

    @property
    def symbols(self):
        self._state.check()
        return tuple(
            self._state.view(key)
            for key in self._state.snapshot.get("symbols", ())
        )

    def function(self, native_id: int):
        self._state.check()
        return self._state.view(("function", None, int(native_id)))

    def lookup(self, stable_id: AIRObjectId):
        self._state.check()
        if stable_id.module_id != self._unit_ref.module_id:
            raise ValueError("AIR ID belongs to another manager invocation")
        revision = int(self._state.snapshot.get("revision", 0))
        if stable_id.revision != revision or stable_id.generation != self._state.expected_generation:
            raise RuntimeError("AIR ID belongs to a stale AIR revision or generation")
        if stable_id.kind == "module":
            if stable_id.backend_key != _normalize_key(self._state.snapshot["module"]):
                raise ValueError("AIR module ID does not identify this module")
            return self
        view = self._state.view(stable_id.backend_key)
        view._record
        return view


class _MutableMixin:
    @property
    def _transaction(self):
        transaction = self._state.transaction
        if transaction is None:
            raise RuntimeError("AIR edit requires an active transaction")
        self._state.check()
        return transaction

    def _mutate(self, operation: str, *args):
        value = getattr(self._transaction, operation)(*args)
        self._state.refresh()
        return value

    def _require_view(self, view, expected_type, *kinds):
        if not isinstance(view, expected_type):
            expected_name = (
                " or ".join(item.__name__ for item in expected_type)
                if isinstance(expected_type, tuple)
                else expected_type.__name__
            )
            raise TypeError(
                f"AIR mutation requires {expected_name}, got {type(view).__name__}"
            )
        if view._state is not self._state:
            raise ValueError("AIR mutation operand belongs to another module or transaction")
        if kinds and view.id.kind not in kinds:
            raise TypeError(
                f"AIR mutation requires {', '.join(kinds)}, got {view.id.kind}"
            )
        view._record
        return view


class MutableFunctionView(_MutableMixin, FunctionView):
    def create_local(self, name: str, type_view: TypeView):
        self._require_view(type_view, TypeView, "type")
        key = self._mutate(
            "create_local", self.native_id, str(name), type_view.native_id
        )
        return self._state.view(key)

    def create_preg(self, type_view: TypeView):
        self._require_view(type_view, TypeView, "type")
        key = self._mutate("create_preg", self.native_id, type_view.native_id)
        return self._state.view(key)

    def clone_local(self, source: DatumView):
        self._require_view(source, DatumView, "local")
        key = self._mutate(
            "clone_local",
            self.native_id,
            source.id.function_id,
            source.native_id,
            False,
        )
        return self._state.view(key)

    def clone_iv(self, source: DatumView):
        self._require_view(source, DatumView, "iv")
        key = self._mutate(
            "clone_local",
            self.native_id,
            source.id.function_id,
            source.native_id,
            True,
        )
        return self._state.view(key)

    def clone_preg(self, source: PregView):
        self._require_view(source, PregView, "preg")
        key = self._mutate(
            "clone_preg",
            self.native_id,
            source.id.function_id,
            source.native_id,
        )
        return self._state.view(key)


class MutableEntryView(_MutableMixin, EntryView):
    pass


class MutableDatumView(_MutableMixin, DatumView):
    pass


class MutablePregView(_MutableMixin, PregView):
    pass


class MutableBlockView(_MutableMixin, BlockView):
    pass


class MutableStatementView(_MutableMixin, StatementView):
    def insert_before(self, template: "StatementView"):
        self._require_view(template, StatementView, "statement")
        key = self._mutate(
            "insert_statement", self.id.backend_key, template.id.backend_key, False
        )
        return self._state.view(key)

    def insert_after(self, template: "StatementView"):
        self._require_view(template, StatementView, "statement")
        key = self._mutate(
            "insert_statement", self.id.backend_key, template.id.backend_key, True
        )
        return self._state.view(key)

    def replace_with(self, template: "StatementView"):
        self._require_view(template, StatementView, "statement")
        key = self._mutate(
            "replace_statement", self.id.backend_key, template.id.backend_key
        )
        return self._state.view(key)

    def erase(self) -> None:
        self._mutate("erase_statement", self.id.backend_key)

    def _mapped_clone(
        self,
        source,
        after,
        *,
        formal_map=None,
        local_map=None,
        preg_map=None,
        iv_map=None,
    ):
        self._require_view(source, StatementView, "statement")

        def pairs(mapping, source_kind, destination_type, destination_kinds, *, key=False):
            result = []
            for original, replacement in (mapping or {}).items():
                self._require_view(original, AIRView, source_kind)
                self._require_view(
                    replacement, destination_type, *destination_kinds
                )
                result.append(
                    (
                        original.native_id,
                        replacement.id.backend_key if key else replacement.native_id,
                    )
                )
            return tuple(result)

        key = self._mutate(
            "clone_mapped_statement",
            self.id.backend_key,
            source.id.backend_key,
            bool(after),
            pairs(formal_map, "formal", NodeView, ("node",), key=True),
            pairs(local_map, "local", DatumView, ("local",)),
            pairs(preg_map, "preg", PregView, ("preg",)),
            pairs(iv_map, "iv", DatumView, ("local", "iv")),
        )
        return self._state.view(key)

    def clone_before(self, source: "StatementView", **maps):
        return self._mapped_clone(source, False, **maps)

    def clone_after(self, source: "StatementView", **maps):
        return self._mapped_clone(source, True, **maps)


class MutableNodeView(_MutableMixin, NodeView):
    def set_child(self, index: int, child: NodeView) -> None:
        self._require_view(child, (NodeView, BlockView), "node", "block")
        self._mutate(
            "set_node_child", self.id.backend_key, int(index), child.id.backend_key
        )

    def set_source_position(self, position: SourcePosition) -> None:
        self._mutate(
            "set_node_source_position",
            self.id.backend_key,
            position.file_id,
            position.line,
            position.column,
            position.count,
            position.statement_begin,
            position.basic_block_begin,
        )

    def set_attribute(self, attribute: AIRAttribute) -> None:
        self._mutate(
            "set_node_attribute",
            self.id.backend_key,
            attribute.key,
            attribute.element_type,
            attribute.count,
            attribute.payload,
        )

    def set_call_target(self, entry: EntryView) -> None:
        self._require_view(entry, EntryView, "entry")
        self._mutate("set_node_entry", self.id.backend_key, entry.native_id)

    def set_result_preg(self, preg: PregView) -> None:
        self._require_view(preg, PregView, "preg")
        self._mutate("set_node_result_preg", self.id.backend_key, preg.native_id)

    def set_symbol(self, datum: DatumView) -> None:
        self._require_view(datum, DatumView, "formal", "local", "symbol")
        self._mutate(
            "set_node_symbol", self.id.backend_key, datum.id.backend_key
        )

    def set_iv(self, datum: DatumView) -> None:
        self._require_view(datum, DatumView, "iv", "local")
        self._mutate("set_node_iv", self.id.backend_key, datum.id.backend_key)

    def set_result_type(self, type_view: TypeView) -> None:
        self._require_view(type_view, TypeView, "type")
        self._mutate("set_node_result_type", self.id.backend_key, type_view.native_id)


class MutableTypeView(_MutableMixin, TypeView):
    pass


class MutableConstantView(_MutableMixin, ConstantView):
    pass


class MutableModuleView(_MutableMixin, ModuleView):
    """Transaction-scoped module view; all edits mark the candidate dirty."""

    def remove_function(self, function: FunctionView) -> None:
        self._require_view(function, FunctionView, "function")
        self._mutate("remove_function", function.native_id)

    def verify(self) -> bool:
        return bool(self._transaction.verify())

    @property
    def source_to_candidate(self):
        raw = _backend_value(self._transaction, "source_to_candidate")
        source_revision = int(self._state.snapshot.get("revision", 0))
        source_generation = int(
            _backend_value(self._state.glob_scope, "air_pass_generation")
        )

        def source_id(key):
            kind, owner, native_id = _normalize_key(key)
            return AIRObjectId(
                kind,
                self._state.module_id,
                native_id,
                owner,
                source_revision,
                source_generation,
            )

        return MappingProxyType(
            {
                source_id(source): self._state.stable_id(candidate)
                for source, candidate in raw
            }
        )


_VIEW_TYPES = {
    "function": FunctionView,
    "entry": EntryView,
    "formal": DatumView,
    "local": DatumView,
    "symbol": DatumView,
    "iv": DatumView,
    "preg": PregView,
    "block": BlockView,
    "statement": StatementView,
    "node": NodeView,
    "type": TypeView,
    "constant": ConstantView,
}

_MUTABLE_VIEW_TYPES = {
    "function": MutableFunctionView,
    "entry": MutableEntryView,
    "formal": MutableDatumView,
    "local": MutableDatumView,
    "symbol": MutableDatumView,
    "iv": MutableDatumView,
    "preg": MutablePregView,
    "block": MutableBlockView,
    "statement": MutableStatementView,
    "node": MutableNodeView,
    "type": MutableTypeView,
    "constant": MutableConstantView,
}


def analysis_success(value: T, *, diagnostics=(), metrics=None) -> AnalysisResult[T]:
    return AnalysisResult(True, value, tuple(diagnostics), freeze_metrics(metrics))


def analysis_failure(*diagnostics: Diagnostic, metrics=None) -> AnalysisResult:
    return AnalysisResult(False, None, tuple(diagnostics), freeze_metrics(metrics))


def transform_success(*, diagnostics=(), metrics=None, preserved=None) -> TransformResult:
    return TransformResult(
        True,
        tuple(diagnostics),
        freeze_metrics(metrics),
        preserved or PreservedAnalyses.none(),
    )


def transform_failure(*diagnostics: Diagnostic, metrics=None) -> TransformResult:
    return TransformResult(False, tuple(diagnostics), freeze_metrics(metrics))
