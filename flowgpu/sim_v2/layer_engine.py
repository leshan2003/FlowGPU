"""Integrated fixed-window MoE layer execution for Simulator v2."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

from .detailed_network import DetailedFlowResult, FluidNetwork
from .dispatcher import AssignmentBinding, Dispatcher
from .multicast import build_multicast_tree
from .network_model import Flow
from .schema import Assignment, RuntimeCapabilities, ValidationError


@dataclass(frozen=True)
class ReplicaExecution:
    replica_id: str
    chip_id: str
    assignment_count: int
    dispatch_arrival_s: float
    compute_start_s: float
    compute_finish_s: float
    compute_queue_s: float
    kernel_warning: str


@dataclass(frozen=True)
class LayerCriticalPath:
    assignment_id: str
    replica_id: str
    dispatch_finish_s: float
    compute_finish_s: float
    combine_finish_s: float


@dataclass(frozen=True)
class MoELayerResult:
    layer_id: int
    completion_s: float
    assignment_count: int
    replica_executions: Tuple[ReplicaExecution, ...]
    flow_results: Mapping[str, DetailedFlowResult]
    link_bytes: Mapping[str, int]
    network_energy_j: float
    network_contention_s: float
    local_assignment_fraction: float
    dispatch_flow_count: int
    combine_flow_count: int
    logical_dispatch_bytes: int
    physical_dispatch_bytes: int
    logical_combine_bytes: int
    physical_combine_bytes: int
    critical_path: LayerCriticalPath
    warnings: Tuple[str, ...]


class MoELayerEngine:
    """Execute one fixed scheduling window with network/compute overlap.

    Assignments are first bound to static replicas by ``Dispatcher``.  Dispatch
    flows then compete in the detailed fluid network.  A replica-local grouped
    expert kernel becomes ready after every assignment bound to that replica
    arrives.  Each chip executes one expert batch at a time.  Compute
    completions release combine flows back into the same live network, so
    dispatch and combine traffic can contend.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        capabilities: RuntimeCapabilities = RuntimeCapabilities(),
        routing_metric: str = "latency",
        chip_reduce_latency_s: float = 0.0,
        node_reduce_latency_s: float = 0.0,
    ) -> None:
        if (capabilities.hardware_multicast and
                not capabilities.dispatch_dedup_node):
            raise ValidationError(
                "hardware_multicast requires dispatch_dedup_node"
            )
        self.dispatcher = dispatcher
        self.capabilities = capabilities
        self.routing_metric = routing_metric
        if chip_reduce_latency_s < 0 or not math.isfinite(
                chip_reduce_latency_s):
            raise ValidationError(
                "chip_reduce_latency_s must be finite and non-negative"
            )
        if (capabilities.combine_reduce_chip and
                chip_reduce_latency_s <= 0):
            raise ValidationError(
                "combine_reduce_chip requires calibrated positive "
                "chip_reduce_latency_s"
            )
        self.chip_reduce_latency_s = chip_reduce_latency_s
        if node_reduce_latency_s < 0 or not math.isfinite(
                node_reduce_latency_s):
            raise ValidationError(
                "node_reduce_latency_s must be finite and non-negative"
            )
        if (capabilities.combine_reduce_node and
                node_reduce_latency_s <= 0):
            raise ValidationError(
                "combine_reduce_node requires calibrated positive "
                "node_reduce_latency_s"
            )
        self.node_reduce_latency_s = node_reduce_latency_s

    @staticmethod
    def _token_key(assignment: Assignment) -> Tuple[int, int, int, int]:
        return (
            assignment.request_id,
            assignment.token_id,
            assignment.step,
            assignment.layer_id,
        )

    def _dispatch_flow(
        self, binding: AssignmentBinding, earliest_start_s: float = 0.0
    ) -> Flow:
        assignment = binding.assignment
        chip_endpoint = self.dispatcher.chip_endpoints[
            binding.replica.chip_id
        ]
        home_endpoint = self.dispatcher.home_endpoints[
            assignment.home_node
        ]
        if (self.dispatcher.ingress_source_endpoint is not None and
                chip_endpoint ==
                self.dispatcher.ingress_source_endpoint):
            source = chip_endpoint
            waypoints = tuple()
        elif self.dispatcher.ingress_source_endpoint is None:
            source = home_endpoint
            waypoints = tuple()
        else:
            source = self.dispatcher.ingress_source_endpoint
            waypoints = (home_endpoint,)
        return Flow(
            flow_id="dispatch:" + assignment.assignment_id,
            src=source,
            dst=chip_endpoint,
            nbytes=self.dispatcher.hidden_bytes,
            earliest_start_s=earliest_start_s,
            traffic_class="dispatch",
            waypoints=waypoints,
        )

    def _combine_flow(
        self,
        binding: AssignmentBinding,
        earliest_start_s: float,
    ) -> Flow:
        assignment = binding.assignment
        chip_endpoint = self.dispatcher.chip_endpoints[
            binding.replica.chip_id
        ]
        home_endpoint = self.dispatcher.home_endpoints[
            assignment.home_node
        ]
        if (self.dispatcher.ingress_source_endpoint is not None and
                chip_endpoint ==
                self.dispatcher.ingress_source_endpoint):
            destination = chip_endpoint
            waypoints = tuple()
        elif self.dispatcher.ingress_source_endpoint is None:
            destination = home_endpoint
            waypoints = tuple()
        else:
            destination = self.dispatcher.ingress_source_endpoint
            waypoints = (home_endpoint,)
        return Flow(
            flow_id="combine:" + assignment.assignment_id,
            src=chip_endpoint,
            dst=destination,
            nbytes=self.dispatcher.hidden_bytes,
            earliest_start_s=earliest_start_s,
            traffic_class="combine",
            waypoints=waypoints,
        )

    def run(
        self,
        assignments: Iterable[Assignment],
        replica_ready_s: Mapping[str, float] = None,
        start_time_s: float = 0.0,
    ) -> MoELayerResult:
        items = tuple(assignments)
        if not items:
            raise ValidationError("MoE layer window must contain assignments")
        if not math.isfinite(start_time_s) or start_time_s < 0:
            raise ValidationError(
                "layer start_time_s must be finite and non-negative"
            )
        layers = {assignment.layer_id for assignment in items}
        if len(layers) != 1:
            raise ValidationError(
                "MoELayerEngine.run accepts exactly one layer"
            )
        layer_id = next(iter(layers))
        token_homes = {}
        for assignment in items:
            token = self._token_key(assignment)
            previous = token_homes.setdefault(token, assignment.home_node)
            if previous != assignment.home_node:
                raise ValidationError(
                    "one token-layer cannot have multiple home nodes"
                )
        dispatch_plan = self.dispatcher.plan(items, replica_ready_s or {})
        if dispatch_plan.assignment_count != len(items):
            raise RuntimeError("dispatcher assignment conservation failed")

        binding_by_assignment = {
            binding.assignment.assignment_id: binding
            for binding in dispatch_plan.bindings
        }
        bindings_by_replica: Dict[str, list] = {}
        for binding in dispatch_plan.bindings:
            bindings_by_replica.setdefault(
                binding.replica.replica_id, []
            ).append(binding)

        network = FluidNetwork(
            self.dispatcher.topology, self.routing_metric
        )
        flow_kind = {}
        dispatch_node_chip_groups = {}
        dispatch_node_indices = {}
        multicast_states = {}

        def launch_dispatch_fanout(group_key, release_s):
            node_index = dispatch_node_indices[group_key]
            destination_node = group_key[1]
            source = self.dispatcher.home_endpoints[destination_node]
            by_chip = dispatch_node_chip_groups[group_key]
            for chip_index, chip_id in enumerate(sorted(by_chip)):
                group = by_chip[chip_id]
                flow = Flow(
                    flow_id="dispatch:node%d:chip%d" %
                    (node_index, chip_index),
                    src=source,
                    dst=self.dispatcher.chip_endpoints[chip_id],
                    nbytes=self.dispatcher.hidden_bytes,
                    earliest_start_s=release_s,
                    traffic_class="dispatch",
                )
                flow_kind[flow.flow_id] = (
                    "dispatch",
                    tuple(
                        binding.assignment.assignment_id
                        for binding in group
                    ),
                )
                network.add_flow(flow)

        def launch_multicast_segment(token_key, link_id, release_s):
            state = multicast_states[token_key]
            if link_id in state["launched"]:
                return
            link = state["links"][link_id]
            flow = Flow(
                flow_id="dispatch:mc%d:%s" %
                (state["index"], link_id),
                src=link.src,
                dst=link.dst,
                nbytes=self.dispatcher.hidden_bytes,
                earliest_start_s=release_s,
                traffic_class="dispatch",
                route_link_ids=(link_id,),
            )
            flow_kind[flow.flow_id] = (
                "dispatch_multicast_segment",
                (token_key, link_id),
            )
            network.add_flow(flow)
            state["launched"].add(link_id)

        def start_multicast(token_key, release_s):
            state = multicast_states[token_key]
            source = state["tree"].source
            if source in state["destination_keys"]:
                launch_dispatch_fanout(
                    state["destination_keys"][source], release_s
                )
            for link_id in state["tree"].root_link_ids:
                launch_multicast_segment(token_key, link_id, release_s)

        if self.capabilities.dispatch_dedup_node:
            dispatch_node_groups: Dict[tuple, list] = {}
            for binding in dispatch_plan.bindings:
                destination_node = self.dispatcher.replica_plan.chip(
                    binding.replica.chip_id
                ).node_id
                key = (
                    self._token_key(binding.assignment),
                    destination_node,
                )
                dispatch_node_groups.setdefault(key, []).append(binding)
            ordered_node_keys = sorted(dispatch_node_groups, key=str)
            for node_index, key in enumerate(ordered_node_keys):
                dispatch_node_indices[key] = node_index
                group = dispatch_node_groups[key]
                by_chip = {}
                for binding in group:
                    by_chip.setdefault(
                        binding.replica.chip_id, []
                    ).append(binding)
                dispatch_node_chip_groups[key] = by_chip
            if self.capabilities.hardware_multicast:
                keys_by_token = {}
                for key in ordered_node_keys:
                    keys_by_token.setdefault(key[0], []).append(key)
                for multicast_index, token_key in enumerate(sorted(
                        keys_by_token, key=str)):
                    keys = keys_by_token[token_key]
                    assignment = dispatch_node_groups[keys[0]][0].assignment
                    home = self.dispatcher.home_endpoints[
                        assignment.home_node
                    ]
                    destination_keys = {
                        self.dispatcher.home_endpoints[key[1]]: key
                        for key in keys
                    }
                    tree = build_multicast_tree(
                        self.dispatcher.topology,
                        home,
                        destination_keys,
                        self.routing_metric,
                    )
                    multicast_states[token_key] = {
                        "index": multicast_index,
                        "tree": tree,
                        "links": {
                            link.link_id: link for link in tree.links
                        },
                        "destination_keys": destination_keys,
                        "launched": set(),
                    }
                    if self.dispatcher.ingress_source_endpoint is None:
                        start_multicast(token_key, start_time_s)
                    else:
                        flow = Flow(
                            flow_id="dispatch:mc%d:ingress" %
                            multicast_index,
                            src=self.dispatcher.ingress_source_endpoint,
                            dst=home,
                            nbytes=self.dispatcher.hidden_bytes,
                            earliest_start_s=start_time_s,
                            traffic_class="dispatch",
                        )
                        flow_kind[flow.flow_id] = (
                            "dispatch_multicast_ingress", token_key
                        )
                        network.add_flow(flow)
            else:
                for key in ordered_node_keys:
                    node_index = dispatch_node_indices[key]
                    group = dispatch_node_groups[key]
                    assignment = group[0].assignment
                    home = self.dispatcher.home_endpoints[
                        assignment.home_node
                    ]
                    destination = self.dispatcher.home_endpoints[key[1]]
                    if self.dispatcher.ingress_source_endpoint is None:
                        source = home
                        waypoints = tuple()
                    else:
                        source = self.dispatcher.ingress_source_endpoint
                        waypoints = (home,)
                    nbytes = (
                        0 if source == destination and not waypoints
                        else self.dispatcher.hidden_bytes
                    )
                    flow = Flow(
                        flow_id="dispatch:node%d" % node_index,
                        src=source,
                        dst=destination,
                        nbytes=nbytes,
                        earliest_start_s=start_time_s,
                        traffic_class="dispatch",
                        waypoints=waypoints,
                    )
                    flow_kind[flow.flow_id] = ("dispatch_node", key)
                    network.add_flow(flow)
        elif self.capabilities.dispatch_dedup_chip:
            dispatch_groups: Dict[tuple, list] = {}
            for binding in dispatch_plan.bindings:
                key = (
                    self._token_key(binding.assignment),
                    binding.replica.chip_id,
                )
                dispatch_groups.setdefault(key, []).append(binding)
        else:
            dispatch_groups = {
                (binding.assignment.assignment_id,): [binding]
                for binding in dispatch_plan.bindings
            }
        if not self.capabilities.dispatch_dedup_node:
            for index, key in enumerate(sorted(dispatch_groups, key=str)):
                group = dispatch_groups[key]
                base = self._dispatch_flow(group[0], start_time_s)
                flow = Flow(
                    flow_id="dispatch:g%d" % index,
                    src=base.src,
                    dst=base.dst,
                    nbytes=base.nbytes,
                    earliest_start_s=base.earliest_start_s,
                    traffic_class=base.traffic_class,
                    waypoints=base.waypoints,
                )
                flow_kind[flow.flow_id] = (
                    "dispatch",
                    tuple(item.assignment.assignment_id for item in group),
                )
                network.add_flow(flow)

        arrived: Dict[str, Dict[str, float]] = {
            replica_id: {} for replica_id in bindings_by_replica
        }
        ready_batches: Dict[str, list] = {}
        running_chips = set()
        compute_heap = []
        executions: Dict[str, ReplicaExecution] = {}
        combine_finished: Dict[str, float] = {}
        seen_network_results = set()
        warnings = set()
        combine_groups: Dict[tuple, list] = {}
        if self.capabilities.combine_reduce_chip:
            for binding in dispatch_plan.bindings:
                key = (
                    self._token_key(binding.assignment),
                    binding.replica.chip_id,
                )
                combine_groups.setdefault(key, []).append(binding)
        combine_ready = {
            key: {} for key in combine_groups
        }
        launched_combine_groups = set()
        node_combine_groups: Dict[tuple, list] = {}
        node_input_expected = {}
        node_input_finished: Dict[tuple, list] = {}
        launched_node_combines = set()
        if self.capabilities.combine_reduce_node:
            for binding in dispatch_plan.bindings:
                source_node = self.dispatcher.replica_plan.chip(
                    binding.replica.chip_id
                ).node_id
                key = (
                    self._token_key(binding.assignment),
                    source_node,
                )
                node_combine_groups.setdefault(key, []).append(binding)
            for key, group in node_combine_groups.items():
                if self.capabilities.combine_reduce_chip:
                    node_input_expected[key] = len({
                        binding.replica.chip_id for binding in group
                    })
                else:
                    node_input_expected[key] = len(group)
                node_input_finished[key] = []

        def launch_node_final(group_key, release_s):
            group = node_combine_groups[group_key]
            assignment = group[0].assignment
            source_node = group_key[1]
            source = self.dispatcher.home_endpoints[source_node]
            home = self.dispatcher.home_endpoints[assignment.home_node]
            if self.dispatcher.ingress_source_endpoint is None:
                destination = home
                waypoints = tuple()
            else:
                destination = self.dispatcher.ingress_source_endpoint
                waypoints = (home,)
            nbytes = (
                0 if source == destination and not waypoints
                else self.dispatcher.hidden_bytes
            )
            index = sorted(node_combine_groups, key=str).index(group_key)
            flow = Flow(
                flow_id="combine:node%d:final" % index,
                src=source,
                dst=destination,
                nbytes=nbytes,
                earliest_start_s=release_s + self.node_reduce_latency_s,
                traffic_class="combine",
                waypoints=waypoints,
            )
            flow_kind[flow.flow_id] = (
                "combine",
                tuple(
                    binding.assignment.assignment_id for binding in group
                ),
            )
            network.add_flow(flow)
            launched_node_combines.add(group_key)

        def accept_network_results(results):
            for result in results:
                flow_id = result.flow.flow_id
                if flow_id in seen_network_results:
                    continue
                seen_network_results.add(flow_id)
                kind, payload = flow_kind[flow_id]
                if kind == "dispatch_multicast_ingress":
                    start_multicast(payload, result.finish_s)
                    continue
                if kind == "dispatch_multicast_segment":
                    token_key, link_id = payload
                    state = multicast_states[token_key]
                    link = state["links"][link_id]
                    destination_key = state[
                        "destination_keys"
                    ].get(link.dst)
                    if destination_key is not None:
                        launch_dispatch_fanout(
                            destination_key, result.finish_s
                        )
                    for child in state["tree"].children[link_id]:
                        launch_multicast_segment(
                            token_key, child, result.finish_s
                        )
                    continue
                if kind == "dispatch_node":
                    launch_dispatch_fanout(payload, result.finish_s)
                    continue
                if kind == "combine_node_input":
                    node_input_finished[payload].append(result.finish_s)
                    if (len(node_input_finished[payload]) ==
                            node_input_expected[payload] and
                            payload not in launched_node_combines):
                        launch_node_final(
                            payload,
                            max(node_input_finished[payload]),
                        )
                    continue
                assignment_ids = payload
                if kind == "dispatch":
                    touched_replicas = set()
                    for assignment_id in assignment_ids:
                        binding = binding_by_assignment[assignment_id]
                        replica_id = binding.replica.replica_id
                        arrived[replica_id][assignment_id] = result.finish_s
                        touched_replicas.add(replica_id)
                    for replica_id in sorted(touched_replicas):
                        expected = len(bindings_by_replica[replica_id])
                        if len(arrived[replica_id]) == expected:
                            arrival = max(arrived[replica_id].values())
                            binding = bindings_by_replica[replica_id][0]
                            chip_id = binding.replica.chip_id
                            ready_batches.setdefault(chip_id, []).append(
                                (arrival, replica_id)
                            )
                            ready_batches[chip_id].sort()
                else:
                    for assignment_id in assignment_ids:
                        combine_finished[assignment_id] = result.finish_s

        accept_network_results(network.results.values())

        def schedule_ready(now_s):
            for chip_id in sorted(ready_batches):
                if chip_id in running_chips or not ready_batches[chip_id]:
                    continue
                arrival, replica_id = ready_batches[chip_id][0]
                if arrival > now_s + 1e-15:
                    continue
                ready_batches[chip_id].pop(0)
                bindings = bindings_by_replica[replica_id]
                expert = bindings[0].replica.expert
                model = self.dispatcher.kernel_model(
                    expert, bindings[0].replica
                )
                prediction = model.predict(len(bindings))
                start = max(now_s, arrival)
                finish = start + prediction.latency_s
                execution = ReplicaExecution(
                    replica_id=replica_id,
                    chip_id=chip_id,
                    assignment_count=len(bindings),
                    dispatch_arrival_s=arrival,
                    compute_start_s=start,
                    compute_finish_s=finish,
                    compute_queue_s=start - arrival,
                    kernel_warning=prediction.warning or "",
                )
                executions[replica_id] = execution
                if prediction.warning:
                    warnings.add(prediction.warning)
                running_chips.add(chip_id)
                heapq.heappush(
                    compute_heap, (finish, replica_id, chip_id)
                )

        schedule_ready(network.now_s)
        while len(combine_finished) < len(items):
            next_network = (
                network.next_event_time()
                if network.unfinished_count else math.inf
            )
            next_compute = (
                compute_heap[0][0] if compute_heap else math.inf
            )
            event_time = min(next_network, next_compute)
            if not math.isfinite(event_time):
                raise RuntimeError(
                    "MoE layer deadlock before all combines completed"
                )
            completed_network = network.advance_to(event_time)
            accept_network_results(completed_network)

            completed_compute = []
            while (compute_heap and
                   compute_heap[0][0] <= event_time + 1e-15):
                finish, replica_id, chip_id = heapq.heappop(compute_heap)
                running_chips.remove(chip_id)
                completed_compute.append((replica_id, finish))
            for replica_id, finish in sorted(completed_compute):
                for binding in bindings_by_replica[replica_id]:
                    if self.capabilities.combine_reduce_chip:
                        group_key = (
                            self._token_key(binding.assignment),
                            binding.replica.chip_id,
                        )
                        combine_ready[group_key][
                            binding.assignment.assignment_id
                        ] = finish
                    elif self.capabilities.combine_reduce_node:
                        source_node = self.dispatcher.replica_plan.chip(
                            binding.replica.chip_id
                        ).node_id
                        group_key = (
                            self._token_key(binding.assignment),
                            source_node,
                        )
                        index = sorted(
                            node_combine_groups, key=str
                        ).index(group_key)
                        flow = Flow(
                            flow_id="combine:node%d:input:%s" %
                            (index, binding.assignment.assignment_id),
                            src=self.dispatcher.chip_endpoints[
                                binding.replica.chip_id
                            ],
                            dst=self.dispatcher.home_endpoints[source_node],
                            nbytes=self.dispatcher.hidden_bytes,
                            earliest_start_s=finish,
                            traffic_class="combine",
                        )
                        flow_kind[flow.flow_id] = (
                            "combine_node_input", group_key
                        )
                        network.add_flow(flow)
                    else:
                        flow = self._combine_flow(binding, finish)
                        flow_kind[flow.flow_id] = (
                            "combine",
                            (binding.assignment.assignment_id,),
                        )
                        network.add_flow(flow)
            if self.capabilities.combine_reduce_chip:
                for group_index, group_key in enumerate(sorted(
                        combine_groups, key=str)):
                    if group_key in launched_combine_groups:
                        continue
                    group = combine_groups[group_key]
                    if len(combine_ready[group_key]) != len(group):
                        continue
                    release = (
                        max(combine_ready[group_key].values()) +
                        self.chip_reduce_latency_s
                    )
                    if self.capabilities.combine_reduce_node:
                        binding = group[0]
                        source_node = self.dispatcher.replica_plan.chip(
                            binding.replica.chip_id
                        ).node_id
                        node_key = (
                            self._token_key(binding.assignment),
                            source_node,
                        )
                        flow = Flow(
                            flow_id="combine:g%d:node-input" % group_index,
                            src=self.dispatcher.chip_endpoints[
                                binding.replica.chip_id
                            ],
                            dst=self.dispatcher.home_endpoints[source_node],
                            nbytes=self.dispatcher.hidden_bytes,
                            earliest_start_s=release,
                            traffic_class="combine",
                        )
                        flow_kind[flow.flow_id] = (
                            "combine_node_input", node_key
                        )
                    else:
                        base = self._combine_flow(group[0], release)
                        flow = Flow(
                            flow_id="combine:g%d" % group_index,
                            src=base.src,
                            dst=base.dst,
                            nbytes=base.nbytes,
                            earliest_start_s=base.earliest_start_s,
                            traffic_class=base.traffic_class,
                            waypoints=base.waypoints,
                        )
                        flow_kind[flow.flow_id] = (
                            "combine",
                            tuple(
                                item.assignment.assignment_id
                                for item in group
                            ),
                        )
                    network.add_flow(flow)
                    launched_combine_groups.add(group_key)
            accept_network_results(network.results.values())
            schedule_ready(event_time)

        flow_results = network.results
        dispatch_results = tuple(
            flow for flow in flow_results.values()
            if flow.flow.traffic_class == "dispatch"
        )
        combine_results = tuple(
            flow for flow in flow_results.values()
            if flow.flow.traffic_class == "combine"
        )
        dispatch_flow_count = len(dispatch_results)
        combine_flow_count = len(combine_results)
        physical_dispatch_bytes = sum(
            flow.flow.nbytes for flow in dispatch_results
        )
        physical_combine_bytes = sum(
            flow.flow.nbytes for flow in combine_results
        )
        if len(executions) != len(bindings_by_replica):
            raise RuntimeError("not every replica batch executed")
        if len(combine_finished) != len(items):
            raise RuntimeError("combine conservation failed")

        critical_assignment = max(
            combine_finished,
            key=lambda assignment_id: (
                combine_finished[assignment_id], assignment_id
            ),
        )
        critical_binding = binding_by_assignment[critical_assignment]
        critical_replica = critical_binding.replica.replica_id
        critical = LayerCriticalPath(
            assignment_id=critical_assignment,
            replica_id=critical_replica,
            dispatch_finish_s=arrived[
                critical_replica
            ][critical_assignment],
            compute_finish_s=executions[
                critical_replica
            ].compute_finish_s,
            combine_finish_s=combine_finished[critical_assignment],
        )
        local_count = sum(
            self.dispatcher.replica_plan.chip(
                binding.replica.chip_id
            ).node_id == binding.assignment.home_node
            for binding in dispatch_plan.bindings
        )
        network_energy = sum(
            result.energy_j for result in flow_results.values()
        )
        contention = sum(
            result.contention_delay_s
            for result in flow_results.values()
        )
        result = MoELayerResult(
            layer_id=layer_id,
            completion_s=max(combine_finished.values()),
            assignment_count=len(items),
            replica_executions=tuple(
                executions[key] for key in sorted(executions)
            ),
            flow_results=flow_results,
            link_bytes=network.link_bytes,
            network_energy_j=network_energy,
            network_contention_s=contention,
            local_assignment_fraction=local_count / len(items),
            dispatch_flow_count=dispatch_flow_count,
            combine_flow_count=combine_flow_count,
            logical_dispatch_bytes=(
                len(items) * self.dispatcher.hidden_bytes
            ),
            physical_dispatch_bytes=physical_dispatch_bytes,
            logical_combine_bytes=(
                len(items) * self.dispatcher.hidden_bytes
            ),
            physical_combine_bytes=physical_combine_bytes,
            critical_path=critical,
            warnings=tuple(sorted(warnings)),
        )
        # Imported locally to avoid a module cycle: validation refers to the
        # immutable result schema above.
        from .validation import validate_layer_result
        validate_layer_result(result)
        return result
