use super::startup_handles::host_ram_offload_hint;

/// Why a split start fell back from a local launch.
pub(super) struct CapacityFallback<'a> {
    pub(super) announced: bool,
    pub(super) model_name: &'a str,
    pub(super) required_bytes: u64,
    pub(super) local_capacity_bytes: u64,
    pub(super) host_ram_offload_gain_bytes: u64,
}

pub(super) enum SplitStartRetry {
    Wait(String),
    Stop(anyhow::Error),
}

pub(super) fn split_start_retry_decision(
    err: anyhow::Error,
    fallback: &CapacityFallback<'_>,
) -> SplitStartRetry {
    let err_msg = format!("{err:#}");
    if !is_retryable_split_start_failure(&err_msg) {
        return SplitStartRetry::Stop(err);
    }
    // An incoming peer can supply stages even when no outbound join is pending.
    if fallback.announced && split_participant_shortage(&err_msg) {
        return SplitStartRetry::Wait(format!(
            "{err_msg}; {}",
            waiting_for_split_peer_message(fallback)
        ));
    }
    SplitStartRetry::Wait(err_msg)
}

fn waiting_for_split_peer_message(fallback: &CapacityFallback<'_>) -> String {
    let offload = host_ram_offload_hint(
        fallback.required_bytes,
        fallback.local_capacity_bytes,
        fallback.host_ram_offload_gain_bytes,
    );
    format!(
        "Model {} needs {:.1} GB but this node plans on {:.1} GB; waiting for eligible peers to join and take split stages. Have another node join this mesh, or connect this node with --join or --auto{offload}",
        fallback.model_name,
        fallback.required_bytes as f64 / 1e9,
        fallback.local_capacity_bytes as f64 / 1e9,
    )
}

pub(super) fn is_retryable_split_start_failure(message: &str) -> bool {
    split_participants_are_still_converging(message) || split_control_transport_failed(message)
}

fn split_participant_shortage(message: &str) -> bool {
    message.contains("at least two participating nodes")
        || message.contains("at least two stage participants")
}

fn split_participants_are_still_converging(message: &str) -> bool {
    split_participant_shortage(message)
        || message.contains("split_capacity_shortfall")
        || message.contains("canonical coordinator")
        || (message.contains("split topology lock stage")
            && message.contains("matched 0 eligible nodes"))
}

fn split_control_transport_failed(message: &str) -> bool {
    let is_control_operation =
        message.contains("load split stage") || message.contains("stage_control_unreachable");
    let is_transport_failure = message.contains("connection lost")
        || message.contains("stream finished early")
        || message.contains("timeout waiting for stage control response");
    is_control_operation && is_transport_failure
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capacity_fallback() -> CapacityFallback<'static> {
        CapacityFallback {
            announced: true,
            model_name: "Qwen3-32B",
            required_bytes: 20_240_000_000,
            local_capacity_bytes: 12_878_610_432,
            host_ram_offload_gain_bytes: 18_548_836_761,
        }
    }

    #[test]
    fn capacity_fallback_waits_for_incoming_peers() {
        for error in [
            "split runtime needs at least two participating nodes for m; found 1 eligible",
            "split runtime needs at least two stage participants",
        ] {
            let SplitStartRetry::Wait(message) =
                split_start_retry_decision(anyhow::anyhow!(error), &capacity_fallback())
            else {
                panic!("a capacity fallback must keep waiting for incoming peers");
            };
            assert!(message.contains(error), "{message}");
            assert!(
                message.contains("waiting for eligible peers to join"),
                "{message}"
            );
        }
    }

    #[test]
    fn retry_decision_preserves_other_retryable_failures() {
        let mut fallback = capacity_fallback();
        fallback.announced = false;
        let quorum = "split runtime needs at least two participating nodes";
        let SplitStartRetry::Wait(message) =
            split_start_retry_decision(anyhow::anyhow!(quorum), &fallback)
        else {
            panic!("a forced split must keep waiting");
        };
        assert_eq!(message, quorum);

        for error in [
            "split_capacity_shortfall: mesh has insufficient capacity",
            "load split stage stage-1: connection lost: closed",
        ] {
            let SplitStartRetry::Wait(message) =
                split_start_retry_decision(anyhow::anyhow!(error), &capacity_fallback())
            else {
                panic!("a transient split failure must remain retryable");
            };
            assert_eq!(message, error);
        }
    }

    #[test]
    fn retry_decision_stops_on_native_or_configuration_failures() {
        for error in [
            "load skippy stage 0 runtime: unable to allocate CUDA0 buffer",
            "split topology lock stage 3 selector worker matched 2 eligible nodes",
            "split topology lock manifest abc does not match resolved package manifest def",
        ] {
            let SplitStartRetry::Stop(actual) =
                split_start_retry_decision(anyhow::anyhow!(error), &capacity_fallback())
            else {
                panic!("a terminal failure must not become a participant wait");
            };
            assert_eq!(actual.to_string(), error);
        }
    }

    #[test]
    fn the_participant_wait_message_says_what_to_do() {
        let mut fallback = capacity_fallback();
        let message = waiting_for_split_peer_message(&fallback);
        assert!(message.contains("needs 20.2 GB"), "{message}");
        assert!(message.contains("plans on 12.9 GB"), "{message}");
        assert!(
            message.contains("Have another node join this mesh"),
            "{message}"
        );
        assert!(message.contains("--join or --auto"), "{message}");
        assert!(message.contains("gpu.host_ram_offload = true"), "{message}");

        fallback.host_ram_offload_gain_bytes = 0;
        assert!(!waiting_for_split_peer_message(&fallback).contains("host_ram_offload"));

        fallback.required_bytes = 40_000_000_000;
        fallback.host_ram_offload_gain_bytes = 18_548_836_761;
        assert!(!waiting_for_split_peer_message(&fallback).contains("host_ram_offload"));
    }

    #[test]
    fn participant_shortage_is_retryable() {
        assert!(is_retryable_split_start_failure(
            "split runtime needs at least two participating nodes for model"
        ));
        assert!(is_retryable_split_start_failure(
            "split runtime needs at least two stage participants"
        ));
    }

    #[test]
    fn transient_capacity_shortfall_is_retryable() {
        assert!(is_retryable_split_start_failure(
            "split_capacity_shortfall: unable to plan split topology: max_placeable_layers_at_evaluated_shape=21/66"
        ));
    }

    #[test]
    fn missing_locked_participant_is_retryable() {
        assert!(is_retryable_split_start_failure(
            "split topology lock stage 3 selector \"worker\" matched 0 eligible nodes; available: local"
        ));
    }

    #[test]
    fn canonical_coordinator_mismatch_is_retryable() {
        assert!(is_retryable_split_start_failure(
            "split topology stage 0 node-a does not match canonical coordinator node-b"
        ));
        assert!(is_retryable_split_start_failure(
            "split topology lock stage 0 must be canonical coordinator node-b"
        ));
    }

    #[test]
    fn ambiguous_or_invalid_lock_is_not_retryable() {
        assert!(!is_retryable_split_start_failure(
            "split topology lock stage 3 selector \"worker\" matched 2 eligible nodes"
        ));
        assert!(!is_retryable_split_start_failure(
            "split topology lock manifest abc does not match resolved package manifest def"
        ));
    }

    #[test]
    fn stage_control_transport_failure_is_retryable() {
        assert!(is_retryable_split_start_failure(
            "load split stage stage-1: connection lost: closed"
        ));
        assert!(is_retryable_split_start_failure(
            "load split stage stage-2: stage_control_unreachable: stream finished early"
        ));
        assert!(is_retryable_split_start_failure(
            "load split stage stage-3: timeout waiting for stage control response"
        ));
    }

    #[test]
    fn runtime_or_unrelated_transport_failure_is_not_retryable() {
        assert!(!is_retryable_split_start_failure(
            "load skippy stage 0 runtime: unable to allocate CUDA0 buffer"
        ));
        assert!(!is_retryable_split_start_failure(
            "artifact transfer connection lost: closed"
        ));
    }
}
