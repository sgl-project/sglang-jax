from abc import ABC, abstractmethod
from collections.abc import Iterator


class EngineBase(ABC):
    """
    Abstract base class for engine interfaces that support generation, weight updating, and memory control.
    This base class provides a unified API for both HTTP-based engines and engines.
    """

    @abstractmethod
    def generate(
        self,
        prompt: list[str] | str | None = None,
        sampling_params: list[dict] | dict | None = None,
        input_ids: list[list[int]] | list[int] | None = None,
        image_data: list[str] | str | None = None,
        return_logprob: list[bool] | bool | None = False,
        logprob_start_len: list[int] | int | None = None,
        top_logprobs_num: list[int] | int | None = None,
        token_ids_logprob: list[list[int]] | list[int] | None = None,
        lora_path: list[str | None] | str | None | None = None,
        custom_logit_processor: list[str] | str | None = None,
        return_hidden_states: bool | None = None,
        stream: bool | None = None,
        bootstrap_host: list[str] | str | None = None,
        bootstrap_port: list[int] | int | None = None,
        bootstrap_room: list[int] | int | None = None,
        data_parallel_rank: int | None = None,
    ) -> dict | Iterator[dict]:
        """Generate outputs based on given inputs."""
        pass

    @abstractmethod
    def flush_cache(self):
        """Flush the cache of the engine."""
        pass

    @abstractmethod
    def release_memory_occupation(self):
        """Release GPU memory occupation temporarily."""
        pass

    @abstractmethod
    def resume_memory_occupation(self):
        """Resume GPU memory occupation which is previously released."""
        pass

    @abstractmethod
    def shutdown(self):
        """Shutdown the engine and clean up resources."""
        pass
