from .arxiv import ArxivSummarizationAdapter
from .longbench import LongBenchAdapter
from .mooncake import MooncakeTraceAdapter
from .servegen import ServeGenBridge
from .sharegpt import ShareGPTAdapter
from .swe_agent import SWEAgentTrajectoriesAdapter
from .swebench_verified import SWEBenchVerifiedAdapter
from .thoughtworks_agent import ThoughtworksAgenticAdapter

ADAPTERS = {
    "sharegpt": ShareGPTAdapter,
    "swe-agent": SWEAgentTrajectoriesAdapter,
    "swebench-verified": SWEBenchVerifiedAdapter,
    "thoughtworks": ThoughtworksAgenticAdapter,
    "arxiv": ArxivSummarizationAdapter,
    "longbench": LongBenchAdapter,
    "mooncake": MooncakeTraceAdapter,
    "servegen": ServeGenBridge,
}
