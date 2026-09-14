"""Explicit OpenAI-compatible payload capabilities; no online discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit


PROTECTED = {'messages', 'prompt', 'model', 'stream', 'stream_options', 'tools', 'functions',
             'tool_calls', 'tool_call_id', 'function_call', 'request_id', 'max_tokens', 'max_completion_tokens', 'n'}


def api_url(base_url: str, path: str = '/v1/chat/completions') -> str:
    root = base_url.rstrip('/')
    parts = urlsplit(root)
    if parts.scheme not in {'http', 'https'} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('base_url must be an HTTP(S) API root without credentials, query or fragment')
    if root.endswith('/v1') and path.startswith('/v1/'):
        path = path[3:]
    return root + path


@dataclass(slots=True)
class TargetCapabilities:
    allowed_sampling: tuple[str, ...] = ()
    omitted_sampling: tuple[str, ...] = ('seed', 'temperature', 'top_p')
    output_parameter: str = 'max_tokens'
    include_usage: bool = True
    extra_parameters: dict = field(default_factory=dict)

    def payload(self, request, model: str, stream: bool) -> dict:
        if self.output_parameter not in {'max_tokens', 'max_completion_tokens'}:
            raise ValueError('unsupported logical output parameter mapping')
        if PROTECTED & self.extra_parameters.keys():
            raise ValueError('extra parameters cannot override frozen content, identity, output cap or tools')
        if set(self.allowed_sampling) & PROTECTED:
            raise ValueError('protected fields cannot be sampling parameters')
        if set(self.allowed_sampling) & set(self.omitted_sampling):
            raise ValueError('allowed and omitted parameters overlap')
        unknown = request.sampling.keys() - set(self.allowed_sampling) - set(self.omitted_sampling)
        if unknown:
            raise ValueError('unsupported sampling parameters: ' + ', '.join(sorted(unknown)))
        if set(request.sampling) & self.extra_parameters.keys():
            raise ValueError('provider parameters cannot silently override frozen sampling')
        if request.endpoint_kind != 'chat' or not request.messages:
            raise ValueError('agentic target requires chat messages')
        if any(set(m) != {'role', 'content'} or m['role'] not in {'system', 'user', 'assistant'}
               or not isinstance(m['content'], str) for m in request.messages):
            raise ValueError('agentic target requires compile-time normalized text messages')
        body = {'model': model, 'stream': stream, 'messages': request.messages,
                self.output_parameter: request.max_output_tokens,
                **{k: v for k, v in request.sampling.items() if k in self.allowed_sampling},
                **self.extra_parameters}
        if stream and self.include_usage:
            body['stream_options'] = {'include_usage': True}
        return body
