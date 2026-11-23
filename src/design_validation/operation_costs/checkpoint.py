"""Atomic, immutable, missing-only checkpoints bound to a scientific workspace."""
from pathlib import Path
import re

from ..personal_time_inputs import digest, need, read, write


class Workspace:
    def __init__(self, output, binding):
        self.output = Path(output)
        self.binding = digest(binding)
        self.manifest = {}
        self.created = 0
        self.reused = 0

    def path(self, name):
        need(type(name) is str and re.fullmatch(r'[A-Za-z0-9_-]+', name), 'Safe phase name')
        return self.output / 'checkpoints' / (name+'.json')

    def get(self, name, context):
        path = self.path(name)
        need(not path.is_symlink(), 'No checkpoint symlink')
        if not path.exists():
            return None
        value = read(path)
        need(set(value) == {'schema', 'binding', 'name', 'context', 'result', 'contentSHA256'}
             and value['schema'] == 'operation-cost-phase-v1'
             and value['binding'] == self.binding and value['name'] == name
             and value['context'] == context
             and value['contentSHA256'] == digest({k:v for k,v in value.items() if k != 'contentSHA256'}),
             'Foreign, truncated or modified checkpoint')
        self.manifest[name] = value['contentSHA256']
        self.reused += 1
        return value['result']

    def put(self, name, context, result):
        value = dict(schema='operation-cost-phase-v1', binding=self.binding,
                     name=name, context=context, result=result)
        value['contentSHA256'] = digest(value)
        write(self.path(name), value)
        self.manifest[name] = value['contentSHA256']
        self.created += 1
        return result

    def obtain(self, name, context, generate):
        value = self.get(name, context)
        return self.put(name, context, generate()) if value is None else value
