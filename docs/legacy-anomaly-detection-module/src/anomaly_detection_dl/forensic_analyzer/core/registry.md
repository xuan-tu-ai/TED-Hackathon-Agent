# registry.py

Registry system implementing the Strategy Pattern to manage symptom and pattern scorer classes. Provides class methods to register symptoms and scorers with their associated config schemas for runtime discovery.

## Source Code

```python
from typing import Dict, Type, Callable

class ForensicRegistry:
    """ 
    A registry system using Strategy Pattern to manage symptoms and pattern's scorers.
    """

    SYMPTOMS: Dict[str, Type] = {}
    SYMPTOMS_CONFIGS: Dict[str, Type] = {}
    PATTERN_SCORERS: Dict[str, Type] = {}

    @classmethod
    def register_symptom(cls, name: str, config_schema):
        def wrapper(symptom_cls):
            cls.SYMPTOMS[name] = symptom_cls
            cls.SYMPTOMS_CONFIGS[name] = config_schema
            return symptom_cls
        return wrapper
    
    @classmethod
    def register_pattern_scorer(cls, name: str):
        def wrapper(scorer_cls):
            cls.PATTERN_SCORERS[name] = scorer_cls
            return scorer_cls
        return wrapper
```