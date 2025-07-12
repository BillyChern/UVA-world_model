# Make unified_video_action importable as a package
import os, sys as _sys
_root = os.path.dirname(__file__)
_inner = os.path.join(_root, 'unified_video_action')
if os.path.isdir(_inner) and _inner not in _sys.path:
    _sys.path.insert(0, _inner)
    # allow submodule imports like `unified_video_action.gym_util` to resolve
    if _inner not in __path__:
        __path__.append(_inner) 