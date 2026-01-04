# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings

# Suppress noisy DeprecationWarnings from SWIG-generated types (importlib bootstrap).
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPy.* has no __module__ attribute",
    category=DeprecationWarning,
)
# Also catch other SWIG-generated wrappers (e.g., swigvarlink) that emit the
# same __module__ attribute warning during import bootstrap.
warnings.filterwarnings(
    "ignore",
    message=r"builtin type swig.* has no __module__ attribute",
    category=DeprecationWarning,
)
# Some SWIG wrappers emit from importlib bootstrap; silence those too.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"importlib\._bootstrap",
)
