// include/strata/base/platform/env.hpp - setting and removing environment variables on Linux and on Windows, which
// has no setenv or unsetenv.
#pragma once

#include <cstdlib>

namespace strata::platform {

/// Sets the environment variable `name` to `value`; with `overwrite` false, a value already set is kept. On Windows an
/// empty `value` removes the variable.
inline void set_env(const char* name, const char* value, bool overwrite = true) {
    if (!overwrite && std::getenv(name) != nullptr)
        return;
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

/// Removes the environment variable `name`.
inline void unset_env(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

}  // namespace strata::platform
