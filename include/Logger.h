#include <stdio.h>
#include <stdarg.h>

static void log_print(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

#define LOG(fmt, ...) \
    log_print(fmt, ##__VA_ARGS__);

