#include <filesystem>
namespace fs = std::filesystem;

struct DebugContext {
    fs::path currentPs4Lib;
    FILE *logfile;

    void init(fs::path logfilePath) {
        logfile = fopen(logfilePath.c_str(), "w+");
        setvbuf(logfile, NULL, _IONBF, 0);    
    }
};

#if defined LOGGER_IMPL
DebugContext g_DebugContext;
#else
extern DebugContext g_DebugContext;
#endif

#define LOGFILE(fmt, ...) \
    fprintf(g_DebugContext.logfile, fmt, ##__VA_ARGS__);
