#ifndef __JINBO_LOG_HPP__
#define __JINBO_LOG_HPP__

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cstdio>
#include <list>
#include <string>
#include <vector>

#define unuse(x) ((x) = (x))
#define LOG_TIME_SIZE (128)
#define LOG_TXT_SIZE (256)
#define LOG_FULL_TXT_SIZE (512)

#define JINBO_DEBUG (1)
#define PRINT_FULL_FUNCTION_NAME (0)
#define PRINT_THREAD_PROCESS_ID_AND_FULL_TIME (1)

#if PRINT_FULL_FUNCTION_NAME
#    define FUNC_FORMAT __PRETTY_FUNCTION__
#else
#    define FUNC_FORMAT __func__
#endif

#define JINBO_COLOR_WHITE "\033[1m\033[40;37m"
#define JINBO_COLOR_GREEN "\033[1m\033[40;32m"
#define JINBO_COLOR_YELLOW "\033[1m\033[40;33m"
#define JINBO_COLOR_RED "\033[1m\033[40;31m"
#define JINBO_COLOR_PINK "\033[1m\033[40;35m"
#define JINBO_COLOR_BLUE "\033[1m\033[40;34m"

static const int JINBO_LOG_LEVEL_NOTHING = 0;
static const int JINBO_LOG_LEVEL_DEBUG   = 1;
static const int JINBO_LOG_LEVEL_INFO    = 2;
static const int JINBO_LOG_LEVEL_WARNING = 3;
static const int JINBO_LOG_LEVEL_ERROR   = 4;
static const int JINBO_LOG_LEVEL_ALWAYS  = 99;

__attribute__((unused)) static int getLogLevel()
{
    char *env_log_level = getenv("JINBO_LOG_LEVEL");
    // int level            = JINBO_LOG_LEVEL_WARNING;
    int level = JINBO_LOG_LEVEL_DEBUG;
    if (env_log_level)
    {
        try
        {
            level = std::stoi(env_log_level);
        }
        catch (...)
        {
            level = JINBO_LOG_LEVEL_WARNING;  // warning
        }

        if (level < JINBO_LOG_LEVEL_DEBUG || level > JINBO_LOG_LEVEL_ERROR)
        {
            level = JINBO_LOG_LEVEL_WARNING;
        }
    }
    return level;
}

__attribute__((unused)) static int logLevel = getLogLevel();

// Print only file names (filter out directories)
#define FILE_NAME(x) (strrchr((x), '/') ? strrchr((x), '/') + 1 : (x))

__attribute__((unused)) static thread_local char logTime[LOG_TIME_SIZE]     = {0};
__attribute__((unused)) static thread_local char logTxt[LOG_TXT_SIZE]       = {0};
__attribute__((unused)) static thread_local char fullTxt[LOG_FULL_TXT_SIZE] = {0};

__attribute__((unused)) static void getTimeStr(char *t)
{
    struct timeval tv;
    struct timezone tz;
    struct tm *p;

    gettimeofday(&tv, &tz);
    p = localtime(&tv.tv_sec);
#if (PRINT_THREAD_PROCESS_ID_AND_FULL_TIME)
    snprintf(t, LOG_TIME_SIZE, "%04d/%02d/%02d %02d:%02d:%02d.%06ld %u %d", p->tm_year + 1900, p->tm_mon + 1,
             p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec, tv.tv_usec, (unsigned int)pthread_self(), getpid());
#else
    snprintf(t, LOG_TIME_SIZE, "%02d:%02d:%02d.%06ld", p->tm_hour, p->tm_min, p->tm_sec, tv.tv_usec);
//     snprintf(t, LOG_TIME_SIZE, "%s@%s %02d:%02d:%02d.%06ld", get_host_name().c_str(), get_local_ip().c_str(),
//              p->tm_hour, p->tm_min, p->tm_sec, tv.tv_usec);
#endif
}

#define jinbolog(level, color, isPrintNodeInfo, format, ...)                                                       \
    do                                                                                                             \
    {                                                                                                              \
        if (level >= logLevel)                                                                                     \
        {                                                                                                          \
            getTimeStr(logTime);                                                                                   \
            sprintf(logTxt, format, ##__VA_ARGS__);                                                                \
            if (isPrintNodeInfo)                                                                                   \
            {                                                                                                      \
                sprintf(fullTxt, "%s%s [%s:%s:%d] %s \033[0m\n", color, logTime, FILE_NAME(__FILE__), FUNC_FORMAT, \
                        __LINE__, logTxt);                                                                         \
            }                                                                                                      \
            else                                                                                                   \
            {                                                                                                      \
                sprintf(fullTxt, "%s%s [%s:%s:%d] %s \033[0m\n", color, logTime, FILE_NAME(__FILE__), FUNC_FORMAT, \
                        __LINE__, logTxt);                                                                         \
            }                                                                                                      \
            fprintf(stdout, "%s", fullTxt);                                                                        \
        }                                                                                                          \
    } while (0);

#if (JINBO_DEBUG)

#    define logi(format, ...) jinbolog(JINBO_LOG_LEVEL_INFO, JINBO_COLOR_WHITE, 0, format, ##__VA_ARGS__)

#    define logd(format, ...) jinbolog(JINBO_LOG_LEVEL_DEBUG, JINBO_COLOR_GREEN, 0, format, ##__VA_ARGS__)

#    define logw(format, ...) jinbolog(JINBO_LOG_LEVEL_WARNING, JINBO_COLOR_YELLOW, 0, format, ##__VA_ARGS__)

#    define loge(format, ...) jinbolog(JINBO_LOG_LEVEL_ERROR, JINBO_COLOR_RED, 0, format, ##__VA_ARGS__)

#    define logm(format, ...) jinbolog(JINBO_LOG_LEVEL_ERROR, JINBO_COLOR_PINK, 0, format, ##__VA_ARGS__)

#    define loga(format, ...) jinbolog(JINBO_LOG_LEVEL_ALWAYS, JINBO_COLOR_BLUE, 0, format, ##__VA_ARGS__)

#    define logmark logm

#    define login() logi("IN")

#    define logout() logi("OUT")

#else
#    define logi(format, ...)
#    define logd(format, ...)
#    define logw(format, ...)
#    define loge(format, ...)
#    define logmark(format, ...)
#    define login(format, ...)
#    define logout(format, ...)

#endif  // end JINBO_DEBUG

#endif  // end __JINBO_LOG_H__