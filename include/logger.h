// logger.h
#ifndef LOGGER_H
#define LOGGER_H

#include <stdio.h>
#include "log_levels.h"
#include "log_config.h"
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#define LOG(level, fmt, ...) \
do { \
    if (level >= CURRENT_LOG_LEVEL) { \
        time_t t = time(NULL); \
        struct tm* tm_info = localtime(&t); \
        char time_str[20]; \
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info); \
        fprintf(stderr, "%s[%s]:s2s_nn:(%s.%d):tid:(%lu) " fmt "\n",time_str,log_level_to_string(level), __func__, __LINE__, syscall(SYS_gettid), ##__VA_ARGS__); \
        fflush(stderr); \
    } \
} while (0)


static inline const char* log_level_to_string(int level) {
    switch (level) {
        case LOG_LEVEL_DEBUG: return "DEBUG";
        case LOG_LEVEL_INFO:  return "INFO";
        case LOG_LEVEL_WARN:  return "WARN";
        case LOG_LEVEL_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

#endif // LOGGER_H

