#pragma once
#include <errno.h>
#include <cstring>
#include <ctime>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

class Logger {
  public:
    enum LogLevel { ERROR, WARN, INFO, TRACE };
    enum LogSubSys { ALL, INIT, NET, TUNER, OPTIMIZER, TIMER };
    static const char *LogLevelName[];
    static const char *LogSubSysName[];
    Logger(LogLevel level, LogSubSys subSys, const char *file, int32_t line)
        : level(level), subSys(subSys), file(file), line(line), logStream(std::cerr) {
      std::time_t now = std::time(nullptr);
      char buffer[80];
      std::strftime(buffer, 80, "%Y/%m/%d %H:%M:%S", std::localtime(&now));
      logStream << "[" << LogLevelName[level] << ", " << LogSubSysName[subSys] << "] " << buffer << " " << file
                << ":" << line << " : ";
    }
    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;
    Logger(Logger &&) = delete;
    Logger &operator=(Logger &&) = delete;

    static void setVerboseLevel(const char *levelStr) {
      if (levelStr == nullptr)
        return;
      if (strcmp(levelStr, Logger::LogLevelName[0]) == 0) {
        verboseLevel = Logger::LogLevel::ERROR;
      } else if (strcmp(levelStr, Logger::LogLevelName[1]) == 0) {
        verboseLevel = Logger::LogLevel::WARN;
      } else if (strcmp(levelStr, Logger::LogLevelName[2]) == 0) {
        verboseLevel = Logger::LogLevel::INFO;
      } else if (strcmp(levelStr, Logger::LogLevelName[3]) == 0) {
        verboseLevel = Logger::LogLevel::TRACE;
      } else {
        std::cerr << "Invalid log level, setting failed." << std::endl;
      }
    }

    static void setSubSys(const char *subSysStr) {
      if (subSysStr == nullptr)
        return;
      if (strcmp(subSysStr, Logger::LogSubSysName[0]) == 0) {
        verboseSubSys = Logger::LogSubSys::ALL;
      } else if (strcmp(subSysStr, Logger::LogSubSysName[1]) == 0) {
        verboseSubSys = Logger::LogSubSys::INIT;
      } else if (strcmp(subSysStr, Logger::LogSubSysName[2]) == 0) {
        verboseSubSys = Logger::LogSubSys::NET;
      } else if (strcmp(subSysStr, Logger::LogSubSysName[3]) == 0) {
        verboseSubSys = Logger::LogSubSys::TUNER;
      } else if (strcmp(subSysStr, Logger::LogSubSysName[4]) == 0) {
        verboseSubSys = Logger::LogSubSys::OPTIMIZER;
      } else if (strcmp(subSysStr, Logger::LogSubSysName[5]) == 0) {
        verboseSubSys = Logger::LogSubSys::TIMER;
      } else {
        std::cerr << "Invalid log subSys, setting failed." << std::endl;
      }
    }

    std::ostream &stream() { return logStream; }
    static LogLevel verboseLevel;
    static LogSubSys verboseSubSys;
    ~Logger() noexcept(false) {
      logStream << std::endl;
      if (level == LogLevel::ERROR) {
        throw std::runtime_error("Error occurred");
      }
    }

  private:
    LogLevel level;
    LogSubSys subSys;
    const char *file;
    int32_t line;
    std::ostream &logStream;
};

const char *Logger::LogLevelName[] = {"ERROR", "WARN", "INFO", "TRACE"};
const char *Logger::LogSubSysName[] = {"ALL", "INIT", "NET", "TUNER", "OPTIMIZER", "TIMER"};

Logger::LogLevel Logger::verboseLevel = Logger::LogLevel::ERROR;
Logger::LogSubSys Logger::verboseSubSys = Logger::LogSubSys::ALL;

class LoggerVoidify {
  public:
    // This has to be an operator with a precedence lower than << but
    // higher than "?:". See its usage.
    // void operator&(std::ostream &) {}
    std::ostream& operator&(std::ostream &os) {
      return os;
    }
    static std::ostream& nullStream() {
        static NullStream nullStream;
        return nullStream;
    }
  private:
    class NullStream : public std::ostream {
    public:
        NullStream() : std::ostream(&nullBuffer) {}
    private:
        class NullBuffer : public std::streambuf {
        protected:
            virtual int overflow(int c) override {
                return c;
            }
        } nullBuffer;
    };
};

#define LOG(level, subSys) Logger(level, subSys, __FILE__, __LINE__).stream()

#define LOG_VERBOSE(level, subSys)                                                     \
  (!(level <= Logger::verboseLevel) ? LoggerVoidify() & LoggerVoidify::nullStream() : LOG_SUBSYS(level, subSys))

#define LOG_SUBSYS(level, subSys)                                                     \
  (!(subSys == Logger::verboseSubSys || Logger::verboseSubSys == Logger::LogSubSys::ALL) ? LoggerVoidify() & LoggerVoidify::nullStream() : LOG(level, subSys))

#define CHECK(x)                                                               \
  if (!(x))                                                                    \
  LOG(Logger::LogLevel::ERROR, Logger::LogSubSys::ALL) << "Check failed: " #x << " :"

#define THROWERROR (LOG_VERBOSE(Logger::LogLevel::ERROR, Logger::LogSubSys::ALL))
#define WARN(subSys) (LOG_VERBOSE(Logger::LogLevel::WARN, subSys))
#define INFO(subSys) (LOG_VERBOSE(Logger::LogLevel::INFO, subSys))

#ifdef TUNER_TRACE
#define TRACE(subSys) (LOG_VERBOSE(Logger::LogLevel::TRACE, subSys))
#else
#define TRACE(subSys)                                                          \
  while (false)                                                                \
  (LOG_VERBOSE(Logger::LogLevel::TRACE, subSys))
#endif

#ifdef TUNER_TRACE
#include <chrono>
#include <iostream>
#define PERFDEBUG(func)                                                        \
  do {                                                                         \
    auto perfdebugStart = std::chrono::high_resolution_clock::now();           \
    func;                                                                      \
    auto perfdebugEnd = std::chrono::high_resolution_clock::now();             \
    auto perfdebugDuration =                                                   \
        std::chrono::duration_cast<std::chrono::microseconds>(perfdebugEnd -   \
                                                              perfdebugStart); \
    if (perfdebugDuration.count() > 10)                                        \
      TRACE << #func << " timecost: " << perfdebugDuration.count() << " us";   \
  } while (0);
#else
#define PERFDEBUG(...)
#endif

// Check system calls
#define SYSCHECK(call, name)                                                   \
  do {                                                                         \
    int32_t retval;                                                            \
    SYSCHECKVAL(call, name, retval);                                           \
  } while (false)

#define SYSCHECKVAL(call, name, retval)                                        \
  do {                                                                         \
    SYSCHECKSYNC(call, name, retval);                                          \
    if (retval == -1) {                                                        \
      WARN(Logger::LogSubSys::NET) << "Call to " name " failed : " << strerror(errno);                 \
      return;                                                                  \
    }                                                                          \
  } while (false)

#define SYSCHECKSYNC(call, name, retval)                                       \
  do {                                                                         \
    retval = call;                                                             \
    if (retval == -1 &&                                                        \
        (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) {         \
      WARN(Logger::LogSubSys::NET) << "Call to " name " returned " << strerror(errno);                 \
    } else {                                                                   \
      break;                                                                   \
    }                                                                          \
  } while (true)

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    cudaError_t result = (x);                                                  \
    if (result != cudaSuccess) {                                               \
      const char *msg = cudaGetErrorString(result);                            \
      std::stringstream safe_call_ss;                                          \
      safe_call_ss << "\nerror: " #x " failed with error"                      \
                   << "\nfile: " << __FILE__ << "\nline: " << __LINE__         \
                   << "\nmsg: " << msg;                                        \
      throw std::runtime_error(safe_call_ss.str());                            \
    }                                                                          \
  } while (0)

#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int32_t e = cmd;                                                           \
    if (e != MPI_SUCCESS) {                                                    \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#if 0
int32_t main() {
  // Read verbose level from environment variable
  const char* verbose = std::getenv("TUNER_VERBOSE");
  if (verbose != nullptr) {
    Logger::setVerboseLevel(verbose);
  }
  // Test logging
  TRACE(Logger::LogSubSys::ALL) << "This is a trace message";
  INFO(Logger::LogSubSys::ALL) << "This is an info message";
  WARN(Logger::LogSubSys::ALL) << "This is a warning message";
  try {
    CHECK(0);
  } catch (const std::runtime_error& e) {
    std::cerr << "Caught expected exception: " << e.what() << std::endl;
  }
  return 0;
}
#endif
