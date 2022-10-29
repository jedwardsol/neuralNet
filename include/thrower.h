#pragma once

#include <system_error>
#include <string>
#include <sstream>
#include <source_location>
#include <filesystem>

#include <Windows.h>


[[nodiscard]] inline int lastError()  noexcept
{
    return GetLastError();
}

[[noreturn]] inline  void  throw_system_error(const std::string &what, const std::source_location location=std::source_location::current())
{
    auto where = std::ostringstream{};
    auto path  = std::filesystem::path{location.file_name()};

    where << path.filename().string() << '(' << location.line() << ")!" << location.function_name();

    throw std::system_error{lastError(), std::system_category(), where.str() + " : " + what};
}

[[noreturn]] inline  void  throw_system_error(int error, const std::string& what, const std::source_location location = std::source_location::current())
{
    auto where = std::ostringstream{};
    auto path = std::filesystem::path{ location.file_name() };

    where << path.filename().string() << '(' << location.line() << ")!" << location.function_name();

    throw std::system_error{ error, std::system_category(), where.str() + " : " + what };
}

[[noreturn]] inline  void  throw_runtime_error(const std::string &what, const std::source_location location=std::source_location::current())
{
    auto where = std::ostringstream{};
    auto path  = std::filesystem::path{location.file_name()};

    where << path.filename().string() << '(' << location.line() << ")!" << location.function_name();

    throw std::runtime_error{where.str() + " : " + what};
}
