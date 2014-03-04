AC_INIT([cpuminer], [2.3.3])

AC_PREREQ([2.59c])
AC_CANONICAL_SYSTEM
AC_CONFIG_SRCDIR([cpu-miner.c])
AM_INIT_AUTOMAKE([gnu])
AC_CONFIG_HEADERS([cpuminer-config.h])

dnl Make sure anyone changing configure.ac/Makefile.am has a clue
AM_MAINTAINER_MODE

dnl Checks for programs
AC_PROG_CC_C99
AC_PROG_GCC_TRADITIONAL
AM_PROG_CC_C_O
AM_PROG_AS
AC_PROG_RANLIB
AC_PROG_CXX

dnl Checks for header files
AC_HEADER_STDC
AC_CHECK_HEADERS([sys/endian.h sys/param.h syslog.h])
# sys/sysctl.h requires sys/types.h on FreeBSD
# sys/sysctl.h requires sys/param.h on OpenBSD
AC_CHECK_HEADERS([sys/sysctl.h], [], [],
[#include <sys/types.h>
#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
])

AC_CHECK_DECLS([be32dec, le32dec, be32enc, le32enc], [], [],
[AC_INCLUDES_DEFAULT
#ifdef HAVE_SYS_ENDIAN_H
#include <sys/endian.h>
#endif
])

AC_FUNC_ALLOCA
AC_CHECK_FUNCS([getopt_long])

case $target in
  i*86-*-*)
    have_x86=true
    ;;
  x86_64-*-*|amd64-*-*)
    have_x86_64=true
    ;;
  arm*-*-*)
    have_arm=true
    ;;
esac

PTHREAD_FLAGS="-pthread"
WS2_LIBS=""

case $target in
  *-*-mingw*)
    have_win32=true
    PTHREAD_FLAGS=""
    WS2_LIBS="-lws2_32"
    ;;
esac

dnl Figure out where OpenCL is, either passed or system-wide
AC_ARG_WITH([opencl-framework],
	[AC_HELP_STRING([--with-opencl-framework],[use OpenCL framework (OSX)])],
	[with_opencl_framework="yes"],
	[with_opencl_framework="no"])
if test "x$with_opencl_framework" = xno; then
	AC_CHECK_LIB([OpenCL], [OpenCL], [OPENCL_LIBS="-lOpenCL"], AC_MSG_ERROR([OpenCL library could not be found]))
else
	OPENCL_LIBS="-framework OpenCL"
fi
OPENCL_INCLUDES=""
AC_ARG_WITH([opencl-inc],
	[AC_HELP_STRING([--with-opencl-inc],[specify OpenCL include paths])],
	[with_opencl_inc=$withval],
	[with_opencl_inc="auto"])
if test "x$with_opencl_inc" = xauto; then
	AC_CHECK_HEADERS(CL/cl.h, [found="yes"])
	AC_CHECK_HEADERS(OpenCL/opencl.h, [found="yes"])
	if test "x$found" = x; then
		AC_MSG_ERROR([OpenCL headers could not be found])
	fi
else
	OPENCL_INCLUDES="-I$with_opencl_inc"
fi



AC_CHECK_LIB(jansson, json_loads, request_jansson=false, request_jansson=true)
AC_CHECK_LIB([pthread], [pthread_create], PTHREAD_LIBS="-lpthread",
  AC_CHECK_LIB([pthreadGC2], [pthread_create], PTHREAD_LIBS="-lpthreadGC2",
    AC_CHECK_LIB([pthreadGC1], [pthread_create], PTHREAD_LIBS="-lpthreadGC1",
      AC_CHECK_LIB([pthreadGC], [pthread_create], PTHREAD_LIBS="-lpthreadGC"
))))

AM_CONDITIONAL([WANT_JANSSON], [test x$request_jansson = xtrue])
AM_CONDITIONAL([HAVE_WINDOWS], [test x$have_win32 = xtrue])
AM_CONDITIONAL([ARCH_x86], [test x$have_x86 = xtrue])
AM_CONDITIONAL([ARCH_x86_64], [test x$have_x86_64 = xtrue])
AM_CONDITIONAL([ARCH_ARM], [test x$have_arm = xtrue])

if test x$request_jansson = xtrue
then
	JANSSON_LIBS="compat/jansson/libjansson.a"
else
	JANSSON_LIBS=-ljansson
fi

LIBCURL_CHECK_CONFIG(, 7.15.2, ,
  [AC_MSG_ERROR([Missing required libcurl >= 7.15.2])])

AC_SUBST(OPENCL_LIBS)
AC_SUBST(OPENCL_INCLUDES)
AC_SUBST(JANSSON_LIBS)
AC_SUBST(PTHREAD_FLAGS)
AC_SUBST(PTHREAD_LIBS)
AC_SUBST(WS2_LIBS)

AC_CONFIG_FILES([
	Makefile
	compat/Makefile
	compat/jansson/Makefile
	])
AC_OUTPUT
