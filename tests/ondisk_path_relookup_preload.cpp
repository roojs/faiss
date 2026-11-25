/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Interposer for OnDiskEmbeddedPath.SameDirSymlinkIsNotReresolved.
 *
 * The first lookup of FAISS_OD_LINK is forwarded unchanged. After that call
 * returns, the symlink is pointed at FAISS_OD_SECRET. A later open of the
 * same path therefore sees a different inode than the first lookup. The
 * real open, stat, and truncate still run in libc.
 */

#define _GNU_SOURCE

#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>

/* glibc headers may macro-rename these to the *64 symbols. */
#undef fopen
#undef fopen64
#undef open
#undef open64
#undef openat
#undef openat64
#undef stat
#undef stat64
#undef lstat
#undef lstat64
#undef fstatat
#undef fstatat64
#undef realpath
#undef access
#undef faccessat
#undef truncate
#undef truncate64

static int armed = 0;

static FILE* (*real_fopen)(const char*, const char*);
static int (*real_open)(const char*, int, ...);
static int (*real_openat)(int, const char*, int, ...);
static int (*real_stat)(const char*, struct stat*);
static int (*real_lstat)(const char*, struct stat*);
static int (*real_fstatat)(int, const char*, struct stat*, int);
static int (*real_xstat)(int, const char*, struct stat*);
static int (*real_lxstat)(int, const char*, struct stat*);
static int (*real_fxstatat)(int, int, const char*, struct stat*, int);
static char* (*real_realpath)(const char*, char*);
static int (*real_access)(const char*, int);
static int (*real_faccessat)(int, const char*, int, int);
static int (*real_truncate)(const char*, off_t);

static int watched(const char* path) {
    const char* link_path;
    const char* base;
    if (armed || path == NULL) {
        return 0;
    }
    link_path = getenv("FAISS_OD_LINK");
    if (link_path == NULL) {
        return 0;
    }
    if (strcmp(path, link_path) == 0) {
        return 1;
    }
    base = strrchr(link_path, '/');
    base = base == NULL ? link_path : base + 1;
    return strcmp(path, base) == 0;
}

static void arm(void) {
    const char* link_path;
    const char* secret;
    if (armed) {
        return;
    }
    link_path = getenv("FAISS_OD_LINK");
    secret = getenv("FAISS_OD_SECRET");
    if (link_path == NULL || secret == NULL) {
        return;
    }
    /* Mark armed before the symlink swap so the unlink/symlink lookups do
     * not re-enter this hook. */
    armed = 1;
    unlink(link_path);
    if (symlink(secret, link_path) != 0) {
        armed = 0;
    }
}

__attribute__((constructor)) static void init_relookup_preload(void) {
    real_fopen = reinterpret_cast<FILE* (*)(const char*, const char*)>(
            dlsym(RTLD_NEXT, "fopen"));
    real_open = reinterpret_cast<int (*)(const char*, int, ...)>(
            dlsym(RTLD_NEXT, "open"));
    real_openat = reinterpret_cast<int (*)(int, const char*, int, ...)>(
            dlsym(RTLD_NEXT, "openat"));
    real_stat = reinterpret_cast<int (*)(const char*, struct stat*)>(
            dlsym(RTLD_NEXT, "stat"));
    real_lstat = reinterpret_cast<int (*)(const char*, struct stat*)>(
            dlsym(RTLD_NEXT, "lstat"));
    real_fstatat =
            reinterpret_cast<int (*)(int, const char*, struct stat*, int)>(
                    dlsym(RTLD_NEXT, "fstatat"));
    real_xstat = reinterpret_cast<int (*)(int, const char*, struct stat*)>(
            dlsym(RTLD_NEXT, "__xstat"));
    real_lxstat = reinterpret_cast<int (*)(int, const char*, struct stat*)>(
            dlsym(RTLD_NEXT, "__lxstat"));
    real_fxstatat =
            reinterpret_cast<int (*)(int, int, const char*, struct stat*, int)>(
                    dlsym(RTLD_NEXT, "__fxstatat"));
    real_realpath = reinterpret_cast<char* (*)(const char*, char*)>(
            dlsym(RTLD_NEXT, "realpath"));
    real_access = reinterpret_cast<int (*)(const char*, int)>(
            dlsym(RTLD_NEXT, "access"));
    real_faccessat = reinterpret_cast<int (*)(int, const char*, int, int)>(
            dlsym(RTLD_NEXT, "faccessat"));
    real_truncate = reinterpret_cast<int (*)(const char*, off_t)>(
            dlsym(RTLD_NEXT, "truncate"));
}

extern "C" {

FILE* fopen(const char* path, const char* mode) {
    FILE* f = real_fopen(path, mode);
    if (watched(path)) {
        arm();
    }
    return f;
}

FILE* fopen64(const char* path, const char* mode) {
    return fopen(path, mode);
}

int open(const char* path, int flags, ...) {
    mode_t mode = 0;
    int fd;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    fd = real_open(path, flags, mode);
    if (watched(path)) {
        arm();
    }
    return fd;
}

int open64(const char* path, int flags, ...) {
    mode_t mode = 0;
    va_list ap;
    va_start(ap, flags);
    if (flags & O_CREAT) {
        mode = (mode_t)va_arg(ap, int);
    }
    va_end(ap);
    return open(path, flags, mode);
}

int openat(int dirfd, const char* path, int flags, ...) {
    mode_t mode = 0;
    int fd;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    fd = real_openat(dirfd, path, flags, mode);
    if (watched(path)) {
        arm();
    }
    return fd;
}

int openat64(int dirfd, const char* path, int flags, ...) {
    mode_t mode = 0;
    va_list ap;
    va_start(ap, flags);
    if (flags & O_CREAT) {
        mode = (mode_t)va_arg(ap, int);
    }
    va_end(ap);
    return openat(dirfd, path, flags, mode);
}

int stat(const char* path, struct stat* buf) {
    int rc = real_stat(path, buf);
    if (watched(path)) {
        arm();
    }
    return rc;
}

int stat64(const char* path, struct stat64* buf) {
    return stat(path, (struct stat*)buf);
}

int lstat(const char* path, struct stat* buf) {
    int rc = real_lstat(path, buf);
    if (watched(path)) {
        arm();
    }
    return rc;
}

int lstat64(const char* path, struct stat64* buf) {
    return lstat(path, (struct stat*)buf);
}

int fstatat(int dirfd, const char* path, struct stat* buf, int flags) {
    int rc = real_fstatat(dirfd, path, buf, flags);
    if (watched(path)) {
        arm();
    }
    return rc;
}

int fstatat64(int dirfd, const char* path, struct stat64* buf, int flags) {
    return fstatat(dirfd, path, (struct stat*)buf, flags);
}

int __xstat(int ver, const char* path, struct stat* buf) {
    int rc = real_xstat(ver, path, buf);
    if (watched(path)) {
        arm();
    }
    return rc;
}

int __lxstat(int ver, const char* path, struct stat* buf) {
    int rc = real_lxstat(ver, path, buf);
    if (watched(path)) {
        arm();
    }
    return rc;
}

int __fxstatat(
        int ver,
        int dirfd,
        const char* path,
        struct stat* buf,
        int flags) {
    int rc = real_fxstatat(ver, dirfd, path, buf, flags);
    if (watched(path)) {
        arm();
    }
    return rc;
}

char* realpath(const char* path, char* resolved) {
    char* rc = real_realpath(path, resolved);
    if (watched(path)) {
        arm();
    }
    return rc;
}

int access(const char* path, int mode) {
    int rc = real_access(path, mode);
    if (watched(path)) {
        arm();
    }
    return rc;
}

int faccessat(int dirfd, const char* path, int mode, int flags) {
    int rc = real_faccessat(dirfd, path, mode, flags);
    if (watched(path)) {
        arm();
    }
    return rc;
}

int truncate(const char* path, off_t length) {
    int rc = real_truncate(path, length);
    if (watched(path)) {
        arm();
    }
    return rc;
}

int truncate64(const char* path, off_t length) {
    return truncate(path, length);
}

} // extern "C"
