#define	SCE_S_IRWXU	0000700			/* RWX mask for owner */
#define	SCE_S_IRUSR	0000400			/* R for owner */
#define	SCE_S_IWUSR	0000200			/* W for owner */
#define	SCE_S_IXUSR	0000100			/* X for owner */

#define	SCE_S_IRWXG	0000070			/* RWX mask for group */
#define	SCE_S_IRGRP	0000040			/* R for group */
#define	SCE_S_IWGRP	0000020			/* W for group */
#define	SCE_S_IXGRP	0000010			/* X for group */
#define	SCE_S_IRWXO	0000007			/* RWX mask for other */
#define	SCE_S_IROTH	0000004			/* R for other */
#define	SCE_S_IWOTH	0000002			/* W for other */
#define	SCE_S_IXOTH	0000001			/* X for other */

#define	SCE_S_IFMT	 0170000		/* type of file mask */
#define	SCE_S_IFIFO	 0010000		/* named pipe (fifo) */
#define	SCE_S_IFCHR	 0020000		/* character special */
#define	SCE_S_IFDIR	 0040000		/* directory */
#define	SCE_S_IFBLK	 0060000		/* block special */
#define	SCE_S_IFREG	 0100000		/* regular */
#define	SCE_S_IFLNK	 0120000		/* symbolic link */
#define	SCE_S_IFSOCK 0140000		/* socket */
#define	SCE_S_ISVTX	 0001000		/* save swapped text even after use */

// open mode
#define SCE_KERNEL_S_INONE         0

#define SCE_KERNEL_S_IRUSR         (SCE_S_IRUSR | SCE_S_IRGRP | SCE_S_IROTH | SCE_S_IXUSR | \
									SCE_S_IXGRP | SCE_S_IXOTH)
#define SCE_KERNEL_S_IWUSR         (SCE_S_IWUSR | SCE_S_IWGRP | SCE_S_IWOTH | SCE_S_IXUSR | \
									SCE_S_IXGRP | SCE_S_IXOTH)
#define SCE_KERNEL_S_IXUSR         (SCE_S_IXUSR | SCE_S_IXGRP | SCE_S_IXOTH)
#define SCE_KERNEL_S_IRWXU         (SCE_KERNEL_S_IRUSR | SCE_KERNEL_S_IWUSR)
// read write
#define SCE_KERNEL_S_IRWU          (SCE_KERNEL_S_IRUSR | SCE_KERNEL_S_IWUSR)
// read
#define SCE_KERNEL_S_IRU           (SCE_KERNEL_S_IRUSR)

#define SCE_KERNEL_S_IFMT          SCE_S_IFMT
#define SCE_KERNEL_S_IFDIR         SCE_S_IFDIR
#define SCE_KERNEL_S_IFREG         SCE_S_IFREG

// open flags
#define	SCE_O_RDONLY	0x0000		/* open for reading only */
#define	SCE_O_WRONLY	0x0001		/* open for writing only */
#define	SCE_O_RDWR		0x0002		/* open for reading and writing */
#define	SCE_O_ACCMODE	0x0003		/* mask for above modes */
#define	SCE_O_NONBLOCK	0x0004		/* no delay */
#define	SCE_O_APPEND	0x0008		/* set append mode */
#define	SCE_O_CREAT		0x0200		/* create if nonexistent */
#define	SCE_O_TRUNC		0x0400		/* truncate to zero length */
#define	SCE_O_EXCL		0x0800		/* error if already exists */
#define	SCE_O_DSYNC		0x1000		/* synchronous data writes(omit inode writes) */
#define	SCE_O_NOCTTY	0x8000		/* don't assign controlling terminal */
#define	SCE_O_FSYNC		0x0080		/* synchronous writes */
#define	SCE_O_SYNC		0x0080		/* POSIX synonym for O_FSYNC */
#define SCE_O_DIRECT	0x00010000  /* Attempt to bypass buffer cache */
								/* Defined by POSIX Extended API Set Part 2 */
#define	SCE_O_DIRECTORY	0x00020000	/* Fail if not directory */

#define SCE_KERNEL_O_RDONLY        O_RDONLY
#define SCE_KERNEL_O_WRONLY        O_WRONLY 
#define SCE_KERNEL_O_RDWR          O_RDWR
#define SCE_KERNEL_O_NONBLOCK      O_NONBLOCK
#define SCE_KERNEL_O_APPEND        O_APPEND
#define SCE_KERNEL_O_CREAT         O_CREAT
#define SCE_KERNEL_O_TRUNC         O_TRUNC
#define SCE_KERNEL_O_EXCL          O_EXCL
#define SCE_KERNEL_O_DIRECT        O_DIRECT
#define SCE_KERNEL_O_FSYNC         O_FSYNC
#define SCE_KERNEL_O_SYNC          O_SYNC
#define SCE_KERNEL_O_DSYNC         O_DSYNC
#define SCE_KERNEL_O_DIRECTORY     O_DIRECTORY

typedef int sce_file_flags_t;
typedef unsigned int sce_mode_t;

#include <fcntl.h>

static inline int sceFlagsToLinux(sce_file_flags_t flags) {
    int linuxFlags = 0;
    if (flags & SCE_O_RDONLY) {
        linuxFlags |= O_RDONLY;
    }
    if (flags & SCE_O_WRONLY) {
        linuxFlags |= O_WRONLY;
    }
    if (flags & SCE_O_RDWR) {
        linuxFlags |= O_RDWR;
    }
    if (flags & SCE_O_ACCMODE) {
        linuxFlags |= O_ACCMODE;
    }
    if (flags & SCE_O_NONBLOCK) {
        linuxFlags |= O_NONBLOCK;
    }
    if (flags & SCE_O_APPEND) {
        linuxFlags |= O_APPEND;
    }
    if (flags & SCE_O_CREAT) {
        linuxFlags |= O_CREAT;
    }
    if (flags & SCE_O_TRUNC) {
        linuxFlags |= O_TRUNC;
    }
    if (flags & SCE_O_EXCL) {
        linuxFlags |= O_EXCL;
    }
    if (flags & SCE_O_DSYNC) {
        linuxFlags |= O_DSYNC;
    }
    if (flags & SCE_O_NOCTTY) {
        linuxFlags |= O_NOCTTY;
    }
    if (flags & SCE_O_FSYNC) {
        linuxFlags |= O_FSYNC;
    }
    if (flags & SCE_O_SYNC) {
        linuxFlags |= O_SYNC;
    }
    if (flags & SCE_O_DIRECT) {
        linuxFlags |= O_DIRECT;
    }

    return linuxFlags;
}

static inline mode_t sceModeToLinux(sce_mode_t mode) {
    mode_t linuxMode = 0;
    if (mode & SCE_S_IRWXU) {
        linuxMode |= S_IRWXU;
    }
    if (mode & SCE_S_IRUSR) {
        linuxMode |= S_IRUSR;
    }
    if (mode & SCE_S_IWUSR) {
        linuxMode |= S_IWUSR;
    }
    if (mode & SCE_S_IXUSR) {
        linuxMode |= S_IXUSR;
    }
    if (mode & SCE_S_IRWXG) {
        linuxMode |= S_IRWXG;
    }
    if (mode & SCE_S_IRGRP) {
        linuxMode |= S_IRGRP;
    }
    if (mode & SCE_S_IWGRP) {
        linuxMode |= S_IWGRP;
    }
    if (mode & SCE_S_IXGRP) {
        linuxMode |= S_IXGRP;
    }
    if (mode & SCE_S_IRWXO) {
        linuxMode |= S_IRWXO;
    }
    if (mode & SCE_S_IROTH) {
        linuxMode |= S_IROTH;
    }
    if (mode & SCE_S_IWOTH) {
        linuxMode |= S_IWOTH;
    }
    if (mode & SCE_S_IXOTH) {
        linuxMode |= S_IXOTH;
    }
//    if (mode & SCE_S_ISUID) {
//        linux_flags |= S_ISUID;
//    }
//    if (mode & SCE_S_ISGID) {
//        linux_flags |= S_ISGID;
//    }
    if (mode & SCE_S_ISVTX) {
        linuxMode |= S_ISVTX;
    }
    return linuxMode;
}