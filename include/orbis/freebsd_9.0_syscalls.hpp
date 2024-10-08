#pragma once
#include <map>
#include <string>

#define BSD_SYS_TABLE(OP) \
	OP(SYS_syscall,	0) \
	OP(SYS_exit,	1) \
	OP(SYS_fork,	2) \
	OP(SYS_read,	3) \
	OP(SYS_write,	4) \
	OP(SYS_open,	5) \
	OP(SYS_close,	6) \
	OP(SYS_wait4,	7) \
	OP(SYS_link,	9) \
	OP(SYS_unlink,	10) \
	OP(SYS_chdir,	12) \
	OP(SYS_fchdir,	13) \
	OP(SYS_mknod,	14) \
	OP(SYS_chmod,	15) \
	OP(SYS_chown,	16) \
	OP(SYS_break,	17) \
	OP(SYS_freebsd4_getfsstat,	18) \
	OP(SYS_getpid,	20) \
	OP(SYS_mount,	21) \
	OP(SYS_unmount,	22) \
	OP(SYS_setuid,	23) \
	OP(SYS_getuid,	24) \
	OP(SYS_geteuid,	25) \
	OP(SYS_ptrace,	26) \
	OP(SYS_recvmsg,	27) \
	OP(SYS_sendmsg,	28) \
	OP(SYS_recvfrom,	29) \
	OP(SYS_accept,	30) \
	OP(SYS_getpeername,	31) \
	OP(SYS_getsockname,	32) \
	OP(SYS_access,	33) \
	OP(SYS_chflags,	34) \
	OP(SYS_fchflags,	35) \
	OP(SYS_sync,	36) \
	OP(SYS_kill,	37) \
	OP(SYS_getppid,	39) \
	OP(SYS_dup,	41) \
	OP(SYS_pipe,	42) \
	OP(SYS_getegid,	43) \
	OP(SYS_profil,	44) \
	OP(SYS_ktrace,	45) \
	OP(SYS_getgid,	47) \
	OP(SYS_getlogin,	49) \
	OP(SYS_setlogin,	50) \
	OP(SYS_acct,	51) \
	OP(SYS_sigaltstack,	53) \
	OP(SYS_ioctl,	54) \
	OP(SYS_reboot,	55) \
	OP(SYS_revoke,	56) \
	OP(SYS_symlink,	57) \
	OP(SYS_readlink,	58) \
	OP(SYS_execve,	59) \
	OP(SYS_umask,	60) \
	OP(SYS_chroot,	61) \
	OP(SYS_msync,	65) \
	OP(SYS_vfork,	66) \
	OP(SYS_sbrk,	69) \
	OP(SYS_sstk,	70) \
	OP(SYS_vadvise,	72) \
	OP(SYS_munmap,	73) \
	OP(SYS_mprotect,	74) \
	OP(SYS_madvise,	75) \
	OP(SYS_mincore,	78) \
	OP(SYS_getgroups,	79) \
	OP(SYS_setgroups,	80) \
	OP(SYS_getpgrp,	81) \
	OP(SYS_setpgid,	82) \
	OP(SYS_setitimer,	83) \
	OP(SYS_swapon,	85) \
	OP(SYS_getitimer,	86) \
	OP(SYS_getdtablesize,	89) \
	OP(SYS_dup2,	90) \
	OP(SYS_fcntl,	92) \
	OP(SYS_select,	93) \
	OP(SYS_fsync,	95) \
	OP(SYS_setpriority,	96) \
	OP(SYS_socket,	97) \
	OP(SYS_connect,	98) \
	OP(SYS_99, 99) \
	OP(SYS_getpriority,	100) \
	OP(SYS_bind,	104) \
	OP(SYS_setsockopt,	105) \
	OP(SYS_listen,	106) \
	OP(SYS_gettimeofday,	116) \
	OP(SYS_getrusage,	117) \
	OP(SYS_getsockopt,	118) \
	OP(SYS_readv,	120) \
	OP(SYS_writev,	121) \
	OP(SYS_settimeofday,	122) \
	OP(SYS_fchown,	123) \
	OP(SYS_fchmod,	124) \
	OP(SYS_setreuid,	126) \
	OP(SYS_setregid,	127) \
	OP(SYS_rename,	128) \
	OP(SYS_flock,	131) \
	OP(SYS_mkfifo,	132) \
	OP(SYS_sendto,	133) \
	OP(SYS_shutdown,	134) \
	OP(SYS_socketpair,	135) \
	OP(SYS_mkdir,	136) \
	OP(SYS_rmdir,	137) \
	OP(SYS_utimes,	138) \
	OP(SYS_adjtime,	140) \
	OP(SYS_setsid,	147) \
	OP(SYS_quotactl,	148) \
	OP(SYS_nlm_syscall,	154) \
	OP(SYS_nfssvc,	155) \
	OP(SYS_freebsd4_statfs,	157) \
	OP(SYS_freebsd4_fstatfs,	158) \
	OP(SYS_lgetfh,	160) \
	OP(SYS_getfh,	161) \
	OP(SYS_freebsd4_getdomainname,	162) \
	OP(SYS_freebsd4_setdomainname,	163) \
	OP(SYS_freebsd4_uname,	164) \
	OP(SYS_sysarch,	165) \
	OP(SYS_rtprio,	166) \
	OP(SYS_semsys,	169) \
	OP(SYS_msgsys,	170) \
	OP(SYS_shmsys,	171) \
	OP(SYS_freebsd6_pread,	173) \
	OP(SYS_freebsd6_pwrite,	174) \
	OP(SYS_setfib,	175) \
	OP(SYS_ntp_adjtime,	176) \
	OP(SYS_setgid,	181) \
	OP(SYS_setegid,	182) \
	OP(SYS_seteuid,	183) \
	OP(SYS_stat,	188) \
	OP(SYS_fstat,	189) \
	OP(SYS_lstat,	190) \
	OP(SYS_pathconf,	191) \
	OP(SYS_fpathconf,	192) \
	OP(SYS_getrlimit,	194) \
	OP(SYS_setrlimit,	195) \
	OP(SYS_getdirentries,	196) \
	OP(SYS_freebsd6_mmap,	197) \
	OP(SYS___syscall,	198) \
	OP(SYS_freebsd6_lseek,	199) \
	OP(SYS_freebsd6_truncate,	200) \
	OP(SYS_freebsd6_ftruncate,	201) \
	OP(SYS___sysctl,	202) \
	OP(SYS_mlock,	203) \
	OP(SYS_munlock,	204) \
	OP(SYS_undelete,	205) \
	OP(SYS_futimes,	206) \
	OP(SYS_getpgid,	207) \
	OP(SYS_poll,	209) \
	OP(SYS_freebsd7___semctl,	220) \
	OP(SYS_semget,	221) \
	OP(SYS_semop,	222) \
	OP(SYS_freebsd7_msgctl,	224) \
	OP(SYS_msgget,	225) \
	OP(SYS_msgsnd,	226) \
	OP(SYS_msgrcv,	227) \
	OP(SYS_shmat,	228) \
	OP(SYS_freebsd7_shmctl,	229) \
	OP(SYS_shmdt,	230) \
	OP(SYS_shmget,	231) \
	OP(SYS_clock_gettime,	232) \
	OP(SYS_clock_settime,	233) \
	OP(SYS_clock_getres,	234) \
	OP(SYS_ktimer_create,	235) \
	OP(SYS_ktimer_delete,	236) \
	OP(SYS_ktimer_settime,	237) \
	OP(SYS_ktimer_gettime,	238) \
	OP(SYS_ktimer_getoverrun,	239) \
	OP(SYS_nanosleep,	240) \
	OP(SYS_ntp_gettime,	248) \
	OP(SYS_minherit,	250) \
	OP(SYS_rfork,	251) \
	OP(SYS_openbsd_poll,	252) \
	OP(SYS_issetugid,	253) \
	OP(SYS_lchown,	254) \
	OP(SYS_aio_read,	255) \
	OP(SYS_aio_write,	256) \
	OP(SYS_lio_listio,	257) \
	OP(SYS_getdents,	272) \
	OP(SYS_lchmod,	274) \
	OP(SYS_netbsd_lchown,	275) \
	OP(SYS_lutimes,	276) \
	OP(SYS_netbsd_msync,	277) \
	OP(SYS_nstat,	278) \
	OP(SYS_nfstat,	279) \
	OP(SYS_nlstat,	280) \
	OP(SYS_preadv,	289) \
	OP(SYS_pwritev,	290) \
	OP(SYS_freebsd4_fhstatfs,	297) \
	OP(SYS_fhopen,	298) \
	OP(SYS_fhstat,	299) \
	OP(SYS_modnext,	300) \
	OP(SYS_modstat,	301) \
	OP(SYS_modfnext,	302) \
	OP(SYS_modfind,	303) \
	OP(SYS_kldload,	304) \
	OP(SYS_kldunload,	305) \
	OP(SYS_kldfind,	306) \
	OP(SYS_kldnext,	307) \
	OP(SYS_kldstat,	308) \
	OP(SYS_kldfirstmod,	309) \
	OP(SYS_getsid,	310) \
	OP(SYS_setresuid,	311) \
	OP(SYS_setresgid,	312) \
	OP(SYS_aio_return,	314) \
	OP(SYS_aio_suspend,	315) \
	OP(SYS_aio_cancel,	316) \
	OP(SYS_aio_error,	317) \
	OP(SYS_oaio_read,	318) \
	OP(SYS_oaio_write,	319) \
	OP(SYS_olio_listio,	320) \
	OP(SYS_yield,	321) \
	OP(SYS_mlockall,	324) \
	OP(SYS_munlockall,	325) \
	OP(SYS___getcwd,	326) \
	OP(SYS_sched_setparam,	327) \
	OP(SYS_sched_getparam,	328) \
	OP(SYS_sched_setscheduler,	329) \
	OP(SYS_sched_getscheduler,	330) \
	OP(SYS_sched_yield,	331) \
	OP(SYS_sched_get_priority_max,	332) \
	OP(SYS_sched_get_priority_min,	333) \
	OP(SYS_sched_rr_get_interval,	334) \
	OP(SYS_utrace,	335) \
	OP(SYS_freebsd4_sendfile,	336) \
	OP(SYS_kldsym,	337) \
	OP(SYS_jail,	338) \
	OP(SYS_nnpfs_syscall,	339) \
	OP(SYS_sigprocmask,	340) \
	OP(SYS_sigsuspend,	341) \
	OP(SYS_freebsd4_sigaction,	342) \
	OP(SYS_sigpending,	343) \
	OP(SYS_freebsd4_sigreturn,	344) \
	OP(SYS_sigtimedwait,	345) \
	OP(SYS_sigwaitinfo,	346) \
	OP(SYS___acl_get_file,	347) \
	OP(SYS___acl_set_file,	348) \
	OP(SYS___acl_get_fd,	349) \
	OP(SYS___acl_set_fd,	350) \
	OP(SYS___acl_delete_file,	351) \
	OP(SYS___acl_delete_fd,	352) \
	OP(SYS___acl_aclcheck_file,	353) \
	OP(SYS___acl_aclcheck_fd,	354) \
	OP(SYS_extattrctl,	355) \
	OP(SYS_extattr_set_file,	356) \
	OP(SYS_extattr_get_file,	357) \
	OP(SYS_extattr_delete_file,	358) \
	OP(SYS_aio_waitcomplete,	359) \
	OP(SYS_getresuid,	360) \
	OP(SYS_getresgid,	361) \
	OP(SYS_kqueue,	362) \
	OP(SYS_kevent,	363) \
	OP(SYS_extattr_set_fd,	371) \
	OP(SYS_extattr_get_fd,	372) \
	OP(SYS_extattr_delete_fd,	373) \
	OP(SYS___setugid,	374) \
	OP(SYS_eaccess,	376) \
	OP(SYS_afs3_syscall,	377) \
	OP(SYS_nmount,	378) \
	OP(SYS___mac_get_proc,	384) \
	OP(SYS___mac_set_proc,	385) \
	OP(SYS___mac_get_fd,	386) \
	OP(SYS___mac_get_file,	387) \
	OP(SYS___mac_set_fd,	388) \
	OP(SYS___mac_set_file,	389) \
	OP(SYS_kenv,	390) \
	OP(SYS_lchflags,	391) \
	OP(SYS_uuidgen,	392) \
	OP(SYS_sendfile,	393) \
	OP(SYS_mac_syscall,	394) \
	OP(SYS_getfsstat,	395) \
	OP(SYS_statfs,	396) \
	OP(SYS_fstatfs,	397) \
	OP(SYS_fhstatfs,	398) \
	OP(SYS_ksem_close,	400) \
	OP(SYS_ksem_post,	401) \
	OP(SYS_ksem_wait,	402) \
	OP(SYS_ksem_trywait,	403) \
	OP(SYS_ksem_init,	404) \
	OP(SYS_ksem_open,	405) \
	OP(SYS_ksem_unlink,	406) \
	OP(SYS_ksem_getvalue,	407) \
	OP(SYS_ksem_destroy,	408) \
	OP(SYS___mac_get_pid,	409) \
	OP(SYS___mac_get_link,	410) \
	OP(SYS___mac_set_link,	411) \
	OP(SYS_extattr_set_link,	412) \
	OP(SYS_extattr_get_link,	413) \
	OP(SYS_extattr_delete_link,	414) \
	OP(SYS___mac_execve,	415) \
	OP(SYS_sigaction,	416) \
	OP(SYS_sigreturn,	417) \
	OP(SYS_getcontext,	421) \
	OP(SYS_setcontext,	422) \
	OP(SYS_swapcontext,	423) \
	OP(SYS_swapoff,	424) \
	OP(SYS___acl_get_link,	425) \
	OP(SYS___acl_set_link,	426) \
	OP(SYS___acl_delete_link,	427) \
	OP(SYS___acl_aclcheck_link,	428) \
	OP(SYS_sigwait,	429) \
	OP(SYS_thr_create,	430) \
	OP(SYS_thr_exit,	431) \
	OP(SYS_thr_self,	432) \
	OP(SYS_thr_kill,	433) \
	OP(SYS__umtx_lock,	434) \
	OP(SYS__umtx_unlock,	435) \
	OP(SYS_jail_attach,	436) \
	OP(SYS_extattr_list_fd,	437) \
	OP(SYS_extattr_list_file,	438) \
	OP(SYS_extattr_list_link,	439) \
	OP(SYS_ksem_timedwait,	441) \
	OP(SYS_thr_suspend,	442) \
	OP(SYS_thr_wake,	443) \
	OP(SYS_kldunloadf,	444) \
	OP(SYS_audit,	445) \
	OP(SYS_auditon,	446) \
	OP(SYS_getauid,	447) \
	OP(SYS_setauid,	448) \
	OP(SYS_getaudit,	449) \
	OP(SYS_setaudit,	450) \
	OP(SYS_getaudit_addr,	451) \
	OP(SYS_setaudit_addr,	452) \
	OP(SYS_auditctl,	453) \
	OP(SYS__umtx_op,	454) \
	OP(SYS_thr_new,	455) \
	OP(SYS_sigqueue,	456) \
	OP(SYS_kmq_open,	457) \
	OP(SYS_kmq_setattr,	458) \
	OP(SYS_kmq_timedreceive,	459) \
	OP(SYS_kmq_timedsend,	460) \
	OP(SYS_kmq_notify,	461) \
	OP(SYS_kmq_unlink,	462) \
	OP(SYS_abort2,	463) \
	OP(SYS_thr_set_name,	464) \
	OP(SYS_aio_fsync,	465) \
	OP(SYS_rtprio_thread,	466) \
	OP(SYS_sctp_peeloff,	471) \
	OP(SYS_sctp_generic_sendmsg,	472) \
	OP(SYS_sctp_generic_sendmsg_iov,	473) \
	OP(SYS_sctp_generic_recvmsg,	474) \
	OP(SYS_pread,	475) \
	OP(SYS_pwrite,	476) \
	OP(SYS_mmap,	477) \
	OP(SYS_lseek,	478) \
	OP(SYS_truncate,	479) \
	OP(SYS_ftruncate,	480) \
	OP(SYS_thr_kill2,	481) \
	OP(SYS_shm_open,	482) \
	OP(SYS_shm_unlink,	483) \
	OP(SYS_cpuset,	484) \
	OP(SYS_cpuset_setid,	485) \
	OP(SYS_cpuset_getid,	486) \
	OP(SYS_cpuset_getaffinity,	487) \
	OP(SYS_cpuset_setaffinity,	488) \
	OP(SYS_faccessat,	489) \
	OP(SYS_fchmodat,	490) \
	OP(SYS_fchownat,	491) \
	OP(SYS_fexecve,	492) \
	OP(SYS_fstatat,	493) \
	OP(SYS_futimesat,	494) \
	OP(SYS_linkat,	495) \
	OP(SYS_mkdirat,	496) \
	OP(SYS_mkfifoat,	497) \
	OP(SYS_mknodat,	498) \
	OP(SYS_openat,	499) \
	OP(SYS_readlinkat,	500) \
	OP(SYS_renameat,	501) \
	OP(SYS_symlinkat,	502) \
	OP(SYS_unlinkat,	503) \
	OP(SYS_posix_openpt,	504) \
	OP(SYS_gssd_syscall,	505) \
	OP(SYS_jail_get,	506) \
	OP(SYS_jail_set,	507) \
	OP(SYS_jail_remove,	508) \
	OP(SYS_closefrom,	509) \
	OP(SYS___semctl,	510) \
	OP(SYS_msgctl,	511) \
	OP(SYS_shmctl,	512) \
	OP(SYS_lpathconf,	513) \
	OP(SYS_cap_new,	514) \
	OP(SYS_cap_getrights,	515) \
	OP(SYS_cap_enter,	516) \
	OP(SYS_cap_getmode,	517) \
	OP(SYS_pdfork,	518) \
	OP(SYS_pdkill,	519) \
	OP(SYS_pdgetpid,	520) \
	OP(SYS_pselect,	522) \
	OP(SYS_getloginclass,	523) \
	OP(SYS_setloginclass,	524) \
	OP(SYS_rctl_get_racct,	525) \
	OP(SYS_rctl_get_rules,	526) \
	OP(SYS_rctl_get_limits,	527) \
	OP(SYS_rctl_add_rule,	528) \
	OP(SYS_rctl_remove_rule,	529) \
	OP(SYS_posix_fallocate,	530) \
	OP(SYS_MAXSYSCALL,	532) \
	OP(SYS_538, 538) \
	OP(SYS_549, 549) \
	OP(SYS_550, 550) \
	OP(SYS_586, 586) \
	OP(SYS_587, 587) \
	OP(SYS_588, 588) \
	OP(SYS_598, 598) \
	OP(SYS_601, 601) \
	OP(SYS_602, 602) \
	OP(SYS_610, 610) \
	OP(SYS_612, 612)

enum OrbisSyscallNr {
#define ENUM_OP(name, value) name = value,
	BSD_SYS_TABLE(ENUM_OP)
#undef ENUM_OP
};

static std::map<OrbisSyscallNr, const char *> bsd_syscall_strings_map = {
#define STRING_OP(name, value) { name, #name },
	BSD_SYS_TABLE(STRING_OP)
#undef STRING_OP
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static std::string to_string(OrbisSyscallNr nr) {
	const auto &it = bsd_syscall_strings_map.find(nr);
	if (it == bsd_syscall_strings_map.end()) {
		return std::string("UNKNOWN_SYSCALL ") + "(" + std::to_string((uint) nr) + ")";
	} else {
		return it->second;
	}
}
#pragma GCC diagnostic pop // -Wunused-function