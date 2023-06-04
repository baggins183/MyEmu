#define	BSD_MAP_SHARED	0x0001		/* share changes */
#define	BSD_MAP_PRIVATE	0x0002		/* changes are private */
#define	BSD_MAP_COPY	BSD_MAP_PRIVATE	/* Obsolete */
#define	BSD_MAP_FIXED	 0x0010	/* map addr must be exactly as requested */
#define	BSD_MAP_RENAME	 0x0020	/* Sun: rename private pages to file */
#define	BSD_MAP_NORESERVE	 0x0040	/* Sun: don't reserve needed swap area */
#define	BSD_MAP_RESERVED0080 0x0080	/* previously misimplemented BSD_MAP_INHERIT */
#define	BSD_MAP_RESERVED0100 0x0100	/* previously unimplemented BSD_MAP_NOEXTEND */
#define	BSD_MAP_HASSEMAPHORE 0x0200	/* region may contain semaphores */
#define	BSD_MAP_STACK	 0x0400	/* region grows down, like a stack */
#define	BSD_MAP_NOSYNC	 0x0800 /* page to but do not sync underlying file */
#define	BSD_MAP_FILE	 0x0000	/* map from file (default) */
#define	BSD_MAP_ANON	 0x1000	/* allocated from memory, swap space */
#define	BSD_MAP_ANONYMOUS	 BSD_MAP_ANON /* For compatibility. */