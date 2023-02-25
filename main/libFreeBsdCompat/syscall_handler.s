    .text
    .globl _syscall_handler

_handler_message:
    .ascii "In syscall handler\12\0"

_syscall_handler:
    push %rbp
    mov %rsp, %rbp
    and $-16, %rsp
    lea _handler_message(%rip), %rdi
    call puts
    mov %rbp, %rsp
    pop %rbp
    ret
