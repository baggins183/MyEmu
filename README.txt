Just notes for now

/////////////////////////////////////////////////////////////////////////////////////
Generate elf symbol rainbow table:
/////////////////////////////////////////////////////////////////////////////////////

$ ./nid_hash_standalone --symbols nid_hash/symbols.txt --hashdb syms.sqlite
syms.sqlite is a database to output to.

Maps symbol name -> hash found in .prx/.sprx symbol tables
example:
NAME         |   HASH
------------------------------
sysctlbyname |   MhC53TKmjVA
memcpy       |   Q3VBxCXhUHs


/////////////////////////////////////////////////////////////////////////////////////
ELF symbols / intercepting library calls
/////////////////////////////////////////////////////////////////////////////////////

Sony obfuscates the symbols (function names, global variables) by using a custom hash function.

This Emu patches the .prx/.sprx dynamic libraries (libraries dumped from the ps4) to plain .so (shared object) files 
that can be loaded on linux with dlopen().
While we patch, if we see "Q3VBxCXhUHs" in a symbol table, we know that's memcpy so we put 'memcpy' in
the patched elf instead.
That make debugging easier.

To intercept a library function call,
implement it in one of the dynamic libraries in the ps4lib_overloads folder, which will be preloaded by the emu program
and get precedence when the dynamic linker is trying to resolve a library call (dynamic relocation).

For now, "_ps4__" gets prepended to all the patched symbols (if the original name of the hash is known), so youd need to
define "_ps4__memcpy".
Then when native ps4 code tries to call _ps4__memcpy outside of its own unit (executable or dyn lib),
the host (linux) dynamic linker will resolve the symbol to our own implementation in the compatibility layer.

/////////////////////////////////////////////////////////////////////////////////////
Intercepting syscalls
/////////////////////////////////////////////////////////////////////////////////////
Currently have working syscall handlers, using "syscall user dispatch" (https://docs.kernel.org/admin-guide/syscall-user-dispatch.html)
This registers a handler for when native ps4 code executes syscall instruction, which the ps4 kernel is meant to handle.
Then we basically enter a switch statement where we can have our own handler from each
ps4 syscall number (see include/orbis/orbis_syscalls.h for the list of ps4 syscalls, which is like FreeBSD).
Syscall user dispatch needs to be toggled on/off when we "enter" or "leave" native ps4 code. When turned "off",
syscalls will pass directly to the linux kernel, which is what we want when making syscalls from our own code.

Alternatively, we could intercept library calls to the syscall "wrapper" functions,
like read(), write(), etc, throgh preload tricks.
That only works if we assume the ps4 code is calling the library wrapper and not directly
executing syscall instructions. That's probably true, but won't always work.

For example, if code in "libkernel.prx" calls the open() library function, which is implemented
in "libkernel.prx" itself, that call was already resolved when libkernel.prx was linked.
It didn't require dynamic linking, so we can't intercept it through
preloading our own "open()" implementation.

So in general, we do compatibility at the syscall level and not at the syscall wrapper function level.

/////////////////////////////////////////////////////////////////////////////////////
Compiler
/////////////////////////////////////////////////////////////////////////////////////

Plan is to translate Gcn bytecode to spir-v.
Gcn seems to all be public.
Already supported in LLVM: can disassemble gcn code found in games using
AMDGPU disassembler (with some tweaks to allow disassembling for the Bonaire arch, see Gcn/llvm_patch.diff).

Will need to turn Gcn arbitrary control flow into legal spir-v (structured?) control flow.

Need to handle weird things like predicated execution with the exec mask, ambiguity between condition codes and uniforms (which are
stored in scalar gprs).

/////////////////////////////////////////////////////////////////////////////////////
GNM (PS4 graphics library)
/////////////////////////////////////////////////////////////////////////////////////

Plan is to intercept command buffer submissions (Pm4 packets?) at the lowest level, and
translate that into vulkan commands.
So interception will happen between the ps4 graphics driver and the ps4 GPU.
as opposed to:
intercepting graphics API calls from the game program to the PS4 graphics driver.

Might be hard, but seems more thorough than intercepting Graphics API calls when the API isnt public.
The Pm4 format seems to be public (Sea islands docs?), there are a lot of AMD GPU docs available.

/////////////////////////////////////////////////////////////////////////////////////

TODO
The names like ps4lib_overloads, system_compat suck right now. Preloading the overload libraries is also wonky and should be
better

Stuff in general is messy, no effort to clean it up yet