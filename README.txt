Just notes for now

/////////////////////////////////////////////////////////////////////////////////////
Patching .prx, .sprx dynamic libraries, main executables
/////////////////////////////////////////////////////////////////////////////////////

we patch executables/dynamic libraries, dumped from the ps4, into .so files that can be loaded by the (linux) host
with dlopen().

Naming convention goes like
libkernel.prx (native/Ps4) -> libkernel.prx.so (host/linux)

Some things that get patched:
-Add ELF section headers that the linux dynamic linker needs to see
-patch obfuscated symbols (hashed by Sony) to known names, remake symbol table/string table
-remember where .init, .fini routines are
-lots of other stuff I forgot/need to revisit

TODO

We remove .init, fini sections from the ELF (which contain initializers, constructors/destructors for global variables, etc).
Normally the dynamic linker calls into the .init section when it loads an executable or dynamic library.
That causes problems with our patched .so files, we need syscall interception to be off while inside the dynamic linker (dlopen), but turned on inside
ps4 code in .init or .fini code.
So we remove the dynamic tags/section headers for .init, .fini sections, so the dynamic linker doesn't try to
do them, and manually call into the .init entry points after dlopen loading the patched elf's with dlopen().
we store info about each patched ELF, including the address of .init/.fini entry points in a json file, e.g. libkernel.prx.so.json.
we need to call the init routines in order of library dependecies. Havent really bothered yet, we should just set up a static
order for all the system libraries (libkernel.prx.so, libSceLibcInternal.prx.so, etc) and figure out a specific game's dependencies
on the fly.

/////////////////////////////////////////////////////////////////////////////////////
ELF symbols / intercepting library calls
/////////////////////////////////////////////////////////////////////////////////////

Sony obfuscates the symbols (function names, global variables) in it's executables/dynamic libraries by using a custom hash function.

The nid_hash folder has the hash function found on the internet.
The nid_hash_standalone executable can create a sqlite database with 2 columns

NAME         |   HASH
------------------------------
sysctlbyname |   MhC53TKmjVA
memcpy       |   Q3VBxCXhUHs

You need to generate it from nid_hash/symbols.txt (basically the NAME col) instead of uploading the full database with indexing to git.

$ ./nid_hash_standalone --symbols nid_hash/symbols.txt --hashdb syms.sqlite
where syms.sqlite is a name of a database to output to.

This db gets used as a rainbow table.

While we patch Ps4 ELF's, if we see "Q3VBxCXhUHs" in a symbol table, we know that's memcpy so we put 'memcpy' in
the patched elf instead.
That make debugging way easier. Since we patch the symbol tables and load it's library with dlopen(), GDB/LLDB can show callstacks with the readable names instead
of the garbage hashed names.

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
AMDGPU disassembler (with some tIaks to allow disassembling for the Bonaire arch, see Gcn/llvm_patch.diff).

Will need to turn Gcn arbitrary control flow into legal spir-v (structured?) control flow.

Need to handle Iird things like predicated execution with the exec mask, ambiguity between condition codes and uniforms (which are
stored in scalar gprs).

How to handle buffer, image, texture memory ops with Vulkan concepts (descriptors, etc).
How vertex shader outputs and pixel shader inputs get linked together.
How to fetch vertex attributes (ps4 uses subroutines?)

/////////////////////////////////////////////////////////////////////////////////////
GNM (PS4 graphics library)
/////////////////////////////////////////////////////////////////////////////////////

Plan is to intercept command buffer submissions (Pm4 packets?) at the lowest level, and
translate that into vulkan commands.
So interception will happen between the ps4 graphics driver and the ps4 GPU.
as opposed to:
intercepting graphics API calls between the game program and the PS4 graphics driver (libSceGnmDriver.prx?).

Might be hard, but seems more thorough than intercepting Graphics API calls when the API isnt public.
The Pm4 format seems to be public (Sea islands docs?), there are a lot of AMD GPU docs available.

/////////////////////////////////////////////////////////////////////////////////////
Thread local storage (TLS)
/////////////////////////////////////////////////////////////////////////////////////

TODO, been a year since looking at this, need to figure out how ps4 TLS implementation is different than
linux/bsd.
Will probably be hard

/////////////////////////////////////////////////////////////////////////////////////
TODO
/////////////////////////////////////////////////////////////////////////////////////

Say where to dump game files to, command line options, environemnt variables.

The names like ps4lib_overloads, system_compat suck right now. Preloading the overload libraries is also wonky and should be
better

Stuff in general is messy, no effort to clean it up yet

Add helpful links everywhere
(AMD docs, ELF format/dynamic linking)