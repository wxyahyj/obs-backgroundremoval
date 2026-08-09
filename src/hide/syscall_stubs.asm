; syscall_stubs.asm - x64 直接/间接 syscall stub
;
; 设计 (见 docs/worklog/罗技驱动调用与隐藏绕过反作弊hook.md):
;   - SSN 不硬编码 (随 Windows 版本变化), 运行时由 syscall_init.cpp 填入全局
;   - 优先"间接 syscall": jmp 到 ntdll 内完好的 syscall;ret 字节序列,
;     执行 syscall 时 RIP 落在 ntdll .text, 规避反作弊的 RIP 范围检查
;   - SyscallAddr 未初始化 (0) 时回退"直接 syscall": 在本 stub 内执行 syscall
;
; x64 syscall 约定: 第1参数经 r10 传递 (syscall 破坏 rcx), SSN 放 eax,
; 第2~4参数 rdx/r8/r9, 第5+参数在调用者栈帧, stub 无需操作栈。

.data

public g_NtCreateFile_SSN
g_NtCreateFile_SSN          DWORD 0
public g_NtCreateFile_SyscallAddr
g_NtCreateFile_SyscallAddr  QWORD 0

public g_NtDeviceIoControlFile_SSN
g_NtDeviceIoControlFile_SSN DWORD 0
public g_NtDeviceIoControlFile_SyscallAddr
g_NtDeviceIoControlFile_SyscallAddr QWORD 0

public g_NtClose_SSN
g_NtClose_SSN               DWORD 0
public g_NtClose_SyscallAddr
g_NtClose_SyscallAddr       QWORD 0

public g_NtOpenDirectoryObject_SSN
g_NtOpenDirectoryObject_SSN DWORD 0
public g_NtOpenDirectoryObject_SyscallAddr
g_NtOpenDirectoryObject_SyscallAddr QWORD 0

public g_NtQueryDirectoryObject_SSN
g_NtQueryDirectoryObject_SSN DWORD 0
public g_NtQueryDirectoryObject_SyscallAddr
g_NtQueryDirectoryObject_SyscallAddr QWORD 0

.code

; --- SysNtCreateFile ---
public SysNtCreateFile
SysNtCreateFile PROC
    mov     r10, rcx
    mov     eax, DWORD PTR [g_NtCreateFile_SSN]
    cmp     QWORD PTR [g_NtCreateFile_SyscallAddr], 0
    je      @F
    jmp     QWORD PTR [g_NtCreateFile_SyscallAddr]   ; 间接: 跳进 ntdll 执行 syscall;ret
@@:
    syscall                                          ; 直接: RIP 在本模块
    ret
SysNtCreateFile ENDP

; --- SysNtDeviceIoControlFile ---
public SysNtDeviceIoControlFile
SysNtDeviceIoControlFile PROC
    mov     r10, rcx
    mov     eax, DWORD PTR [g_NtDeviceIoControlFile_SSN]
    cmp     QWORD PTR [g_NtDeviceIoControlFile_SyscallAddr], 0
    je      @F
    jmp     QWORD PTR [g_NtDeviceIoControlFile_SyscallAddr]
@@:
    syscall
    ret
SysNtDeviceIoControlFile ENDP

; --- SysNtClose ---
public SysNtClose
SysNtClose PROC
    mov     r10, rcx
    mov     eax, DWORD PTR [g_NtClose_SSN]
    cmp     QWORD PTR [g_NtClose_SyscallAddr], 0
    je      @F
    jmp     QWORD PTR [g_NtClose_SyscallAddr]
@@:
    syscall
    ret
SysNtClose ENDP

; --- SysNtOpenDirectoryObject ---
public SysNtOpenDirectoryObject
SysNtOpenDirectoryObject PROC
    mov     r10, rcx
    mov     eax, DWORD PTR [g_NtOpenDirectoryObject_SSN]
    cmp     QWORD PTR [g_NtOpenDirectoryObject_SyscallAddr], 0
    je      @F
    jmp     QWORD PTR [g_NtOpenDirectoryObject_SyscallAddr]
@@:
    syscall
    ret
SysNtOpenDirectoryObject ENDP

; --- SysNtQueryDirectoryObject ---
public SysNtQueryDirectoryObject
SysNtQueryDirectoryObject PROC
    mov     r10, rcx
    mov     eax, DWORD PTR [g_NtQueryDirectoryObject_SSN]
    cmp     QWORD PTR [g_NtQueryDirectoryObject_SyscallAddr], 0
    je      @F
    jmp     QWORD PTR [g_NtQueryDirectoryObject_SyscallAddr]
@@:
    syscall
    ret
SysNtQueryDirectoryObject ENDP

END
