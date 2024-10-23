import psutil
import sys

"""
(/Users/james/Documents/pytorch_test/env) james@Jamess-MacBook-Pro pytorch_test % ulimit -a
-t: cpu time (seconds)              unlimited
-f: file size (blocks)              unlimited
-d: data seg size (kbytes)          unlimited
-s: stack size (kbytes)             8176
-c: core file size (blocks)         0
-v: address space (kbytes)          unlimited
-l: locked-in-memory size (kbytes)  unlimited
-u: processes                       10666
-n: file descriptors                256
"""

"""
RuntimeError: Too many open files.
Communication with the workers is no longer possible.
Please increase the limit using `ulimit -n` in the shell or change the sharing strategy
by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code
"""

def get_python_procs():
    target_process_name = "python3.11"
    python_processes = list()
    for process in psutil.process_iter(['pid','name']):
        try:
            pname = process.info['name']
            pid   = process.info['pid'] 
            if pname == target_process_name:
                #print(f"Found process {pname} (PID: {pid}) ")
                python_processes.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return python_processes

def print_python_process_minimal( process ):
    #pid = int(sys.argv[1])
    ##pid = 12345  # Replace with the actual PID of the process
    pname = process.info['name']
    pid = process.info['pid']
    try:
        ppid = process.ppid()
        # max_fds = process.rlimit(psutil.RLIMIT_NOFILE)
        num_fds = process.num_fds()

        #process = psutil.Process(pid)
        open_files = process.open_files()

        print(f"    Python process {pname} (PID: {pid})  PPID: {ppid}  opened: {len(open_files)}  FDS: {num_fds}")

    except psutil.ZombieProcess as e:
        # Line of code: num_fds = process.num_fds()
        #psutil.ZombieProcess: PID still exists but it's a zombie (pid=93097, ppid=92266, name='python3.11')
        print(f"    Python process {pname} (PID: {pid})  (ZombieProcess unable to obtain more info)")
    #for open_file in open_files:
    #    print(f"    File: {open_file.path}, File Descriptor: {open_file.fd}")
    #print(f"  Number of opened files {len(open_files)}")

def print_python_process(process):
    tab = "    "
    pid = process.pid
    try:
        # Get the process instance
        #process = psutil.Process(pid)

        # Display basic process information
        print(f"Process ID (PID): {process.pid}")
        print(f"{tab}Process Name: {process.name()}")
        print(f"{tab}Executable Path: {process.exe()}")
        print(f"{tab}Current Working Directory: {process.cwd()}")
        #print(f"{tab}Command Line: {process.cmdline()}")
        print(f"{tab}Status: {process.status()}")
        print(f"{tab}Parent PID: {process.ppid()}")
        print(f"{tab}Parent Process: {process.parent().name()} (PID: {process.ppid()})")
        print(f"{tab}User: {process.username()}")

        # Display CPU usage
        print(f"{tab}CPU Usage (%): {process.cpu_percent(interval=1.0)}")
        print(f"{tab}CPU Times: {process.cpu_times()}")

        # Display memory usage
        print(f"{tab}Memory Usage (RSS): {process.memory_info().rss / (1024 ** 2)} MB")
        print(f"{tab}Memory Percent: {process.memory_percent()}%")
        print(f"{tab}Memory Info: {process.memory_info()}")

        # Display I/O statistics
        num_fds = process.num_fds()
        print(f"{tab}Number of file descriptors: {num_fds}")
        """
        io_counters = process.io_counters()
        print(f"Read Count: {io_counters.read_count}")
        print(f"Write Count: {io_counters.write_count}")
        print(f"Read Bytes: {io_counters.read_bytes / (1024 ** 2)} MB")
        print(f"Write Bytes: {io_counters.write_bytes / (1024 ** 2)} MB")
        """

        # Display open files
        print(f"{tab}Open Files:")
        for f in process.open_files():
            print(f"{tab}{tab}{f.path} (fd: {f.fd})")

        # Display network connections
        """
        print("{tab}Network Connections:")
        for conn in process.connections(kind='inet'):
            print(f"{tab}Laddr: {conn.laddr}, Raddr: {conn.raddr}, Status: {conn.status}")
        """

        #print( dir(process) )
        print(f"{tab}Context switches  {process.num_ctx_switches()}")

        # Display memory maps
        """
        print("Memory Maps:")
        for mmap in process.memory_maps():
            print(f"  {mmap.path} (RSS: {mmap.rss / (1024 ** 2)} MB, Size: {mmap.size / (1024 ** 2)} MB)")
        """

    except AttributeError as e:
        print(f"AttributeError: {e}")
    except psutil.ZombieProcess:
        print(f"Process {pid} is a zombie")
    except psutil.NoSuchProcess:
        print(f"No process found with PID {pid}")
    except psutil.AccessDenied:
        print(f"Access denied when trying to access PID {pid}")


def print_python_processes():
    python_processes = get_python_procs()
    for process in python_processes:
        print_python_process_minimal( process )
        #print_python_process( process )

def main():
    print_python_processes()

if __name__ == "__main__":
    main()
