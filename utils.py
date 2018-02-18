import platform


def platform_info():
    print("python", platform.python_version())
    print("")
    libs = ["numpy", "torch", "torchvision", "matplotlib"]
    for lib in libs:
            version = get_distribution(lib).version
            print(lib, version)
